import json
import os
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import os, re
import numpy as np
import cv2
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from termcolor import colored


from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import random

from .scene_transform import get_boundingbox

from torchvision import transforms as T

from ..model.encoder.dust3r.utils.image import load_images
from ..model.encoder.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, depthmap_to_canonical_camera_coordinates

from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat, mat2quat

from .ray_utils import build_rays

random.seed(0)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_monoData(path):
    mono = np.load(path)
    if len(mono.shape) == 4:
        mono = mono[0]
    elif len(mono.shape) == 2:
        mono = mono[None]
    return mono

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


@dataclass
class DatasetDTUCfg(DatasetCfgCommon):
    name: Literal['dtu']
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    pair_filepath: str
    split_filepath: list[Path]
    n_views: int
    view_selection_type: Literal['random', 'best']
    test_ref_views: list[int]
    mesh_ref_views: list[int]
    test_context_views: list[int]
    test_target_views: list[int]
    use_test_ref_views_as_src: bool
    single_view: bool
    padding_size: int = 0 
    shift_rays_half_pixel: str = "forward"


class DatasetDTU(IterableDataset):
    cfg: DatasetDTUCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor 
    chunks: list[Path] #* List of paths to chunks.
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetDTUCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.normalize = tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if self.cfg.padding_size != 0:
            self.pad_border_fn = tf.Pad((self.cfg.padding_size, self.cfg.padding_size))
        else:
            self.pad_border_fn = None
        
        self.pair_filepath = self.cfg.pair_filepath
        self.split_filepath = self.cfg.split_filepath
        self.num_all_imgs = 49
        
        
        print(colored("loading all scenes together", 'red'))
        for splitpath in self.cfg.split_filepath:
                # Load the root's index.
                with (splitpath / self.data_stage).with_suffix('.txt').open("r") as f:
                    self.scans = [line.rstrip() for line in f.readlines()]
                    
        if self.cfg.overfit_to_scene is not None:
            self.scans = [self.cfg.overfit_to_scene]
        
        
        self.all_intrinsics = []  # the cam info of the whole scene
        self.all_intrinsics_org_scale = []
        self.all_extrinsics = []
        self.all_near_fars = []
        
        self.chunks, self.ref_src_pairs = self.build_metas()  # load ref-srcs view pairs info of the scene
        
        
        self.allview_ids = [i for i in range(self.num_all_imgs)]
        self.load_cam_info()
        self.build_remap()
        self.define_transforms()
        self.to_tensor = tf.ToTensor()

    
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train"):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        #* iterate over all chunks
        for idx, meta in enumerate(self.chunks):
            # Load the chunk.
            scan, light_idx, ref_view, src_views = meta
            
            if self.stage == 'train':
                view_ids = [ref_view] + src_views[:self.cfg.n_views]
            
            elif self.stage=='val' or self.stage=='test': 
                view_ids = self.cfg.test_context_views + self.cfg.test_target_views
                assert self.cfg.view_sampler.num_context_views == len(self.cfg.test_context_views), "Number of context views does not match the length of test context views"
                ref_view = self.cfg.test_context_views[0]
                
            w2c_ref = self.all_extrinsics[self.remap[ref_view]]
            w2c_ref_inv = np.linalg.inv(w2c_ref)

            imgs, imgs_norm, depths_h, depths_mvs_h, masks = [], [], [], [], []
            intrinsics, w2cs, near_fars = [], [], []
            intrinsics_org_scale = []
            monoNs = []
            #TODO: duster code
            img_paths = []
            
            #* each scene
            proj_matrices = []
            for i, vid in enumerate(view_ids):
                # NOTE that the id in image file names is from 1 to 49 (not 0~48)
                img_filename = os.path.join(str(self.cfg.roots[0]),
                                            f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
                depth_filename = os.path.join(str(self.cfg.roots[0]),
                                            f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')
                
                mask_filename = os.path.join(str(self.cfg.roots[0]),
                                             f'Masks/{scan}_train/mask_{vid:04d}.png')
                
                normal_filename = os.path.join(str(self.cfg.roots[0]),
                                            f'Rectified/{scan}_train/normal/rect_{vid + 1:03d}_{light_idx}_r5000_normal.npy')
                
                img_src = Image.open(img_filename) 
                img = self.transform(img_src)
                img_norm = self.transform_norm(img_src)
                
                imgs += [img]
                imgs_norm += [img_norm]
                
                #TODO: duster code
                img_paths += [img_filename]
                
                masks += [self.transform(Image.open(mask_filename).convert('L'))]
                
                monoNs += [self.transform(read_monoData(normal_filename).transpose(1, 2, 0))]
                
                index_mat = self.remap[vid]
                near_fars.append(self.all_near_fars[index_mat])
                intrinsics.append(self.all_intrinsics[index_mat])
                intrinsics_org_scale.append(self.all_intrinsics_org_scale[index_mat]) 
                w2cs.append(self.all_extrinsics[index_mat] @ w2c_ref_inv) #* reference view to source view
                # w2cs.append(self.all_extrinsics[index_mat])
                
                if os.path.exists(depth_filename):
                    depth_h = self.read_depth(depth_filename)
                    depths_h += [depth_h]


            scale_mat, scale_factor = self.cal_scale_mat(img_hw=[self.cfg.image_shape[0], self.cfg.image_shape[1]],
                                                     intrinsics=intrinsics_org_scale, extrinsics=w2cs,
                                                     near_fars=near_fars, factor=1.1)
            new_near_fars = []
            new_w2cs = []
            new_c2ws = []
            new_depths_h = []
            new_qs = [] #* quaternion
            new_rs = [] #* 3x3 rotation matrix
            new_ts = [] #* translation vector
            new_cs = [] #* camera origin
            
            for i, (intrinsic, extrinsic, depth) in enumerate(zip(intrinsics_org_scale, w2cs, depths_h)):
            
                P = intrinsic @ extrinsic @ scale_mat # perspective matrix scaled by scale_mat
                P = P[:3, :4]
                c2w = load_K_Rt_from_P(None, P)[1] #* camera to world

                camera_o = c2w[:3, 3].copy() #* camera origin
                
                if i == 0:
                    camera_o_canonical = camera_o.copy()
                
                # translate the camera to make the first camera located at the origin
                c2w[:3, 3] -= camera_o_canonical
                
                new_cs.append(camera_o)
                
                w2c = np.linalg.inv(c2w)
                new_w2cs.append(w2c)
                new_c2ws.append(c2w)
                
                new_qs.append(mat2quat(w2c[:3, :3]))
                new_rs.append(c2w[:3, :3].copy())
                new_ts.append(c2w[:3, 3].copy())
                
                dist = np.sqrt(np.sum(camera_o ** 2))
                near = dist - 1 if dist > 1 else 0.1
                far = dist + 1
                new_near_fars.append([0.95 * near, 1.05 * far])
                new_depths_h.append(depth * scale_factor)
            
            q1, t1 = new_qs[0], new_cs[0]
            # quaternion and translation vector that transforms World-to-Cam
            q2, t2 = new_qs[1], new_cs[1]
            c1 = rotate_vector(-t1, qinverse(q1))  # center of camera 1 in world coordinates)
            c2 = rotate_vector(-t2, qinverse(q2))  # center of camera 2 in world coordinates)

            # get 4 x 4 relative pose transformation matrix (from im1 to im2)
            # for val set, q1,t1 is the identity pose, so the relative pose matches the absolute pose
            q12 = qmult(q2, qinverse(q1)) # meaning: q12 * q1 = q2
            t12 = t2 - rotate_vector(t1, q12) # mening: t2 = t12 + q12 * t1
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = quat2mat(q12)
            T[:3, -1] = t12
            T = torch.from_numpy(T)

            new_rs = torch.from_numpy(np.stack(new_rs).astype(np.float32))
            new_ts = torch.from_numpy(np.stack(new_ts).astype(np.float32))

            imgs_norm = torch.stack(imgs_norm)
            imgs = torch.stack(imgs)
            masks = torch.stack(masks)
            depths_h = np.stack(new_depths_h)
            
            intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), np.stack(new_near_fars)
            intrinsics_org_scale = np.stack(intrinsics_org_scale)
            
            focal_lengths = repeat(intrinsics[0, :2, :2].diagonal(), 'xy -> b xy', b=len(intrinsics))
            
            monoNs = np.stack(monoNs)
            start_idx = 0
            
            #! build rays
            rays = build_rays(c2ws, intrinsics[..., :3, :3].copy(), self.cfg.image_shape[0], self.cfg.image_shape[1], 1.0)
            rays_down = build_rays(c2ws, intrinsics[..., :3, :3].copy(), self.cfg.image_shape[0], self.cfg.image_shape[1], 1.0/4)
            
            #* Pointmap
            pts3d_all = []
            valid_mask_all = []
            canonical_id = 0
            for new_depth, int_org, new_c2w in zip(new_depths_h, intrinsics_org_scale, new_c2ws):
                # pts3d, valid_mask = depthmap_to_canonical_camera_coordinates(depthmap=new_depth, camera_intrinsics=int_org[:3, :3], source_camera_pose=new_c2w, canonical_camera_pose=new_c2ws[0]) #* depthmap (HxW), camera_intrinsics (3x3), camera_pose (4x4)
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap=new_depth, camera_intrinsics=int_org[:3, :3], camera_pose=new_c2w)
                pts3d_all.append(pts3d)
                valid_mask_all.append(valid_mask)
            

            pts3d_all = np.stack(pts3d_all)
            valid_mask_all = np.stack(valid_mask_all)
            
            # to tensor
            intrinsics = torch.from_numpy(intrinsics.astype(np.float32)).float()
            intrinsics_org_scale = torch.from_numpy(intrinsics_org_scale.astype(np.float32)).float()
            w2cs = torch.from_numpy(w2cs.astype(np.float32)).float()
            c2ws = torch.from_numpy(c2ws.astype(np.float32)).float()
            near_fars = torch.from_numpy(near_fars.astype(np.float32)).float()
            depths_h = torch.from_numpy(depths_h.astype(np.float32)).float()
            monoNs = torch.from_numpy(monoNs.astype(np.float32)).float()
            
            pts3d_all = torch.from_numpy(pts3d_all.astype(np.float32)).float()
            valid_mask_all = torch.from_numpy(valid_mask_all.astype(np.float32)).bool()
            
            # context_indices, target_indices = np.array([i for i in range(len(src_views))]), np.array([0]) 
            context_indices, target_indices = np.array([i for i in range(self.cfg.view_sampler.num_context_views)]), np.array([i for i in range(self.cfg.view_sampler.num_context_views, len(view_ids))])
            context_images, target_images = imgs[context_indices], imgs[target_indices]
            context_images_norm, target_images_norm = imgs_norm[context_indices], imgs_norm[target_indices]
            context_masks, target_masks = masks[context_indices], masks[target_indices]
            
            # set all c2ws to be equal to the context's  first (0-th index) matrix
            c2ws_gt = c2ws.clone()
            if self.cfg.single_view:
                c2ws[1:context_indices.max()+1] = c2ws[0]
            
            # Skip the example if the images don't have the right shape.
            context_image_invalid = context_images.shape[1:] != (3, self.cfg.image_shape[0], self.cfg.image_shape[1])
            target_image_invalid = target_images.shape[1:] != (3, self.cfg.image_shape[0], self.cfg.image_shape[1])
            if context_image_invalid or target_image_invalid:
                print(
                    f"Skipped bad example {scan}. Context shape was "
                    f"{context_images.shape} and target shape was "
                    f"{target_images.shape}."
                )
                continue
            
            # Resize the world to make the baseline 1.
            context_extrinsics = c2ws[context_indices]
            if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                a, b = context_extrinsics[:, :3, 3]
                scale = (a - b).norm()
                if scale < self.cfg.baseline_epsilon:
                    print(
                        f"Skipped {scan} because of insufficient baseline "
                        f"{scale:.6f}"
                    )
                    continue
                c2ws[:, :3, 3] /= scale
            else:
                scale = 1

            example = {
                "context": {
                    "extrinsics": c2ws[context_indices], #* B x 4 x 4
                    "extrinsics_gt": c2ws_gt[context_indices], #* B x 4 x 4
                    "intrinsics": intrinsics[context_indices][..., :3, :3], #* B x 3 x 3
                    "image": context_images, #* B x 3 x H x W
                    "image_normalized": context_images_norm, #* B x 3 x H x
                    "depth": depths_h[context_indices], #* B x H x W
                    "mask": context_masks, #* B x 3 x H x W
                    # "near": self.get_bound("near", len(context_indices)) / scale,
                    # "far": self.get_bound("far", len(context_indices)) / scale,
                    "near": near_fars[context_indices][:, 0], #* B
                    "far": near_fars[context_indices][:, 1], #* B
                    "near_canonical": 1.0,
                    "far_canonical": 3.4,
                    "index": context_indices,
                    "view_ids": [view_ids[i] for i in context_indices],
                    "mono": monoNs[context_indices], #* B x 3 x H x W
                    "pts3d": pts3d_all[context_indices], #* B x H x W x 3
                    "valid_mask": valid_mask_all[context_indices], #* B x H x W
                    "R": new_rs[context_indices], #* B x 3 x 3
                    "T": new_ts[context_indices], #* B x 3
                    "focal_length": focal_lengths[context_indices], #* B x 2
                    "rays": rays[context_indices], #* B x 3 x
                    "rays_down": rays_down[context_indices], #* B x 3 x
                    "padding_size": self.cfg.padding_size,
                    
                    #! MicKey Specific
                    "image0": context_images[0], #* 3 x H x W
                    "image1": context_images[1], #* 3 x H x W
                    'T_0to1': T,
                    'abs_q_0': q1,
                    'abs_c_0': c1, # (3,)
                    'abs_q_1': q2, # (4,)
                    'abs_c_1': c2, # (3,)
                    'K_color0': intrinsics_org_scale[0][:3, :3],  # (3, 3)
                    'Kori_color0': intrinsics_org_scale[0][:3, :3],  # (3, 3)
                    'K_color1': intrinsics_org_scale[1][:3, :3],  # (3, 3)
                    'Kori_color1': intrinsics_org_scale[1][:3, :3],  # (3, 3)
                    'dataset_name': 'DTU',
                    'scene': scan,
                    
                    
                },
                "target": {
                    "extrinsics": c2ws[target_indices],
                    "extrinsics_gt": c2ws_gt[target_indices], #* B x 4 x 4
                    "intrinsics": intrinsics[target_indices][..., :3, :3],
                    "image": target_images,
                    "image_normalized": target_images_norm,
                    "depth": depths_h[target_indices],
                    "mask": target_masks,
                    # "near": self.get_bound("near", len(target_indices)) / scale,
                    # "far": self.get_bound("far", len(target_indices)) / scale,
                    "near": near_fars[target_indices][:, 0],
                    "far": near_fars[target_indices][:, 1],
                    "index": target_indices,
                    "view_ids": [view_ids[i] for i in target_indices],
                    "mono": monoNs[target_indices],
                    "pts3d": pts3d_all[target_indices],
                    "valid_mask": valid_mask_all[target_indices],
                    "R": new_rs[target_indices],
                    "T": new_ts[target_indices],
                    "focal_length": focal_lengths[target_indices],
                    "rays": rays[target_indices],
                    "rays_down": rays_down[target_indices],
                    "padding_size": self.cfg.padding_size,
                    
                },
                "scene": scan, #* string for scene name
                "scale_mat": scale_mat,
                "scale_factor": scale_factor,
            }
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index
    
    def build_metas(self):
        """
        This function build metas 
        Returns:
            _type_:
        """
        
        metas = []
        ref_src_pairs = {} # referece view와 source view의 pair를 만든다.
        light_idxs = [3]

        with open(self.pair_filepath) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                ref_src_pairs[ref_view] = src_views

        for light_idx in light_idxs: #* test 인 경우 light_idx는 3만 사용
            for scan in self.scans:  #* test인 경우 15개의 scenes
                with open(self.pair_filepath) as f:
                    num_viewpoint = int(f.readline())
                    
                    #! in test stage
                    if self.stage == "test" and len(self.cfg.test_context_views) > 0:
                        context_views = self.cfg.test_context_views
                        assert len(self.cfg.test_target_views) > 0
                        target_views = self.cfg.test_target_views
                        metas += [(scan, light_idx, context_views, target_views)]
                        continue
                    
                    #! in training stage
                    for _ in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                        
                        #* implement random pair selection
                        if self.cfg.view_selection_type == 'random':
                            # indices = [i for i in range(49) if i != ref_view]
                            # src_views = random.sample(indices, self.cfg.n_views-1)
                            random.shuffle(src_views)
                            
                        elif self.cfg.view_selection_type == 'best':
                            pass
                        else:
                            raise NotImplementedError
                        
                        metas += [(scan, light_idx, ref_view, src_views)] # scan, light_idx, ref_view, src_views


        return metas, ref_src_pairs

    def load_cam_info(self):
        for vid in range(self.num_all_imgs):
            proj_mat_filename = os.path.join(str(self.cfg.roots[0]),
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            # intrinsic[:2] *= 4  # * the provided intrinsics is 4x downsampled, now keep the same scale with image
            #TODO: normalize intrinsic by dividing first row with image width and second row with image height
            
            scale_x = self.cfg.image_shape[1] / self.cfg.original_image_shape[1]
            scale_y = self.cfg.image_shape[0] / self.cfg.original_image_shape[0]

            intrinsic[0, :] *= scale_x
            intrinsic[1, :] *= scale_y
            
            self.all_intrinsics_org_scale.append(intrinsic.copy())
            
            intrinsic[:1] /= self.cfg.image_shape[1]  #* the width of the image
            intrinsic[1:2] /= self.cfg.image_shape[0]   #* the height of the image
            
            # near far values should be scaled according to the intrinsic values
            # near_far[0] /= np.max(self.cfg.image_shape)
            # near_far[1] /= np.max(self.cfg.image_shape)
            
            self.all_intrinsics.append(intrinsic)
            
            self.all_extrinsics.append(extrinsic)
            self.all_near_fars.append(near_far)
        
        self.all_intrinsics_debug = self.all_intrinsics.copy()
        self.all_extrinsics_debug = self.all_extrinsics.copy()
    
    
    def read_cam_file(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        
        # TODO: check the validity of the camera space
        
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = depth_min + float(lines[11].split()[1]) * 192
        
        self.depth_min = depth_min
        self.depth_interval = float(lines[11].split()[1]) * 1.06
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics

        return intrinsics_, extrinsics, [depth_min, depth_max]
    
    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640) 
        
        # scale down 4x
        # depth_h = cv2.resize(depth_h, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST) # (128, 160)
        
        #* resize to the target size with cv2
        depth_h = cv2.resize(depth_h, (self.cfg.image_shape[1], self.cfg.image_shape[0]), interpolation=cv2.INTER_NEAREST) 
        
        return depth_h
    
    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)

        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()
    
    def build_remap(self):
        self.remap = np.zeros(np.max(self.allview_ids) + 1).astype('int')
        for i, item in enumerate(self.allview_ids):
            self.remap[item] = i
            
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(), T.Resize((self.cfg.image_shape[0], self.cfg.image_shape[1]))])     
        self.transform_norm = T.Compose([T.ToTensor(), T.Resize((self.cfg.image_shape[0], self.cfg.image_shape[1])), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            
                                           
    
    def __len__(self) -> int:
        # return len(self.index.keys())
        return len(self.chunks)
