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

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import random

from .scene_transform import get_boundingbox
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import InterpolationMode
from torchvision import transforms as T

from ..model.encoder.dust3r.utils.image import load_images
from ..model.encoder.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, depthmap_to_canonical_camera_coordinates

from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat, mat2quat

from .ray_utils import build_rays

from pytorch3d.renderer import PerspectiveCameras
from ..model.ray_diffusion.utils.normalize import normalize_cameras_batch

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
class DatasetBlendedMVSCfg(DatasetCfgCommon):
    name: Literal['blendedmvs']
    root_dir: str
    scan_id: list[str]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    n_views: int
    view_selection_type: Literal['random', 'best']
    test_context_views: list[int]
    test_target_views: list[int]
    test_ref_view: list[int]
    use_test_ref_views_as_src: bool
    single_view: bool
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = False
    padding_size: int = 0 
    shift_rays_half_pixel: str = "forward"
    generic_near_far: bool = False
    use_mask: bool = True


class DatasetBlendedMVS(IterableDataset):
    cfg: DatasetBlendedMVSCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor 
    chunks: list[Path] #* List of paths to chunks.
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetBlendedMVSCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far
        
        self.root_dir = cfg.root_dir
        self.scan_ids = cfg.scan_id
        self.n_views = cfg.n_views 
        
        self.test_ref_view = cfg.test_ref_view
        self.test_context_views = cfg.test_context_views
        self.test_target_views = cfg.test_target_views
        
        self.metas = self.build_list()
        self.num_img = len(self.metas)
        self.dataset = cfg.name
        self.use_mask = cfg.use_mask
        
        self.data_dir = os.path.join(self.root_dir)
        
        if self.cfg.name == "blendedmvs":
            self.img_wh = [768, 576]
        elif self.cfg.name == "mvimage":
            self.img_wh = [960, 544]
            
        self.define_transforms()
        
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train"):
            self.metas = self.shuffle(self.metas)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.metas = [
                chunk
                for chunk_index, chunk in enumerate(self.metas)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]
        
        #* iterate over all chunks
        for idx, meta in enumerate(self.metas):

            # Load the chunk.
            scan_id, ref_view, src_views = meta
            
            if self.stage == 'train':
                view_ids = [ref_view] + src_views[:self.cfg.n_views]
            
            elif self.stage=='val' or self.stage=='test': 
                view_ids = [ref_view] + src_views[:self.cfg.n_views]


            imgs, imgs_norm, depths_h, depths_mvs_h, masks = [], [], [], [], []
            intrinsics, w2cs, near_fars = [], [], []
            intrinsics_org_scale = []
            monoNs = []
            #TODO: duster code
            img_paths = []
            
            world_mats_np = []
            images_list = []
            raw_near_fars = []
            masks_list = []
            
            for i, vid in enumerate(view_ids):
                if self.dataset == "blendedmvs":
                    img_filename = os.path.join(self.data_dir, scan_id, 'blended_images/{:0>8}.jpg'.format(vid))
                elif self.dataset == "mvimage":
                    img_filename = os.path.join(self.data_dir, scan_id, 'images/{:0>8}.jpg'.format(vid))
                    
                    mask_filename = os.path.join(os.path.dirname(self.data_dir), scan_id, 'masks', '{:0>8}_mask.jpg'.format(vid))
                    masks_list.append(mask_filename)
                        
                images_list.append(img_filename)
                
                proj_mat_filename = os.path.join(self.data_dir, scan_id, 'cams/{:0>8}_cam.txt'.format(vid))
                P, near_, far_ = self.read_cam_file(proj_mat_filename)

                raw_near_fars.append(np.array([near_,far_]))
                world_mats_np.append(P) 
                
            raw_near_fars = np.stack(raw_near_fars)
            ref_world_mat = world_mats_np[0]
            ref_w2c = np.linalg.inv(load_K_Rt_from_P(None, ref_world_mat[:3, :4])[1])

            all_images, all_images_norm, all_intrinsics, all_intrinsics_org_scale, all_w2cs, all_w2cs_original = self.load_scene(images_list, 
                                                                                                    world_mats_np, 
                                                                                                    ref_w2c,
                                                                                                    masks_list=masks_list)
        
            # scale_mat, scale_factor = self.cal_scale_mat(
            #     img_hw=[self.img_wh[1], self.img_wh[0]],
            #     intrinsics=all_intrinsics_org_scale,
            #     extrinsics=all_w2cs,
            #     near_fars=raw_near_fars,
            #     factor=1.1)
            
    
            # scaled_intrinsics, scaled_w2cs, scaled_c2ws, scaled_near_fars = self.scale_cam_info(all_images, 
            #                                                                 all_intrinsics, all_w2cs, scale_mat)
            
            imgs_norm = all_images_norm
            imgs = all_images
            
            # intrinsics = scaled_intrinsics
            # w2cs = scaled_w2cs
            # c2ws = scaled_c2ws
            # near_fars = scaled_near_fars

            intrinsics = all_intrinsics.float()
            w2cs = all_w2cs.float()
            c2ws = w2cs.clone().inverse()
            near_fars = torch.from_numpy(raw_near_fars).float()
            
        
            focal_lengths = repeat(intrinsics[0, :2, :2].diagonal(), 'xy -> b xy', b=len(intrinsics))
            principal_points = intrinsics[0, :2, 2]
            start_idx = 0
            
            # context_indices, target_indices = np.array([i for i in range(len(src_views))]), np.array([0]) 
            context_indices, target_indices = np.array([i for i in range(self.cfg.view_sampler.num_context_views)]), np.array([i for i in range(self.cfg.view_sampler.num_context_views, len(view_ids))])
            context_indices, target_indices = torch.from_numpy(context_indices), torch.from_numpy(target_indices)
            
            # tensors used as indices must be long, int, byte or bool tensors
            context_indices, target_indices = context_indices.long(), target_indices.long()
            
            context_images, target_images = imgs[context_indices], imgs[target_indices]
            context_images_norm, target_images_norm = imgs_norm[context_indices], imgs_norm[target_indices]
            
            # Skip the example if the images don't have the right shape.
            context_image_invalid = context_images.shape[1:] != (3, self.cfg.image_shape[0], self.cfg.image_shape[1])
            target_image_invalid = target_images.shape[1:] != (3, self.cfg.image_shape[0], self.cfg.image_shape[1])
            if context_image_invalid or target_image_invalid:
                print(
                    f"Skipped bad example {scan}. Context shape was "
                    f"{context_images.shape} and target shape was "
                    f"{target_images.shape}."
                )
                return
            
            # Resize the world to make the baseline 1.
            context_extrinsics = c2ws[context_indices].clone()
            num_views = context_extrinsics.shape[0]

            if num_views > 1 and self.cfg.make_baseline_1:
                # Extract the positions of all views
                positions = context_extrinsics[:, :3, 3]
                # Compute pairwise distances
                pairwise_distances = [
                    (positions[i] - positions[j]).norm()
                    for i in range(num_views) for j in range(i + 1, num_views)
                ]
                # Compute the average pairwise distance as the baseline
                scale = sum(pairwise_distances) / len(pairwise_distances)
                if scale < self.cfg.baseline_epsilon:
                    print(
                        f"Skipped {scan} because of insufficient baseline "
                        f"{scale:.6f}"
                    )
                    continue
                # Normalize all translations
                c2ws[:, :3, 3] /= scale
            else:
                scale = 1
            
            #! make the first extrinsics to be the reference view (identical rotation, zero translation) by projection matrices
            all_indices = torch.cat([context_indices, target_indices])
            c2ws_all = c2ws[all_indices].clone()
            w2cs_all = torch.inverse(c2ws_all)
            
            # Extract the first extrinsic (rotation and translation)
            def transform_extrinsics(w2cs):
                w2cs = w2cs.detach()
                ref_w2c_inv = torch.linalg.inv(w2cs[0])
                
                return torch.einsum('nij,jk->nik', w2cs, ref_w2c_inv)

            w2cs_ref_all = transform_extrinsics(w2cs_all)
            c2ws_ref_all = w2cs_ref_all.inverse()
            
            c2ws[all_indices] = c2ws_ref_all.clone()
            extrinsics_gt = c2ws.clone()
            if self.cfg.single_view:
                c2ws[context_indices] = c2ws[context_indices[0]].clone().detach()
            #! ---------------------------------------------------------------------------------------------------
            
            R = extrinsics_gt[:, :3, :3].clone()
            T = extrinsics_gt[:, :3, 3].clone()

            nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
            # nf_scale = 1.0
            
            example = {
                "context": {
                    "extrinsics": c2ws[context_indices].clone(), #* B x 4 x 4
                    "extrinsics_gt": extrinsics_gt[context_indices].clone(), #* B x 4 x 4  
                    "R": R[context_indices], #* B x 3 x 3
                    "T": T[context_indices], #* B x 3
                    "intrinsics": intrinsics[context_indices][..., :3, :3], #* B x 3 x 3
                    "focal_length": focal_lengths[context_indices], #* B x 2
                    "image": context_images, #* B x 3 x H x W
                    "image_normalized": context_images_norm, #* B x 3 x H x
                    # "depth": depths_h[context_indices], #* B x H x W
                    "near": near_fars[context_indices][:, 0] / nf_scale if not self.cfg.generic_near_far else self.get_bound("near", len(context_indices)) / nf_scale,
                    "far": near_fars[context_indices][:, 1] / nf_scale if not self.cfg.generic_near_far else self.get_bound("far", len(context_indices)) / nf_scale,
                    "index": context_indices,
                    "view_ids": [view_ids[i] for i in context_indices],
                },
                "target": {
                    "extrinsics": c2ws[target_indices].clone(), #* B x 4 x 4
                    "extrinsics_gt": extrinsics_gt[target_indices].clone(), #* B x 4 x 4  
                    "R": R[target_indices], #* B x 3 x 3
                    "T": T[target_indices], #* B x 3
                    "intrinsics": intrinsics[target_indices][..., :3, :3], #* B x 3 x 3
                    "focal_length": focal_lengths[target_indices], #* B x 2
                    "image": target_images, #* B x 3 x H x W
                    "image_normalized": target_images_norm, #* B x 3 x H x
                    # "depth": depths_h[target_indices], #* B x H x W
                    "near": near_fars[target_indices][:, 0] / nf_scale if not self.cfg.generic_near_far else self.get_bound("near", len(target_indices)) / nf_scale,
                    "far": near_fars[target_indices][:, 1] / nf_scale if not self.cfg.generic_near_far else self.get_bound("far", len(target_indices)) / nf_scale,
                    "index": target_indices,
                    "view_ids": [view_ids[i] for i in target_indices],
                },
                "scene": scan_id, #* string for scene name
                "context_indices": context_indices, #* indices of the context views
                "target_indices": target_indices, #* indices of the target views
                "view_ids": view_ids, #* indices of the views
            }
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    
    def scale_cam_info(self, all_images, all_intrinsics, all_w2cs, scale_mat):
        new_intrinsics = []
        new_w2cs = []
        new_c2ws = []
        new_render_w2cs = []
        new_render_c2ws = []
        proj_matrices = []
        new_near_fars = []
        
        for idx in range(len(all_images)):
            intrinsics = all_intrinsics[idx]
            P = intrinsics @ all_w2cs[idx] @ scale_mat
            P = P.cpu().numpy()[:3, :4]

            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            new_intrinsics.append(intrinsics)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1
            new_near_fars.append([0.95 * near, 1.05 * far])
            
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32) 
            intrinsic_cp = all_intrinsics[idx].clone()[:3, :3]
            intrinsic_cp[:2] /= 4 # 1/4->1/2->1
            proj_mat[0, :4, :4] = all_w2cs[idx]
            proj_mat[1, :3, :3] = intrinsic_cp
            proj_matrices.append(proj_mat)
            
            
        new_intrinsics, new_w2cs, new_c2ws, new_near_fars = \
            np.stack(new_intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), \
            np.stack(new_near_fars)
        
        new_intrinsics = torch.from_numpy(np.float32(new_intrinsics))
        new_w2cs = torch.from_numpy(np.float32(new_w2cs))
        new_c2ws = torch.from_numpy(np.float32(new_c2ws))
        new_near_fars = torch.from_numpy(np.float32(new_near_fars))

        return new_intrinsics, new_w2cs, new_c2ws, new_near_fars
    
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
            
            scale_x = self.cfg.image_shape[1] / self.cfg.original_image_shape[1] #(* the width of the image
            scale_y = self.cfg.image_shape[0] / self.cfg.original_image_shape[0] #(* the height of the image

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
    
    
    def load_scene(self, images_list, world_mats_np, ref_w2c, masks_list=None):
        all_images = []
        all_images_norm = []
        all_masks = []
        all_intrinsics = []
        all_intrinsics_org_scale = []
        all_w2cs = []
        all_w2cs_original = []

        for idx in range(len(images_list)):
            
            img_src = Image.open(images_list[idx]) #* W, H
            original_w, original_h = img_src.size
            
            img = self.transform(img_src)
            img_norm = self.transform_norm(img_src)
                
            if len(masks_list) > 0 and self.use_mask:
                mask = cv2.imread(masks_list[idx], 0) 
                mask = cv2.resize(mask, (self.img_wh[0], self.img_wh[1])) / 254.
                # apply foreground mask to remove background
                img = img * np.expand_dims(mask, -1)
                img_norm = img_norm * np.expand_dims(mask, -1)
                
            all_images.append(img)
            all_images_norm.append(img_norm)

            P = world_mats_np[idx]
            P = P[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            
            scale_x = self.cfg.image_shape[1] / original_w
            scale_y = self.cfg.image_shape[0] / original_h

            intrinsics[0, :] *= scale_x
            intrinsics[1, :] *= scale_y
            
            all_intrinsics_org_scale.append(intrinsics.copy())
            
            intrinsics[:1] /= self.cfg.image_shape[1]  #* the width of the image
            intrinsics[1:2] /= self.cfg.image_shape[0]   #* the height of the image
            
            w2c = np.linalg.inv(c2w)
            all_intrinsics.append(intrinsics)
            # - transform from world system to ref-camera system
            all_w2cs.append(w2c @ np.linalg.inv(ref_w2c))
            all_w2cs_original.append(w2c)

        all_images = torch.from_numpy(np.stack(all_images)).to(torch.float32)
        all_images_norm = torch.from_numpy(np.stack(all_images_norm)).to(torch.float32)
        all_intrinsics = torch.from_numpy(np.stack(all_intrinsics)).to(torch.float32)
        all_intrinsics_org_scale = torch.from_numpy(np.stack(all_intrinsics_org_scale)).to(torch.float32)
        all_w2cs = torch.from_numpy(np.stack(all_w2cs)).to(torch.float32)

        return all_images, all_images_norm, all_intrinsics, all_intrinsics_org_scale, all_w2cs, all_w2cs_original
    
    def build_list(self):
        metas = []  
        
        for scan_id in self.scan_ids:
            
            pair_file = os.path.join(self.root_dir, scan_id, "cams", "pair.txt")    
            # read the pair file
            with open(pair_file) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    
                    if len(self.test_ref_view) > 0:
                        if ref_view not in self.test_ref_view:
                            continue
                        else:
                            # only select 10 views
                            src_views = self.test_ref_view
                    else:
                        if ref_view % 10 != 0:
                            continue
                    metas.append((scan_id, ref_view, src_views))
                    
        print("dataset", self.cfg.name, "metas:", len(metas))
            
        return metas
    
    def read_cam_file(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        P = intrinsics_ @ extrinsics
        # depth_min & depth_interval: line 11
        near = float(lines[11].split()[0])
        far = float(lines[11].split()[-1])
        
        self.depth_min = near
        self.depth_interval = float(lines[11].split()[1]) * 1.06

        return P, near, far
    
    def read_cam_file_mvimage(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        P = intrinsics_ @ extrinsics
        # depth_min & depth_interval: line 11
        near = 400.
        far = 900.
        

        return P, near, far
    
    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        
        # scale down 4x
        depth_h = cv2.resize(depth_h, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST) # (128, 160)
        
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
        return len(self.metas)
