import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import random
from einops import einsum

@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal['re10k']
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    test_context_views: list[int]
    test_target_views: list[int]
    single_view: bool
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    padding_size: int = 0 
    shift_rays_half_pixel: str = "forward"
    draw_figure: bool = False
    


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
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
            
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # print(chunk_path)
            # Load the chunk.
            try :
                chunk = torch.load(chunk_path)
            except:
                print(chunk_path, "failed to load")
                continue
        
            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = self.cfg.test_times_per_scene
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                if times_per_scene > 1:  # specifically for DTU
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                else:
                    scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    # reverse the context
                    # context_indices = torch.flip(context_indices, dims=[0])
                    # print(context_indices)
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue
                
                # restrict target indices to be too large
                if len(target_indices) > 100 and not self.cfg.draw_figure:
                    # downscale by selecting only 10 sliced indices
                    values = torch.linspace(0, len(target_indices)-1, steps=10, dtype=torch.int64)
                    target_indices = target_indices[values]                
                
                # Load the images.
                context_images_in = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images_in)
                context_images_normalized = self.convert_images_normalized(context_images_in)
                
                target_images_in = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images_in)
                target_images_normalized = self.convert_images_normalized(target_images_in)

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1
                
                #! make the first extrinsics to be the reference view (identical rotation, zero translation) by projection matrices
                all_indices = torch.cat([context_indices, target_indices])
                c2ws_all = extrinsics[all_indices].clone()
                w2cs_all = torch.inverse(c2ws_all)
                
                                # Extract the first extrinsic (rotation and translation)
                def transform_extrinsics(w2cs):
                    w2cs = w2cs.detach()
                    ref_w2c_inv = torch.linalg.inv(w2cs[0])
                    
                    return torch.einsum('nij,jk->nik', w2cs, ref_w2c_inv)

                w2cs_ref_all = transform_extrinsics(w2cs_all)
                c2ws_ref_all = w2cs_ref_all.inverse()
                
                extrinsics[all_indices] = c2ws_ref_all.clone()
                extrinsics_gt = extrinsics.clone()
                if self.cfg.single_view:
                    extrinsics[context_indices] = extrinsics[context_indices[0]].clone().detach()
                #! ---------------------------------------------------------------------------------------------------

                R = extrinsics_gt[:, :3, :3].clone()
                T = extrinsics_gt[:, :3, 3].clone()
                focal_lengths = repeat(intrinsics[0, :2, :2].diagonal(), 'xy -> b xy', b=len(intrinsics))

                
                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "extrinsics_gt": extrinsics_gt[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "image_normalized": context_images_normalized,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "near_canonical": (self.get_bound("near", len(context_indices)) / nf_scale)[0],
                        "far_canonical": (self.get_bound("far", len(context_indices)) / nf_scale)[0],
                        "R": R[context_indices],
                        "T": T[context_indices],
                        "focal_length": focal_lengths[context_indices],
                        "index": context_indices,
                        "padding_size": self.cfg.padding_size,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "extrinsics_gt": extrinsics_gt[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "image_normalized": target_images_normalized,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "near_canonical": (self.get_bound("near", len(target_indices)) / nf_scale)[0],
                        "far_canonical": (self.get_bound("far", len(target_indices)) / nf_scale)[0],
                        "R": R[target_indices],
                        "T": T[target_indices],
                        "focal_length": focal_lengths[target_indices],
                        "index": target_indices,
                        "padding_size": self.cfg.padding_size,
                    },
                    "scene": scene,
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
            
            if self.cfg.padding_size != 0:
                torch_images.append(self.to_tensor(image))
            else:
                torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def convert_images_normalized(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.normalize(self.to_tensor(image)))
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

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )