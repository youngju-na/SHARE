from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from pytorch3d.renderer import PerspectiveCameras
from einops import (rearrange, reduce, repeat)

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from ..model.ray_diffusion.utils.rays import cameras_to_rays

@dataclass
class LossCamCfg:
    weight: float


@dataclass
class LossCamCfgWrapper:
    cam: LossCamCfg

class LossCameras(Loss[LossCamCfg, LossCamCfgWrapper]):
    def forward(
        self,
        output: DecoderOutput | None, #! temporaily comment out --------------------------------
        batch: BatchedExample,
        gaussians: Gaussians | None, #! temporaily comment out --------------------------------
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        b, src_view, _, _ = batch['context']['extrinsics_gt'].shape
        _, tgt_view, _, _ = batch['target']['extrinsics_gt'].shape
        
        device = batch['context']['extrinsics'].device
        
        rays_final = rearrange(batch['pred_cam']['rays'], "(b v) c h w -> b v c h w", b=b)
        b, _, _, patch_x, patch_y = rays_final.shape
        
        #TODO: change extrinsics into world-to-cam (cam2world previously)
        context_world2cam = batch['context']['extrinsics_gt'].inverse()
        target_world2cam = batch['target']['extrinsics_gt'].inverse()

        # context_world2cam = rearrange(context_world2cam, "b v x y -> (b v) x y")
        # target_world2cam = rearrange(target_world2cam, "b v x y -> (b v) x y")
        # intrinsics = rearrange(batch['context']['intrinsics'], "b v x y -> (b v) x y")
        intrinsics = batch['context']['intrinsics']
        
        # pytorch3d PerspectiveCameras
        focal_length = intrinsics[:, :, [0, 1], [0, 1]]
        principal_point = intrinsics[:, :, [0, 1], [2, 2]]

        context_cameras = PerspectiveCameras(
            focal_length=rearrange(focal_length, "b v x -> (b v) x"),
            principal_point=rearrange(principal_point, "b v x -> (b v) x"),
            R=rearrange(context_world2cam[:, :, :3, :3], "b v x y -> (b v) x y"), # Rotation
            T=rearrange(context_world2cam[:, :, :3, 3], "b v x -> (b v) x"), # Translation
        ).to(device)
        
        # target_cameras = PerspectiveCameras(
        #     focal_length=rearrange(focal_length, "b v x -> (b v) x"),
        #     principal_point=rearrange(principal_point, "b v x -> (b v) x"),
        #     R=rearrange(target_world2cam[:, :, :3, :3], "b v x y -> (b v) x y"), # Rotation
        #     T=rearrange(target_world2cam[:, :, :3, 3], "b v x -> (b v) x"), # Translation
        # ).to(device)
    
        #! cam to rays
        context_rays = cameras_to_rays(context_cameras, None, num_patches_x=patch_x, num_patches_y=patch_y).to_spatial().reshape(b, src_view, 6, patch_x, patch_y)
        # target_rays = cameras_to_rays(target_cameras, None, num_patches_x=patch_x, num_patches_y=patch_y).rays.transpose(1, 2).reshape(b, tgt_view, 6, patch_x, patch_y)
        
        delta_context = rays_final[:, :src_view, ...] - context_rays
        # delta_target = rays_final[:, src_view:, ...] - target_rays
        
        delta_total = (delta_context**2).mean()
        
        return self.cfg.weight * delta_total