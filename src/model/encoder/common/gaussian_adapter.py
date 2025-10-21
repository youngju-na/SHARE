from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from typing import Literal
from torch import Tensor, nn

from ....geometry.projection import get_world_rays, homogenize_points, get_camera_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance, quaternion_to_matrix, matrix_to_quaternion

from ...utils import sh_utils 

@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    means_cam: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"] | None
    scales: Float[Tensor, "*batch 3"] | None
    rotations: Float[Tensor, "*batch 4"] | None
    harmonics: Float[Tensor, "*batch 3 _"] | None
    opacities: Float[Tensor, " *batch"] | None


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int
    learn_sh_residual_from_canoncal_rgb: bool
    gaussian_2d_scale: bool


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        opacities,
        raw_gaussians, 
        image_shape,
        eps=1e-8,
        use_pred_cams: bool = False,
        pred_cams: Float[Tensor, "*#batch 4 4"] = None,
        predict_only_canonical: bool = False,
        estimation_space: str = "depthmap",
        xyz_only: bool = False,
        scale_factor: float = 1.0,
        extra_info: dict = None,
        learn_sh_residual_from_canoncal_rgb: bool = False,
        gaussian_2d_scale: bool = False,
        
    ):
        
        device = extrinsics.device
        b, v = extrinsics.shape[:2]
        if predict_only_canonical:
            v = 1
        else:
            if use_pred_cams:
                extrinsics = pred_cams
                

        
        # change intrinsic's focal lengths according to the scale factor
        # intrinsics = intrinsics.clone()
        # intrinsics[..., :2, :2] = intrinsics[..., :2, :2] * scale_factor
            
        if xyz_only:
            # origins, directions = get_world_rays(coordinates, extrinsics[:, 0:v], intrinsics[:, 0:v])
            origins, directions = get_camera_rays(coordinates, intrinsics[:, 0:v])
            means_cam = origins + directions * depths[..., None] #* world? cameras? => world 
            means_cam = means_cam.squeeze(-2)
            
            # means_cam = einsum(extrinsics[:, 0:v].inverse(), homogenize_points(means), "b v ... i j, b v ... j -> b v ... i")[..., :3]
            means = einsum(extrinsics[:, 0:v], homogenize_points(means_cam), "b v ... i j, b v ... j -> b v ... i")[..., :3]
            
            # if means_cam.dim() != means.dim():
            #     means_cam = means_cam.unsqueeze(2)
            
            return Gaussians(means=means, means_cam=means_cam, covariances=None, scales=None, rotations=None, harmonics=None, opacities=opacities)
            
        scales, rotations, sh = raw_gaussians.float().split((3, 4, 3 * self.d_sh), dim=-1)

        

        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        
        scales = scale_min + (scale_max - scale_min) * scales
        
        h, w = image_shape
        h, w = h // scale_factor, w // scale_factor
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics[:, 0:1], pixel_size)
        # multiplier = 1.0
        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask
        
        
        if estimation_space == "depthmap":
            scales = scales * depths[..., None] * multiplier[..., None] * 1.0 #! 2.0으로 scale을 좀 크게 만들어보면 어떨지 확인
        else:
            scales = scales * multiplier[..., None] * 1.0 
        
        if learn_sh_residual_from_canoncal_rgb:
                new_sh = torch.zeros_like(sh)
                new_sh[..., 0] = sh_utils.RGB2SH(rearrange(extra_info['images'][0:v], '(b v) c h w -> b v (h w) () () c', b=b, v=v))
                sh = sh + new_sh
                
        if gaussian_2d_scale:
            scales[..., 2] = 0.0 #! 2dgs 
        
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # # Apply c2w_rotations separately to the rotation and scale parameters.
        # if use_pred_cams:
        #     extrinsics = torch.cat([gt_pose[:, 0:1], extrinsics[:, 1:]], dim=1)
        c2w_rotations = extrinsics[..., :3, :3][:, 0:v]

        # Apply c2w_rotations to rotations and scales
        rotations_mat = quaternion_to_matrix(rotations)
        rotations_in_world_mat = c2w_rotations @ rotations_mat @ c2w_rotations.transpose(-1, -2)
        rotations_in_world = matrix_to_quaternion(rotations_in_world_mat)
        
        covariances = build_covariance(scales, rotations)
        
        # Apply the final rotation to the covariance matrix (if necessary)
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
    
        # Compute Gaussian means.
        if estimation_space == "depthmap":
            origins, directions = get_world_rays(coordinates, extrinsics[:, 0:v], intrinsics[:, 0:v])
            means = origins + directions * depths[..., None] #* world? cameras? => world    
            means_cam = einsum(extrinsics[:, 0:v].inverse(), homogenize_points(means), "b v ... i j, b v ... j -> b v ... i")[..., :3]
        elif estimation_space == "pointmap":
            means_cam = depths.clone().unsqueeze(-2)
            means = einsum(extrinsics[:, 0:v], homogenize_points(means_cam), "b v ... i j, b v ... j -> b v ... i")[..., :3]
        elif estimation_space == "world_xyz":
            means = depths.unsqueeze(-2)
            means_cam = einsum(extrinsics[:, 0:v].inverse(), homogenize_points(means), "b v ... i j, b v ... j -> b v ... i")[..., :3]
        elif estimation_space == "cam_xyz":
            means_cam = depths.unsqueeze(-2)
            means = einsum(extrinsics[:, 0:v], homogenize_points(means_cam), "b v ... i j, b v ... j -> b v ... i")[..., :3]
        
        return Gaussians(
            means=means, #* [1, 1, 50176, 1, 1, 3]
            means_cam=means_cam,
            covariances=covariances, #* [1, 1, 50176, 1, 1, 3, 3]
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]), #* [1, 1, 50176, 1, 1, 3, 16]
            opacities=opacities, #* [1, 1, 50176, 1, 1]
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales, #* [1, 1, 50176, 1, 1, 3]
            rotations=rotations_in_world.broadcast_to((*scales.shape[:-1], 4)),
        )
        
        
    def forward_duster(
        self,
        extrinsics,
        intrinsics,
        coordinates,
        depths,
        opacities,
        raw_gaussians,
        image_shape,
        eps=1e-8,
        use_pred_cams: bool = False,
        pred_cams: Float[Tensor, "*#batch 4 4"] = None,
        predict_only_canonical: bool = False,
        estimation_space: str = "depthmap",
    )-> Gaussians:
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        b, v = extrinsics.shape[:2]
        if predict_only_canonical:
            v = 1
        
        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics[:, 0:1], pixel_size)
        
        if estimation_space == "depthmap":
            scales = scales * depths[..., None] * multiplier[..., None] * 1.0 #! 2.0으로 scale을 좀 크게 만들어보면 어떨지 확인
        elif estimation_space == "pointmap":
            scales = scales * multiplier[..., None] * 1.0
        else:
            raise ValueError(f"Unknown estimation space: {estimation_space}")
        
        # scales = scales.sigmoid()
        scales[..., 2] = 0.0 #! core part
        
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Apply c2w_rotations separately to the rotation and scale parameters.
        c2w_rotations = extrinsics[..., :3, :3][:, 0:v] if not use_pred_cams else pred_cams[..., :3, :3] #! camera pose estimation

        # Apply c2w_rotations to rotations and scales
        rotations_mat = quaternion_to_matrix(rotations)
        rotations_in_world_mat = c2w_rotations @ rotations_mat @ c2w_rotations.transpose(-1, -2)
        rotations_in_world = matrix_to_quaternion(rotations_in_world_mat)
        
        covariances = build_covariance(scales, rotations)
        
        # Apply the final rotation to the covariance matrix (if necessary)
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
    
        # Compute Gaussian means.
        if estimation_space == "depthmap":
            origins, directions = get_world_rays(coordinates, extrinsics[:, 0:v], intrinsics[:, 0:v])
            means = origins + directions * depths[..., None] #* world? cameras? => world     
        elif estimation_space == "pointmap":
            means = torch.matmul(depths.unsqueeze(-2).unsqueeze(2), c2w_rotations).squeeze(2)
        
        return Gaussians(
            means=means, #* [1, 2, 50176, 1, 1, 3]
            covariances=covariances, #* [1, 1, 50176, 1, 1, 3, 3]
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]), #* [1, 1, 50176, 1, 1, 3, 16]
            opacities=opacities, #* [1, 1, 50176, 1, 1]
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales, #* [1, 1, 50176, 1, 1, 3]
            rotations=rotations_in_world.broadcast_to((*scales.shape[:-1], 4)),
        )
        
    def forward_transmvsnet(
        self,
        extrinsics,
        intrinsics,
        world_xyz,
        opacities,
        raw_gaussians, 
        image_shape,
        eps=1e-8,
        use_pred_cams: bool = False,
        pred_cams: Float[Tensor, "*#batch 4 4"] = None,
        predict_only_canonical: bool = False,
        estimation_space: str = "depthmap",
    ):
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.float().split((3, 4, 3 * self.d_sh), dim=-1)

        b, v = extrinsics.shape[:2]
        if predict_only_canonical:
            v = 1
        
        #! pred_cam에서 0번째 canonical은 gt를 사용해야함
        if use_pred_cams:
            gt_pose = extrinsics.clone().detach()
        
        # Map scale features to valid scale range.
        # scale_min = self.cfg.gaussian_scale_min
        # scale_max = self.cfg.gaussian_scale_max
        
        # scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        # multiplier = self.get_scale_multiplier(intrinsics[:, 0:1], pixel_size)
        # scales = scales * multiplier[..., None]
        
        # scales[..., -1] -= 1e10 #* 2dgs

        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*(opacities.shape[:-1]), 3, self.d_sh)) * self.sh_mask.to(device)

        # Apply c2w_rotations separately to the rotation and scale parameters.
        if use_pred_cams:
            extrinsics = torch.cat([gt_pose[:, 0:1], extrinsics[:, 1:]], dim=1)
        c2w_rotations = extrinsics[..., :3, :3][:, 0:v]

        # Apply c2w_rotations to rotations and scales
        rotations_mat = quaternion_to_matrix(rotations)
        rotations_in_world_mat = c2w_rotations @ rotations_mat @ c2w_rotations.transpose(-1, -2)
        rotations_in_world = matrix_to_quaternion(rotations_in_world_mat)
        
        covariances = build_covariance(scales, rotations)
        
        # Apply the final rotation to the covariance matrix (if necessary)
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        
        means = world_xyz
        means_cam = torch.matmul(homogenize_points(means).unsqueeze(-2), extrinsics[:, 0:v].inverse()).squeeze(-2)[..., :3]
        
        means = rearrange(means, 'b v r src c -> b v r src () c')
        means_cam = rearrange(means_cam, 'b v r src c -> b v r src () c')
        covariances = rearrange(covariances, 'b v r src c d -> b v r src () c d')
        scales = rearrange(scales, 'b v r src c -> b v r src () c')
        rotations = rearrange(rotations, 'b v r src c -> b v r src () c')
        rotations_in_world = rearrange(rotations_in_world, 'b v r src c -> b v r src () c')
        sh = rearrange(sh, 'b v r src c d -> b v r src () c d')
        opacities = rearrange(opacities, 'b v r src c -> b v r src c')
        
        return Gaussians(
            means=means, #* [1, 1, 50176, 1, 1, 3]
            means_cam=means_cam,
            covariances=covariances, #* [1, 1, 50176, 1, 1, 3, 3]
            harmonics=rotate_sh(sh, c2w_rotations[..., None, None, :, :]), #* [1, 1, 50176, 1, 1, 3, 16]
            opacities=opacities, #* [1, 1, 50176, 1, 1]
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales, #* [1, 1, 50176, 1, 1, 3]
            rotations=rotations_in_world.broadcast_to((*scales.shape[:-1], 4)),
        )
    

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        
        return 7 + 3 * self.d_sh
    
