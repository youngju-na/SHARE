from functools import cache

import torch
from einops import reduce, rearrange
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from src.evaluation.cam_util import compute_angular_error, compute_angular_error_batch
from src.evaluation.eval_translation import get_error, full_scene_scale

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)

@torch.no_grad()
def compute_rotation(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    
    compute_angular_error_batch(rotation1, rotation2)
    
    pass

@torch.no_grad()
def compute_translation(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    pass

@torch.no_grad()
def compute_extrinsics(batch, extrinsics_gt, extrinsics_pred):
    extrinsics_pred = rearrange(extrinsics_pred, 'b v x y -> (b v) x y')
    extrinsics_gt = rearrange(extrinsics_gt, 'b v x y -> (b v) x y')
    
    rot_pred = extrinsics_pred[:, :3, :3]
    rot_gt = extrinsics_gt[:, :3, :3]
    
    trans_pred = extrinsics_pred[:, :3, 3]
    trans_gt = extrinsics_gt[:, :3, 3]
    
    ang_error = compute_angular_error_batch(rot_gt, rot_pred)
    
    scale = full_scene_scale(batch)
    trans_error = get_error('t', rot_pred, trans_pred, rot_gt, trans_gt, gt_scene_sale=scale)
    
    return ang_error

