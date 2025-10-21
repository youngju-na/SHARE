from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import os
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import pickle
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..evaluation.metrics import compute_rotation, compute_translation, compute_extrinsics
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .encoder.dust3r.losses import *
from .utils.loss_utils import l1_loss, ssim, cos_loss, bce_loss, knn_smooth_loss, l2_loss
from .utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, normal2curv, resize_image



from src.model.ray_diffusion.utils.visualization import (
    create_plotly_cameras_visualization,
    create_training_visualizations,
)

from src.model.ray_diffusion.eval.utils import (
    compute_angular_error_batch,
    compute_camera_center_error,
    full_scene_scale,
    n_to_np_rotations,
    compute_geodesic_distance_from_two_matrices,
)

from pytorch3d.renderer import PerspectiveCameras, RayBundle
from src.model.ray_diffusion.utils.rays import (Rays, compute_ndc_coordinates, rays_to_cameras, cameras_to_rays)

class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_depth: bool
    eval_time_skip_steps: int
    surface_level: float
    n_vertices_in_mesh: int
    bboxmin: int | None
    bboxmax: int | None
    center_bbox: bool
    gpu: int
    eval: bool
    extract_mesh: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    render_from_pred: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    max_step: int

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        max_step: int,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.pose_cfg = encoder.cfg.ray
        self.step_tracker = step_tracker
        self.max_step = max_step
        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        
        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            self.pred_mats = {}
            
        #! manually define optimization here
        self.automatic_optimization = False
        self.multi_gpu = False
        self.validation_step_outputs = []
        # torch.autograd.set_detect_anomaly(True)

        # Logger parameters
        self.counter_batch = 0
        self.log_store_ims = True
        self.log_max_ims = 5
        self.log_im_counter_train = 0
        self.log_im_counter_val = 0
        self.log_interval = 50
        
        self.render_from_pred = self.train_cfg.render_from_pred
        self.encoder_cfg = self.encoder.cfg
    
    
        # print('>> Creating train criterion = ConfLoss(Regr3D(L21, norm_mode="avg_dis"), alpha=0.2)')
        # self.pointmap_train_criterion = eval("ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)").to(self.device)
        # print('>> Creating test criterion = Regr3D_ScaleShiftInv(L21, gt_scale=True)')
        # self.pointmap_test_criterion = eval('Regr3D_ScaleShiftInv(L21, gt_scale=True)').to(self.device)
        

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        b, context_v, _, _, _= batch['context']['image'].shape
        # Run the model.
        
        gt_color = rearrange(batch["target"]["image"], 'b v c h w -> (b v) c h w')
        
        encoder_out = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )
        
        gaussians, mickey_out, pred_cam = encoder_out['gaussians'], encoder_out['mickey_out'], encoder_out['pred_cam'] 
        batch['pred_cam'] = pred_cam
        gaussians_anchor, gaussians_offset = encoder_out['gaussians_anchor'], encoder_out['gaussians_offset']
        
        if gaussians is None:
            assert pred_cam is not None, "No both gaussians and pred_cam are None"
            
            output = None

        #TODO: source view render loss
        loss_src_render = None
        # loss_src_render = l1_loss(color_all[:, :context_v], original_context_img)
        #TODO: pointmap loss
        loss_pointmap = None
        
        loss_mask, loss_surface, loss_opac, loss_curv, loss_monoN = None, None, None, None, None
        sugar_loss, mse_loss, depth_loss, cam_loss = None, None, None, None
        total_render_loss = 0
        mse_loss, mse_loss_anchor, mse_loss_offset = None, None, None
        
        if gaussians is not None:
            if pred_cam is not None and self.render_from_pred:
                #TODO: render from pred cam
                color_all, normal_all, depth_all, opac_all = self.decoder.forward(
                    gaussians,
                    pred_cam['mats'],
                    torch.cat([batch["context"]["intrinsics"], batch["target"]["intrinsics"]], dim=1), #TODO: check this
                    torch.cat([batch["context"]["near"], batch["target"]["near"]], dim=1),
                    torch.cat([batch["context"]["far"], batch["target"]["far"]], dim=1),
                    (h, w),
                )
                color = color_all[:, context_v:] 
                normal = normal_all[:, context_v:] if normal_all is not None else None
                depth = depth_all[:, context_v:] if depth_all is not None else None
                opac = opac_all[:, context_v:] if opac_all is not None else None
                
            else:
                #TODO: render from GT cam
                color, normal, depth, opac = None, None, None, None
                color, normal, depth, opac = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode="depth",
                )
                output = {"color": color, "normal": normal, "depth": depth, "opac": opac}
                assert output['color'].shape[-2:] == batch['target']['image'].shape[-2:], "output color shape mismatch" 
                #! gaussian anchor is more attentive to the canonical view in context views
                if encoder_out['gaussians_anchor'] is not None:
                    color_anchor, normal_anchor, depth_anchor, opac_anchor = None, None, None, None
                    color_anchor, normal_anchor, depth_anchor, opac_anchor = self.decoder.forward(
                        gaussians_anchor,
                        batch["context"]["extrinsics_gt"][:, :1],
                        batch["context"]["intrinsics"][:, :1],
                        batch["context"]["near"][:, :1],
                        batch["context"]["far"][:, :1],
                        (h, w),
                        depth_mode="depth",
                    )
                    output['color_anchor'], output['normal_anchor'], output['depth_anchor'], output['opac_anchor'] = color_anchor, normal_anchor, depth_anchor, opac_anchor
                
                #! while gaussian offset is more attentive to the source views in context views
                if encoder_out["gaussians_offset"] is not None:
                    color_offset, normal_offset, depth_offset, opac_offset = None, None, None, None
                    color_offset, normal_offset, depth_offset, opac_offset = self.decoder.forward(
                        gaussians_offset,
                        batch["context"]["extrinsics_gt"][:, 1:context_v],
                        batch["context"]["intrinsics"][:, 1:context_v],
                        batch["context"]["near"][:, 1:context_v],
                        batch["context"]["far"][:, 1:context_v],
                        (h, w),
                        depth_mode="depth",
                    )
                    output['color_offset'], output['normal_offset'], output['depth_offset'], output['opac_offset'] = color_offset, normal_offset, depth_offset, opac_offset
        
        
            rgb_softmax = color[0]
            target_gt = batch["target"]["image"]
            
            mono, monoN = None, None
            
            if depth.dim() != 5:
                depth = depth.unsqueeze(2)
            mask_vis = (opac.detach() > 1e-5) if opac is not None else torch.ones_like(depth)
            
            depth = rearrange(depth, "b v d h w -> (b v) d h w") if depth is not None else None
            mask_vis = rearrange(mask_vis, "b v d h w -> (b v) d h w") if mask_vis is not None else None
            
            normal = rearrange(normal, "b v c h w -> (b v) c h w") if normal is not None else None
            opac = rearrange(opac, "b v c h w -> (b v) c h w") if opac is not None else None
        
            
            '''
            #################################
            ||                            ||
            ||            LOSS            ||
            ||                            ||
            ################################
            '''

            if self.global_step >= 10000 and self.encoder.cfg.use_monoN_loss: #! 1000
                if mono is not None:
                    loss_monoN = cos_loss(normal, monoN, weight=1, dim=1) * 0.01
            
            
            if color.sum() == 0:
                print("output trivial solution: black image")
            
            # self.log("loss/loss_mask", loss_mask)
            if loss_surface is not None:
                self.log("loss/loss_surface", loss_surface)
            # self.log("loss/loss_opac", loss_opac)
            if loss_curv is not None:
                self.log("loss/loss_curv", loss_curv)
            if loss_monoN is not None:
                self.log("loss/loss_monoN", loss_monoN)
            if loss_pointmap is not None:
                self.log("loss/loss_pointmap", loss_pointmap)
            if loss_src_render is not None:
                self.log("loss/loss_src_render", loss_src_render)

            # total_render_loss += loss_mask
            if loss_surface is not None:
                total_render_loss += loss_surface
            # total_render_loss += loss_opac
            if loss_curv is not None:
                total_render_loss += loss_curv
            if loss_monoN is not None:
                total_render_loss += loss_monoN
            #TODO: add pointmap loss
            if loss_pointmap is not None:
                total_render_loss += loss_pointmap 
            if loss_src_render is not None:
                total_render_loss += loss_src_render   
            
            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(color, "b v c h w -> (b v) c h w"),
            )
            self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        
        for loss_fn in self.losses:
            if self.global_step < self.encoder_cfg.pose_estimation_warmup:
                if loss_fn.name != "cam":
                    continue
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            if not isinstance(loss, dict):
                self.log(f"loss/{loss_fn.name}", loss)
            else:
                for key, value in loss.items():
                    if value is not None:
                        self.log(f"loss/{loss_fn.name}_{key}", value)
            
            if "mse" in loss_fn.name :
                mse_loss = loss['all']
                mse_loss_anchor = loss['anchor'] if loss['anchor'] is not None else 0
                mse_loss_offset = loss['offset'] if loss['offset'] is not None else 0
                loss = mse_loss + mse_loss_anchor + mse_loss_offset
            if "depth_loss" in loss_fn.name:
                depth_loss = loss
            if "cam" in loss_fn.name:
                cam_loss = loss
            total_render_loss = total_render_loss + loss
        self.log("loss/total", total_render_loss)
        
        
        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):

            cam_loss_str = f"; cam_loss = {cam_loss:.3f}" if cam_loss is not None else ""
            color_loss_str = f"; color_loss = {mse_loss:.3f}" if mse_loss is not None else ""
            color_anchor_loss_str = f"; color_anchor_loss = {mse_loss_anchor:.3f}" if mse_loss_anchor is not None else ""
            color_offset_loss_str = f"; color_offset_loss = {mse_loss_offset:.3f}" if mse_loss_offset is not None else ""
            
            depth_loss_str = f"; depth_loss = {depth_loss:.3f}" if depth_loss is not None else ""
            sugar_loss_str = f"; sugar_loss = {sugar_loss:.3f}" if sugar_loss is not None else ""
            
            mask_loss_str = f"; mask_loss = {loss_mask:.3f}" if loss_mask is not None else ""
            surface_loss_str = f"; surface_loss = {loss_surface:.3f}" if loss_surface is not None else ""
            opac_loss_str = f"; opac_loss = {loss_opac:.3f}" if loss_opac is not None else ""
            curv_loss_str = f"; curv_loss = {loss_curv:.3f}" if loss_curv is not None else ""
            monoN_loss_str = f"; monoN_loss = {loss_monoN:.3f}" if loss_monoN is not None else ""      
            pointmap_loss_str = f"; pointmap_loss = {loss_pointmap:.3f}" if loss_pointmap is not None else "" 
            src_render_loss_str = f"; src_render_loss = {loss_src_render:.3f}" if loss_src_render is not None else ""                 
            
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                + sugar_loss_str  # Correctly concatenate sugar_loss_str
                + cam_loss_str  # Correctly concatenate cam_loss_str
                + color_loss_str  # Correctly concatenate color_loss_str
                + color_anchor_loss_str  # Correctly concatenate color_anchor_loss_str
                + color_offset_loss_str  # Correctly concatenate color_offset_loss_str
                + depth_loss_str  # Correctly concatenate depth_loss_str
                + mask_loss_str  # Correctly concatenate mask_loss_str
                + surface_loss_str  # Correctly concatenate surface_loss_str
                + opac_loss_str  # Correctly concatenate opac_loss_str
                + curv_loss_str  # Correctly concatenate curv_loss_str
                + monoN_loss_str  # Correctly concatenate monoN_loss_str
                + pointmap_loss_str  # Correctly concatenate pointmap_loss_str
                + src_render_loss_str  # Correctly concatenate src_render_loss_str
                + f"; loss = {total_render_loss:.3f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        
        training_step_ok = self.backward_step(batch['context'],total_render_loss)
        
        #! camera visualization
        # if isinstance(self.logger, WandbLogger) and self.global_step % 1000 == 0:
        if isinstance(self.logger, WandbLogger) and self.global_step % 1000 == 0 and cam_loss is not None:
            self.visualize_camera(batch)
            

        if self.global_step % 1000 == 1 and self.global_rank == 0 and mse_loss is not None:
            # Run video validation step.
            self.render_video_interpolation(batch, mode='train')
            self.render_video_wobble(batch, mode='train')
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

            vis_normal_all = []
            vis_depth_all = []
            for i in range(len(depth)):
                if normal is not None:
                    vis_normal_all.append(normal2rgb(normal[i], mask_vis[i])) 
                vis_depth_all.append(depth2rgb(depth[i], mask_vis[i]))    

            vis_normal = torch.stack(vis_normal_all) if normal is not None else None
            vis_depth = torch.stack(vis_depth_all)
            
            color_anchor = rearrange(output['color_anchor'], "b v c h w -> (b v) c h w") if 'color_anchor' in output.keys() else None
            color_offset = rearrange(output['color_offset'], "b v c h w -> (b v) c h w") if 'color_offset' in output.keys() else None
            color_all = rearrange(output['color'], "b v c h w -> (b v) c h w") if 'color' in output.keys() else None

            if normal is not None:
                self.logger.log_image(f"train/normal_{i}", [prep_image(vis_normal)], step=self.global_step)
            self.logger.log_image(f"train/depth_{i}", [prep_image(vis_depth)], step=self.global_step)

            if color_anchor is not None:
                self.logger.log_image(f"train/color_anchor_{i}", [prep_image(color_anchor)], step=self.global_step)
            if color_offset is not None:
                self.logger.log_image(f"train/color_offset_{i}", [prep_image(color_offset)], step=self.global_step)
            self.logger.log_image(f"train/color_all_{i}", [prep_image(color_all)], step=self.global_step)
            
            #! camera metric evaluation
            pred_mats = rearrange(encoder_out['pred_cam']['mats'], 'b v x y -> (b v) x y')
            # pred_mats = rearrange(batch['context']['extrinsics_gt'], 'b v x y -> (b v) x y') #* c2w
            
            R_gt = rearrange(batch['context']['R'], 'b v x y -> (b v) x y')
            T_gt = rearrange(batch['context']['T'], 'b v x -> (b v) x')
            
            # 카메라 저장
            self.pred_mats[batch['scene'][0]] = pred_mats.detach().cpu()
            
            #! CoPoNeRF Style evaluation
            norm_pred = pred_mats[:, :3, 3][1:] / (torch.linalg.norm(pred_mats[:, :3, 3][1:], dim=-1, keepdim=True) + 1e-6)
            norm_gt = T_gt[1:] / (torch.linalg.norm(T_gt[1:], dim=-1, keepdim=True) + 1e-6)

            # Compute cosine similarity for all pairs
            cosine_similarity = torch.sum(norm_pred * norm_gt, dim=-1)

            # Clamp values for arccos
            cosine_similarity_clamped = torch.clamp(cosine_similarity, -1.0, 1.0)

            # Compute angles in degrees for all pairs
            angle_degrees = torch.arccos(cosine_similarity_clamped) * 180 / np.pi

            # Compute average angle degree
            avg_angle_degree = angle_degrees.mean()

            # Compute geodesic distances
            geodesic = compute_geodesic_distance_from_two_matrices(
                pred_mats[..., :3, :3][1:], R_gt[..., :3, :3][1:]
            ) * 180 / np.pi
        
            self.log(f"train/rotation_angle", geodesic.mean())
            self.log(f"train/translation_angle", avg_angle_degree.mean())
            
            
        
        
        
    def backward_step(self, batch, total_render_loss):
        opt = self.optimizers()

        # update model
        opt.zero_grad()

        # Generate gradients for learning keypoint offsets
        
        total_loss = total_render_loss
        
        total_loss.backward(retain_graph=False)

        opt.step()

        return True
    
    
    def wandb_log_step(self, batch, avg_loss, outputs, probs_grad, training_step_ok):

        # Use PyTorch Lightning's self.log for scalar values
        self.log('train/loss', avg_loss.detach())
        self.log('train/loss_rot', outputs['avg_loss_rot'].detach())
        self.log('train/loss_trans', outputs['avg_loss_trans'].detach())
        
        torch.cuda.empty_cache()
        self.counter_batch += 1
    
    
    def visualize_camera(self, batch, crop_parameters=None):
        """Code borrowed from the original implementation of the Cameras as Rays [ICLR'24].
        """
    
        pred_rays = batch['pred_cam']['rays']
        intrinsics = batch['context']['intrinsics_org_scale'] #* b v 3 3
        
        # pred_cams = []
        # for batch_idx in range(pred_rays.shape[0]):
        #     pred_cam = rays_to_cameras(
        #         Rays.from_spatial(pred_rays)[batch_idx],
        #         None,
        #         num_patches_x=self.pose_cfg.num_patches_x,
        #         num_patches_y=self.pose_cfg.num_patches_y,
        #         focal_lengths = intrinsics[:1, 0, [0,1], [0,1]],
        #         principal_points = intrinsics[:1, 0, [0,1], [2,2]],
        #     ).to(self.device)
        #     pred_cams.append(pred_cam)
        
        context_world2cam = batch['context']['extrinsics_gt'].inverse()
        # pytorch3d PerspectiveCameras
        focal_length = intrinsics[:, :, [0, 1], [0, 1]]
        principal_point = intrinsics[:, :, [0, 1], [2, 2]]
        
        cameras_gt = PerspectiveCameras(
            focal_length=rearrange(focal_length, "b v x -> (b v) x"),
            principal_point=rearrange(principal_point, "b v x -> (b v) x"),
            R=rearrange(context_world2cam[:, :, :3, :3], "b v x y -> (b v) x y"), # Rotation
            T=rearrange(context_world2cam[:, :, :3, 3], "b v x -> (b v) x"), # Translation
        ).to(self.device)
        
        for camera in cameras_gt:
            # AMP may not cast back to float
            camera.R = camera.R.float()
            camera.T = camera.T.float()


        vis_images, cameras_pred_batched = create_training_visualizations(
            pred_rays = pred_rays,
            images=rearrange(batch['context']['image'], "b v c h w -> (b v) c h w"),
            device=self.device,
            cameras_gt=cameras_gt,
            num_images=batch['context']['image'].shape[1],
            calculate_intrinsics=False,
            intrinsics=batch['context']['intrinsics_org_scale'],
        )

        fig = create_plotly_cameras_visualization(cameras_gt, cameras_pred_batched)
        plot = wandb.Plotly(fig)
        wandb.log({f"Vis plotly": plot})

        for i, vis_image in enumerate(vis_images):
            im = wandb.Image(
                vis_image, caption=f"iteration {self.global_step} example {i}"
            )
            wandb.log({f"Vis {i}": im})
    
    def construct_comparison_image(self, image, label):
        # Example function to add label and prepare image for logging
        return add_border(add_label(image, label))
    

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1


        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            encoder_out = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                # target=batch['target'],
            )
        
        
        with self.benchmarker.time("decoder", num_calls=v):
            gaussians = encoder_out['gaussians']

            color, normal, depth, opac = None, None, None, None
            color, normal, depth, opac = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode="depth",
            )
            # rgb_softmax = output_softmax.color[0]
            rgb_softmax = color[0]
            
            
            #! offset들마다 다른 depth를 예측하는지 확인
            color_offsets = []
            depth_offsets = []
            
            for i in range(self.encoder.cfg.num_offset_gaussian):
                color_offset, _, depth_offset, _ = self.decoder.forward(
                    encoder_out['gaussians_offsets_list'][i],
                    batch["context"]["extrinsics"][:, 0:1],
                    batch["context"]["intrinsics"][:, 0:1],
                    batch["context"]["near"][:, 0:1],
                    batch["context"]["far"][:, 0:1],
                    (h, w),
                    depth_mode="depth",
                )
                # rgb_softmax = output_softmax.color[0]
                color_offsets.append(color_offset[0])
                depth_offsets.append(depth_offset[0])
            
        output = {"color": color, "normal": normal, "depth": depth, "opac": opac}

        
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = color[0] #! not masked color
        
        extrinsics_pred = encoder_out['pred_cam']['mats']
        
        rgb_gt = batch["target"]["image"][0]
        extrinsics_gt = batch["target"]["extrinsics"]
        
        masked_color = False
        if masked_color:
            images_prob = color[0] * (rgb_gt > 0).float()
        
        all_indices_str = "_".join([str(x.item()) for x in batch["context"]["view_ids"]]) + "_" + "_".join([str(x.item()) for x in batch["target"]["view_ids"]])
        # Save images.
        if self.test_cfg.save_image:
            # Save color images
            for index, color, color_gt in zip(batch["target"]["index"][0], images_prob, rgb_gt):
                save_image(color, path / scene / all_indices_str / f"color/{index:0>6}.png")
                save_image(color_gt, path / scene / all_indices_str / f"color_gt/{index:0>6}.png")
            # Generate projections
            projections = hcat(*render_projections(gaussians, 256, extra_label="(Softmax)")[0])
            # Save projections in the local directory
            save_image(add_border(projections), path / scene / f"projection/all_gaussians_{self.encoder.cfg.rendering_units}.png")
            
            # Generate projections
            if encoder_out['gaussians_anchor'] is not None:
                projections = hcat(*render_projections(encoder_out['gaussians_anchor'], 256, extra_label="(Softmax)", all_gaussians=gaussians)[0])
                # Save projections in the local directory
                save_image(add_border(projections), path / scene / f"projection/anchor_gaussians_{self.encoder.cfg.rendering_units}.png")
            
            # Generate projections
            if encoder_out['gaussians_offset'] is not None:
                projections = hcat(*render_projections(encoder_out['gaussians_offset'], 256, extra_label="(Softmax)", all_gaussians=gaussians)[0])
                # Save projections in the local directory
                save_image(add_border(projections), path / scene / f"projection/offset_gaussians_{self.encoder.cfg.rendering_units}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )
            
        #TODO: save depth
        if self.test_cfg.save_depth:
            # Color-map the result.
            def depth_map(result):
                near = result[result >= 0][:16_000_000].quantile(0.01).log()
                far = result.view(-1)[:16_000_000].quantile(0.99).log()
                result = result.log()
                result = 1 - (result - near) / (far - near + 1e-6)
                return apply_color_map_to_image(result, "turbo")
            
            def depth_map_abs(result):
                near = 1.
                far = 6.
                result = 1 - (result - near) / (far - near + 1e-6)
                return apply_color_map_to_image(result, "turbo")
            
            def depth_map_relative(result): #! result의 depth가 Ref를 기준으로 얼마나 떨어져 있는지 표현하는 depth map
                near = result.min()
                far = result.max()
                
                result = 1 - (result - near) / (far - near + 1e-6)
                return apply_color_map_to_image(result, "turbo")
            
            extrinsics_np = batch['target']['extrinsics_gt'][0].cpu().numpy()
            intrinsics_np = batch['target']['intrinsics'][0].cpu().numpy()
            
            for index, depths, extrinsic, intrinsic in zip(batch["target"]["index"][0], output['depth'][0], extrinsics_np, intrinsics_np):
                save_image(depth_map_abs(depths), path / scene / f"depth/{index:0>6}.png")

                #! 각각 다른 offset을 이용해서 depth map rendering 
                for k in range(self.encoder.cfg.num_offset_gaussian+1):
                    if k > 0:
                        save_image(
                            depth_map(depth_offsets[k-1]),
                            path / scene / f"depth_offsets/{index:0>6}_offset_{k-1}.png", #! depth map 그냥 뽑기
                        )
                        
                        save_image(
                            depth_map_relative(depth_offsets[k-1] - depths),
                            path / scene / f"depth_offsets/{index:0>6}_relative_offset_{k-1}.png", #! offset_depth - depth map 
                        )
                    else:
                        save_image(
                                depth_map_abs(rearrange(encoder_out['anchor_depth'], '() () (H W) () () -> H W', H=h, W=w)),
                                path / scene / f"anchor_depth/{index:0>6}_anchor.png",
                            )
                    
                
                gt_depth = batch["target"]["depth"][0] if 'depth' in batch["target"] else None
                depth_mask = gt_depth > 0 if gt_depth is not None else None

                # save depth as npy
                np.save(path / scene / f"depth/{index:0>6}.npy", 
                    {"depth": depths.cpu().numpy(), "extrinsic":extrinsic, "intrinsic": intrinsic})

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = [] 
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []
           
            if f"rotation_angle" not in self.test_step_outputs:
                self.test_step_outputs[f"rotation_angle"] = []
            if f"translation_angle" not in self.test_step_outputs:
                self.test_step_outputs[f"translation_angle"] = []
            
            # if f"pred_mats" not in self.self.test_step_outputs:
            #     self.test_step_outputs[f"pred_mats"] = dict()
            

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )
            
            #! camera metric evaluation
            pred_mats = rearrange(encoder_out['pred_cam']['mats'], 'b v x y -> (b v) x y')
            # pred_mats = rearrange(batch['context']['extrinsics_gt'], 'b v x y -> (b v) x y') #* c2w
            
            R_gt = rearrange(batch['context']['R'], 'b v x y -> (b v) x y')
            T_gt = rearrange(batch['context']['T'], 'b v x -> (b v) x')
            
            # 카메라 저장
            self.pred_mats[batch['scene'][0]] = pred_mats.detach().cpu()
            
            #! CoPoNeRF Style evaluation
            norm_pred = pred_mats[:, :3, 3][1:] / (torch.linalg.norm(pred_mats[:, :3, 3][1:], dim=-1, keepdim=True) + 1e-6)
            norm_gt = T_gt[1:] / (torch.linalg.norm(T_gt[1:], dim=-1, keepdim=True) + 1e-6)

            # Compute cosine similarity for all pairs
            cosine_similarity = torch.sum(norm_pred * norm_gt, dim=-1)

            # Clamp values for arccos
            cosine_similarity_clamped = torch.clamp(cosine_similarity, -1.0, 1.0)

            # Compute angles in degrees for all pairs
            angle_degrees = torch.arccos(cosine_similarity_clamped) * 180 / np.pi

            # Compute average angle degree
            avg_angle_degree = angle_degrees.mean()

            # Compute geodesic distances
            geodesic = compute_geodesic_distance_from_two_matrices(
                pred_mats[..., :3, :3][1:], R_gt[..., :3, :3][1:]
            ) * 180 / np.pi

            # Append metrics to results
            self.test_step_outputs[f"rotation_angle"].append(geodesic.mean().item())
            self.test_step_outputs[f"translation_angle"].append(avg_angle_degree.item())

            # Debug output
            print("Rotation:", geodesic, "Translation angles:", angle_degrees)
            print(
                "Rotation error so far:", np.mean(self.test_step_outputs[f"rotation_angle"]),
                "Translation error so far:", np.mean(self.test_step_outputs[f"translation_angle"])
            )
            
            print("view_ids:", all_indices_str, "PSNR:", self.test_step_outputs[f"psnr"][-1], "SSIM:", self.test_step_outputs[f"ssim"][-1], "LPIPS:", self.test_step_outputs[f"lpips"][-1])
            print("PSNR so far:", np.mean(self.test_step_outputs[f"psnr"]), "SSIM so far:", np.mean(self.test_step_outputs[f"ssim"]), "LPIPS so far:", np.mean(self.test_step_outputs[f"lpips"]))



    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()

            pose_save_path = f"/home/youngju/ssd/ufosplat/save_pred_cams/{self.dataset_cfg.name}/pred_mats_{str(self.dataset_cfg.test_context_views)}.pt"
            os.makedirs(os.path.dirname(pose_save_path), exist_ok=True)
            torch.save(self.pred_mats, pose_save_path)
            

        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        encoder_out = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        gaussians_softmax = encoder_out['gaussians']
        
        rgb_softmax = None
        if gaussians_softmax is not None:
            color, normal, depth, opac = self.decoder.forward(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            # rgb_softmax = output_softmax.color[0]
            rgb_softmax = color[0]
        

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            if rgb is not None:
                psnr = compute_psnr(rgb_gt, rgb).mean()
                self.log(f"val/psnr_{tag}", psnr)
                lpips = compute_lpips(rgb_gt, rgb).mean()
                self.log(f"val/lpips_{tag}", lpips)
                ssim = compute_ssim(rgb_gt, rgb).mean()
                self.log(f"val/ssim_{tag}", ssim)
            
            # #! camera metric evaluation
            # pred_cameras = encoder_out['pred_cam']['cameras']
            # _, num_images, _, _, _ = batch['context']['image'].shape
            # gt_scene_scale = full_scene_scale(batch['context'])
            
            # R_gt = rearrange(batch['context']['R'], 'b v x y -> (b v) x y')
            # T_gt = rearrange(batch['context']['T'], 'b v x -> (b v) x')
            # focal_length = rearrange(batch['context']['focal_length'], 'b v x -> (b v) x')

            # cam_errors = []
            
            # R_pred = rearrange(pred_cameras.R, 'b v x y -> (b v) x y')
            # T_pred = rearrange(pred_cameras.T, 'b v x -> (b v) x')
            # f_pred = repeat(pred_cameras.focal_length,'b x -> (b v) x', v=num_images)
            
            # R_pred_rel = n_to_np_rotations(num_images, R_pred).cpu().numpy()
            # R_gt_rel = n_to_np_rotations(num_images, R_gt).cpu().numpy()
            # R_error = compute_angular_error_batch(R_pred_rel, R_gt_rel)

            # CC_error, _ = compute_camera_center_error(
            #     R_pred, T_pred, R_gt, T_gt, gt_scene_scale
            # )

            # cam_errors.append(
            #     {
            #         "R_pred": R_pred.detach().cpu().numpy().tolist(),
            #         "T_pred": T_pred.detach().cpu().numpy().tolist(),
            #         "f_pred": f_pred.detach().cpu().numpy().tolist(),
            #         "R_gt": R_gt.detach().cpu().numpy().tolist(),
            #         "T_gt": T_gt.detach().cpu().numpy().tolist(),
            #         "f_gt": focal_length.detach().cpu().numpy().tolist(),
            #         "scene_scale": gt_scene_scale,
            #         "R_error": R_error.tolist(),
            #         "CC_error": CC_error,
            #     }
            # )
            
            #! camera metric evaluation
            pred_mats = rearrange(encoder_out['pred_cam']['mats'], 'b v x y -> (b v) x y')
            # pred_mats = rearrange(batch['context']['extrinsics_gt'], 'b v x y -> (b v) x y') #* c2w
            
            R_gt = rearrange(batch['context']['R'], 'b v x y -> (b v) x y')
            T_gt = rearrange(batch['context']['T'], 'b v x -> (b v) x')
            
            # 카메라 저장
            self.pred_mats[batch['scene'][0]] = pred_mats.detach().cpu()
            
            #! CoPoNeRF Style evaluation
            norm_pred = pred_mats[:, :3, 3][1:] / (torch.linalg.norm(pred_mats[:, :3, 3][1:], dim=-1, keepdim=True) + 1e-6)
            norm_gt = T_gt[1:] / (torch.linalg.norm(T_gt[1:], dim=-1, keepdim=True) + 1e-6)

            # Compute cosine similarity for all pairs
            cosine_similarity = torch.sum(norm_pred * norm_gt, dim=-1)

            # Clamp values for arccos
            cosine_similarity_clamped = torch.clamp(cosine_similarity, -1.0, 1.0)

            # Compute angles in degrees for all pairs
            angle_degrees = torch.arccos(cosine_similarity_clamped) * 180 / np.pi

            # Compute average angle degree
            avg_angle_degree = angle_degrees.mean()

            # Compute geodesic distances
            geodesic = compute_geodesic_distance_from_two_matrices(
                pred_mats[..., :3, :3][1:], R_gt[..., :3, :3][1:]
            ) * 180 / np.pi
        
            self.log(f"val/rotation_angle_{tag}", geodesic.mean())
            self.log(f"val/translation_angle_{tag}", avg_angle_degree.mean())
            
        

        if rgb_softmax is not None:
            # Construct comparison image.
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_softmax), "Target (Softmax)"),
            )
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

            # Render projections and construct projection image.
            projections = hcat(*render_projections(
                                    gaussians_softmax,
                                    256,
                                    extra_label="(Softmax)",
                                )[0])
            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )


            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)

            # Run video validation step.
            self.render_video_interpolation(batch)
            self.render_video_wobble(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )


    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample, mode='val') -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["target"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["target"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60, mode=mode)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample, mode='val') -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["target"]["extrinsics"][0, 0],
                # (
                #     batch["context"]["extrinsics"][0, 1]
                #     if v == 2
                #     else batch["target"]["extrinsics"][0, 0]
                # ),
                batch["target"]["extrinsics"][0, 1],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                # (
                #     batch["context"]["intrinsics"][0, 1]
                #     if v == 2
                #     else batch["target"]["intrinsics"][0, 0]
                # ),
                batch["target"]["intrinsics"][0, 0],
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb", mode=mode)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        mode: str = "val",
    ) -> None:
        # Render probabilistic estimate of scene.
        encoder_out = self.encoder(batch["context"], self.global_step, False, render_video=True)
        gaussians_prob = encoder_out['gaussians']
        
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        color, normal, depth, opac = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(color[0], depth_map(depth[0].squeeze()))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]
        
        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"{mode}/video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
