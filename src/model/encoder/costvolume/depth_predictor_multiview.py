import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .modules.unet import UNetModel
from .modules.module_3d import CostRegNetWeight

#TODO: camera pose estimation
from pytorch3d.renderer import PerspectiveCameras, RayBundle
from ...ray_diffusion.utils.rays import (Rays, compute_ndc_coordinates, rays_to_cameras, cameras_to_rays)
from ...ray_diffusion.model.dit import DiT, DiTCfg, ModLN

from .block import BasicBlock, ConditionBlock, ConditionModulationBlock
from .modules.lifting import lifting 

from ...utils.sh_utils import rsh_cart_3


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v): # set 0th view as a canonical view
        cur_view_order = init_view_order[idx:] + init_view_order[:idx] #* make n-1
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach().inverse()
                    @ extrinsics[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach() #* camera to world
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref #* relative pose (from point of view of the target view)
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),] #* contains relative pose and its inverse

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth=>disp) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr



class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussian_anchor_feats=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        predict_only_canonical=False,
        pred_pose=False,
        pose_cfg=None,
        estimation_space="depthmap", #* depthmap or pointmap
        lifting_cfg=None,
        backbone="mvsplat",
        pose_estimation_warmup=0,
        feature_fusion_strategy="both",
        feature_padding=False,
        padding_size=0,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # ablation settings
        # Table 3: base
        self.wo_depth_refine = wo_depth_refine
        # Table 3: w/o cost volume
        self.wo_cost_volume = wo_cost_volume
        # Table 3: w/o U-Net
        self.wo_cost_volume_refine = wo_cost_volume_refine
        
        self.predict_only_canonical = predict_only_canonical
        self.feature_fusion_strategy = feature_fusion_strategy
        self.pose_embedding = pose_cfg.pose_embedding
        self.feature_padding = feature_padding
        self.padding_size = padding_size
        
        if pred_pose:
            self.pose_cfg = pose_cfg
            # TODO: camera pose estimation
            if self.pose_cfg.view_embedding:
                self.pose_cfg.in_channels += 8
            
            self.ray_predictor = DiT(
                    cfg=self.pose_cfg,
                )
            
            if self.pose_cfg.view_embedding:
                # self.view_pos_emb = nn.Parameter(torch.randn(2, 8)) #! only for re10k to DTU
                self.view_pos_emb = nn.Parameter(torch.randn(num_views, 8))
        
        self.pose_estimation_warmup = pose_estimation_warmup
        
        # Cost volume refinement: 2D U-Net
        input_channels = feature_channels + feature_channels
        # if self.predict_only_canonical:
        #     input_channels *= 2 #! reference와 target view를 모두 사용하기 때문에 feature를 두 배로 늘림
        
        
        channels = self.regressor_feat_dim
        
        self.lifting = None
        if lifting_cfg.lifting_switch:
            self.lifting = lifting(channels, latent_res=lifting_cfg.latent_res)
            self.lifting_from = lifting_cfg.lifting_from
        

        pre_modules = [
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU()
        ]
        self.pre_corr_project = nn.Sequential(*pre_modules)
        self.corr_project = UNetModel(
                                image_size=None,
                                in_channels=channels,
                                model_channels=channels,
                                out_channels=channels,
                                num_res_blocks=1,
                                attention_resolutions=costvolume_unet_attn_res,
                                channel_mult=costvolume_unet_channel_mult,
                                num_head_channels=32,
                                dims=2,
                                postnorm=True,
                                num_frames=num_views, #! if predict_only_canonical else 1,
                                condition_num_views=num_views, #! if predict_only_canonical else 1,
                                use_cross_view_self_attn=True,
                                cross_attn_condition=True if self.pose_embedding else False,
                                condition_channels=32 if self.pose_embedding else 0,
                            )
        
        post_corr_out_dim = num_depth_candidates + (1 if predict_only_canonical else 0)
        self.post_corr_project = nn.Conv2d(channels, post_corr_out_dim, 3, 1, 1) #* add 1 for weight
        self.post_mean_var = nn.Conv2d(channels*2, channels, 3, 1, 1)
        
        # cost volume u-net skip connection
        self.regressor_residual = nn.Conv2d(
            input_channels, num_depth_candidates, 1, 1, 0
        )
        self.upsampler_corr = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=self.upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
            

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
        )

        # CNN-based feature upsampler
        proj_in_channels = feature_channels + feature_channels + (feature_channels if "mvsplat" in backbone else 0) + (32 if self.pose_embedding else 0)
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=self.upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        
        # self.proj_feature = nn.Conv2d(
        #     upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        # ) 
        
        if self.predict_only_canonical:
            self.proj_feature = nn.Conv2d(
                upsample_out_channels*2, depth_unet_feat_dim, 3, 1, 1
            )
        else:
            self.proj_feature = nn.Conv2d(
                upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
            )
            
        # Depth refinement: 2D U-Net
        if estimation_space == "depthmap":
            input_channels = depth_unet_feat_dim+3
        else:
            input_channels = depth_unet_feat_dim
            
        # if self.predict_only_canonical:
        #     input_channels = input_channels * 2 #! it was 2
        # input_channels = input_channels + 3 * num_views if self.predict_only_canonical else input_channels + 3
        
        channels = depth_unet_feat_dim 
        if wo_depth_refine:  # for ablations
            self.refine_unet = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            self.pre_refine = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
            )
            self.refine_unet = UNetModel(
                        image_size=None,
                        in_channels=channels,
                        model_channels=128,
                        out_channels=depth_unet_feat_dim,
                        num_res_blocks=1, 
                        attention_resolutions=depth_unet_attn_res,
                        channel_mult=depth_unet_channel_mult,
                        num_head_channels=8,
                        dims=2,
                        postnorm=True,
                        condition_num_views=num_views, #! if predict_only_canonical else 1,
                        num_frames=num_views if not self.predict_only_canonical else 1,
                        use_cross_view_self_attn=True,
                        cross_attn_condition=True if self.pose_embedding else False,
                        condition_channels=32 if self.pose_embedding else 0,
                    )
            self.post_refine = nn.Conv2d(depth_unet_feat_dim, post_corr_out_dim, 3, 1, 1)
            self.post_refine_mean_var = nn.Conv2d(channels*2, channels, 3, 1, 1)
        
        # Gaussians prediction: covariance, color
        # gau_in = depth_unet_feat_dim + feature_channels * num_views if self.predict_only_canonical else depth_unet_feat_dim + feature_channels
        if self.predict_only_canonical:
            gau_in = (depth_unet_feat_dim // 2) + feature_channels
            
        else:
            gau_in = depth_unet_feat_dim + feature_channels
        
        # Gaussians prediction: centers, opacity
        if not wo_depth_refine:
            self.disps_channels = depth_unet_feat_dim // 4
            disps_models = [
                nn.Conv2d(self.disps_channels, self.disps_channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(self.disps_channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
            ]
            self.to_disparity = nn.Sequential(*disps_models)
        
        self.to_gaussians = nn.Sequential(
            nn.Linear(depth_unet_feat_dim - self.disps_channels, gaussian_anchor_feats),
            nn.ReLU(),
            nn.Linear(gaussian_anchor_feats, gaussian_raw_channels),
        )
        
        
        self.gaussian_raw_channels = gaussian_raw_channels
        self.gaussian_anchor_feats = gaussian_anchor_feats
        
        
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.dir_norm = ModLN(128, 16*2, eps=1e-6)
        

    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
        use_pred_cams=False,
        use_gt_cams: bool = False,
        pred_cams=None,
        predict_only_canonical=False,
        estimation_space="depth",
        homo_warp=True,
        src_view=2,
        context=None,
        others=None,
        anchor_head_with_mlp=False,
        global_step=None,
        target_feats=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""
        
        # format the input
        b, v, c, h, w = features.shape
        pointmap1, pointmap2, depths, rays = None, None, None, None
        v_all = v
        
        '''
        🇵​​​​​🇴​​​​​🇸​​​​​🇪​​​​​ 🇪​​​​​🇸​​​​​🇹​​​​​🇮​​​​​🇲​​​​​🇦​​​​​🇹​​​​​🇮​​​​​🇴​​​​​🇳​​​​​
        '''
        if use_pred_cams:

            if target_feats is not None:
                features = torch.cat([features, target_feats], dim=1)
                b, v_all, _, _, _ = features.shape
            
            # view number positional embedding
            # This section is intended to handle the positional embedding for the view number.
            if self.pose_cfg.view_embedding:
                features_rays = torch.cat([features, self.view_pos_emb[:v_all].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) .expand(b, -1, -1, h, w)], dim=2)
            else:
                features_rays = features
            
            if self.pose_embedding:
                rays = self.ray_predictor(features_rays.clone()) #* input: (b, v, c, h, w), output: (b, v, c, h, w)
            else:
                rays = self.ray_predictor(features_rays.clone().detach())
            
            rays = rearrange(rays, "b v c h w -> (b v) c h w", b=b, v=v_all)
            # viewing direction
            
            # features = torch.cat([features, rays], dim=1)
            #! wrong way to predict camera pose (batching is not supported in the current implementation of rays_to_cameras)
            intrinsics_all = repeat(intrinsics[:, 0:1, ...], 'b () x y -> b v_all x y', v_all=v_all)
            pred_cam = rays_to_cameras(
                Rays.from_spatial(rays.clone()),
                None,
                num_patches_x=self.pose_cfg.num_patches_x,
                num_patches_y=self.pose_cfg.num_patches_y,
                focal_lengths = intrinsics_all[0, :, [0,1], [0,1]],
                principal_points = intrinsics_all[0, :, [0,1], [2,2]],
            )

            rays = torch.einsum('bchw->bhwc', rays)
            # rays = self.ray_to_plucker(rays)
            
            # b, v_all, c, _, _ = ray_features.shape
            pred_cam.R = pred_cam.R.reshape(b, v_all, 3, 3) #* rotation that transforms from world to camera
            pred_cam.T = pred_cam.T.reshape(b, v_all, 3) #* translation that transforms from world to camera

            #TODO: convert pred_cam (PerspectiveCameras) to Float[Tensor, "batch view 4 4"],
            if isinstance(pred_cam, PerspectiveCameras):
                # get extrinsics from pred_cam R and T
                pred_extrinsics = torch.eye(4, device=features.device).unsqueeze(0).expand(b, v_all, 4, 4).clone()
                pred_extrinsics[:, :, :3, :3] = pred_cam.R #* rotation that transforms from world to camera
                pred_extrinsics[:, :, :3, 3] = pred_cam.T  #* translation that transforms from world to camera
            else:
                pred_extrinsics = pred_cam
            pred_cam2world = pred_extrinsics.inverse()
            
            rays_all, pred_cam2world_all = None, None
            if target_feats is not None:
                rays_all = rays.clone()
                pred_cam2world_all = pred_cam2world.clone()
                
                rays = rays[:v]
                pred_cam2world =  pred_cam2world[:v]
                features = features[:, :v]
                
            if self.pose_embedding:
                feats_dir = torch.cat((rsh_cart_3(rays[..., :3]),rsh_cart_3(rays[..., 3:6])),dim=-1).clone()
                #TODO: camera embedding
                features = rearrange(features, "b v c h w -> (b v) c h w")
                features =  torch.einsum('bchw->bhwc',features)
                features = self.dir_norm(features, feats_dir)
                features = torch.einsum('bhwc->bchw',features)
                features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
            
            rays = torch.einsum('bhwc->bchw', rays)
        
        if use_gt_cams:
            #!GT camera pose embedding
            gt_rays = context['rays_down']
            gt_rays = rearrange(gt_rays, "b v h w c -> (b v) h w c", b=b, v=v)
            # viewing direction
            gt_rays = self.ray_to_plucker(gt_rays)
            feats_dir = torch.cat((rsh_cart_3(gt_rays[...,:3]),rsh_cart_3(gt_rays[...,3:6])),dim=-1)
            
            #TODO: camera embedding
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features =  torch.einsum('bchw->bhwc',features)
            features = self.dir_norm(features, feats_dir)
            features = torch.einsum('bhwc->bchw',features)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
        
        if global_step < self.pose_estimation_warmup:
            return None, None, None, None, {"mats": pred_cam2world, "rays": rays, "cameras": pred_cam}, None
        
        #! feature padding
        if self.feature_padding:
            features = F.pad(features, 
                (self.padding_size, self.padding_size, self.padding_size, self.padding_size), 
                mode='constant', value=1)
        
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics if not use_pred_cams else pred_cam2world,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )
        if cnn_features is not None:
            cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")

        # cost volume constructions
        feat01 = feat_comb_lists[0]
        raw_correlation_in_lists = []
        for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
            # sample feat01 from feat10 via camera projection
            feat01_warped = warp_with_pose_depth_candidates(
                feat10,
                intr_curr,
                pose_curr,
                1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                warp_padding_mode="zeros", #! changed
                )  # [B, C, D, H, W]
            # calculate similarity
            raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(1) / (c**0.5)  # [vB, D, H, W]
            raw_correlation_in_lists.append(raw_correlation_in) #* canonical view와의 유사도를 계산
        
        # average all cost volumes
        if self.lifting is not None:
            raw_correlation_in_3d = torch.mean( #* reference와의 유사도를 평균
                torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=True
            )  # [vxb d, h, w]
        
        raw_correlation_in_2d = torch.mean( #* reference와의 유사도를 평균
            torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
        )

        if predict_only_canonical:
            #! 3D volume
            volume_geo = None
            if self.lifting is not None:
                
                if self.lifting_from == "cost_volume":
                    raw_correlation_in_3d = rearrange(raw_correlation_in_3d, "c v d h w -> (v c) d h w")
                    raw_correlation_in_3d = self.upsampler_corr(raw_correlation_in_3d)
                    raw_correlation_in_3d = rearrange(raw_correlation_in_3d, "(v c) d h w -> v c d h w", v=v, c=1)
                    vol_feat, weight = self.cost_reg_2(raw_correlation_in_3d)
                    vol_feat = (vol_feat * weight).sum(dim=0, keepdim=True) / weight.sum(dim=0, keepdim=True)
                    
                    #! 2D-3D interaction
                    vol_feat = self.lifting.forward_cv(vol_feat, raw_correlation_in_2d)
                
                elif self.lifting_from == "latent": #! current option
                    vol_feat = self.lifting.forward_latent(rearrange(feat01, "(b v) c h w -> b v c h w", b=b, v=v))
                    
            raw_correlation_in_2d = torch.cat([raw_correlation_in_2d, feat01], dim=1)
            raw_correlation_in_2d = self.pre_corr_project(raw_correlation_in_2d)
            
            if self.pose_embedding:
                feats_dir_in = rearrange(feats_dir, '(b v) h w c -> b (h w) v c', b=b, v=v)
                corr_feat = self.corr_project(raw_correlation_in_2d, context=feats_dir_in) #* x: [B, C, D, H, W], feature  # y: [B, H*W, D, C]
            else:
                corr_feat = self.corr_project(raw_correlation_in_2d, context=None)
            corr_feat_views = corr_feat.clone()
            
            if self.feature_fusion_strategy == "weighted_sum":
                #! approach 2: weighted sum based feature
                corr_feat_weight = self.post_corr_project(corr_feat)
                corr_feat_1, corr_weight = corr_feat_weight[:, :-1], corr_feat_weight[:, -1:].sigmoid()
                raw_correlation_weighted_sum = (corr_feat_1 * corr_weight).sum(dim=0, keepdim=True) / corr_weight.sum(dim=0, keepdim=True)
                volume_geo = raw_correlation_weighted_sum
            elif self.feature_fusion_strategy == "mean_variance":
                #! approach 1. mean variance based feature
                corr_feat = rearrange(corr_feat, "(b v) d h w -> b v d h w", b=b)
                corr_feat_mean = corr_feat.mean(dim=1, keepdim=False) # (b, 1, d, h, w)
                corr_feat_var = corr_feat.var(dim=1, keepdim=False) # (b, 1, d, h, w)
                corr_feat_mean_var = torch.cat((corr_feat_mean, corr_feat_var), dim=1) # (b, 2*d, h, w)
                corr_feat_mean_var = self.post_mean_var(corr_feat_mean_var)
                volume_geo = corr_feat_mean_var
            elif self.feature_fusion_strategy == "both":
                corr_feat_weight = self.post_corr_project(corr_feat)
                corr_feat_1, corr_weight = corr_feat_weight[:, :-1], corr_feat_weight[:, -1:].sigmoid()
                raw_correlation_weighted_sum = (corr_feat_1 * corr_weight).sum(dim=0, keepdim=True) / corr_weight.sum(dim=0, keepdim=True)
                
                corr_feat = rearrange(corr_feat, "(b v) d h w -> b v d h w", b=b)
                corr_feat_mean = corr_feat.mean(dim=1, keepdim=False) # (b, 1, d, h, w)
                corr_feat_var = corr_feat.var(dim=1, keepdim=False) # (b, 1, d, h, w)
                corr_feat_mean_var = torch.cat((corr_feat_mean, corr_feat_var), dim=1) # (b, 2*d, h, w)
                corr_feat_mean_var = self.post_mean_var(corr_feat_mean_var)
                volume_geo = raw_correlation_weighted_sum + corr_feat_mean_var
        else:
            #! approach 2: weighted sum based feature
            raw_correlation_in_2d = torch.cat([raw_correlation_in_2d, feat01], dim=1)
            raw_correlation_in_2d = self.pre_corr_project(raw_correlation_in_2d)
            # raw_correlation_in_2d = rearrange(raw_correlation_in_2d, '(b v) c h w -> b c v h w', b=b, v=v)
            rays_in = rearrange(rays, '(b v) c h w -> b (h w) v c', b=b, v=v)
            corr_feat = self.corr_project(raw_correlation_in_2d, context=rays_in) #* x: [B, C, D, H, W], feature  # y: [B, H*W, D, C]
            corr_feat = self.post_corr_project(corr_feat)
            
            volume_geo = corr_feat + feat01
            
                  
        '''
        🇩​​​​​🇪​​​​​🇵​​​​​🇹​​​​​🇭​​​​​ 🇪​​​​​🇸​​​​​🇹​​​​​🇮​​​​​🇲​​​​​🇦​​​​​🇹​​​​​🇮​​​​​🇴​​​​​🇳​​​​​
        '''
        if v != intrinsics.shape[1]:
            # extend the length to v by copying intrinsics[:, 0]
            disp_candi_curr = disp_candi_curr[:v]
            feat01 = feat01[:v]
            if cnn_features is not None:
                cnn_features = cnn_features[:v]
            extra_info['images'] = extra_info['images'][:v]

        # softmax to get coarse depth and density
        if estimation_space == "depthmap":
            pdf = F.softmax(
                self.depth_head_lowres(volume_geo), dim=1
            )  # [2xB, D, H, W]
            coarse_disps = (disp_candi_curr * pdf).sum(
                dim=1, keepdim=True
            )  # (vb, 1, h, w)
            pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
            pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
            fullres_disps = F.interpolate(
                coarse_disps if not predict_only_canonical else coarse_disps[0:1, ...],
                scale_factor=self.upscale_factor,
                mode="bilinear",
                align_corners=True,
            )
            
            # depth refinement
            if cnn_features is not None:
                if self.pose_embedding:
                    feats_dir = rearrange(feats_dir, "(b v) h w c -> (b v) c h w", b=b)
                    proj_feat = torch.cat([feat01, cnn_features, feats_dir, corr_feat_views], dim=1) #* 128 + 128 + 32 + 128
                else:
                    proj_feat = torch.cat([feat01, cnn_features, corr_feat_views], dim=1)
                proj_feat_in_fullres = self.upsampler(proj_feat) #* V C H W
            else:
                proj_feat = torch.cat([feat01, feats_dir, corr_feat_views], dim=1)
                proj_feat_in_fullres = self.upsampler(proj_feat) #* V C H W
                
            if predict_only_canonical: # d
                # proj_feat_in_fullres = rearrange(proj_feat_in_fullres, "(b v) c h w -> b (v c) h w", v=v)
                proj_feat_in_fullres = torch.cat([proj_feat_in_fullres, extra_info['images']], dim=1)
                
                #! 2-view only setting
                # proj_feat_in_fullres = rearrange(proj_feat_in_fullres, "(b v) c h w -> b (v c) h w", v=v)
                # refine_in = proj_feat_in_fullres
                
                refine_in = self.pre_refine(proj_feat_in_fullres)
                if self.pose_embedding:
                    refine_feat = self.refine_unet(refine_in, context=feats_dir_in)
                else:
                    refine_feat = self.refine_unet(refine_in, context=None)
                     
                if self.feature_fusion_strategy == "weighted_sum":
                    refine_feat_weight = self.post_refine(refine_feat)
                    refine_feat, refine_weight = refine_feat_weight[:, :-1], refine_feat_weight[:, -1:].sigmoid() #* (v, 31, h, w), (v, 1, h, w)
                    refine_feat_weighted_sum = (refine_feat * refine_weight).sum(dim=0, keepdim=True) / refine_weight.sum(dim=0, keepdim=True)
                    feature_volume = refine_feat_weighted_sum
                elif self.feature_fusion_strategy == "mean_variance":
                    refine_out_mean = refine_feat.mean(dim=0, keepdim=True)
                    refine_out_var = refine_feat.var(dim=0, keepdim=True)
                    refine_feat_mean_var = torch.cat((refine_out_mean, refine_out_var), dim=1)
                    refine_feat_mean_var = self.post_refine_mean_var(refine_feat_mean_var)
                    feature_volume = refine_feat_mean_var
                elif self.feature_fusion_strategy == "both":
                    refine_feat_weight = self.post_refine(refine_feat)
                    refine_feat, refine_weight = refine_feat_weight[:, :-1], refine_feat_weight[:, -1:].sigmoid() #* (v, 31, h, w), (v, 1, h, w)
                    refine_feat_weighted_sum = (refine_feat * refine_weight).sum(dim=0, keepdim=True) / refine_weight.sum(dim=0, keepdim=True)
                    refine_out_mean = refine_feat.mean(dim=0, keepdim=True)
                    refine_out_var = refine_feat.var(dim=0, keepdim=True)
                    refine_feat_mean_var = torch.cat((refine_out_mean, refine_out_var), dim=1)
                    refine_feat_mean_var = self.post_refine_mean_var(refine_feat_mean_var)
                    feature_volume = refine_feat_weighted_sum + refine_feat_mean_var
            else:
                proj_feat_in_fullres = torch.cat([proj_feat_in_fullres, extra_info['images']], dim=1)
                feature_volume = self.refine_unet(proj_feat_in_fullres)
            
            if predict_only_canonical:
                v=1
            else:
                v=v
            
            if self.wo_depth_refine:
                densities = repeat(
                    pdf_max,
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )
                depths = 1.0 / fullres_disps
                depths = repeat(
                    depths,
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )
                
            else:
                # delta fine depth and density
                delta_disps_density = self.to_disparity(feature_volume[:, :self.disps_channels])
                delta_disps, raw_densities = delta_disps_density.split(
                    gaussians_per_pixel, dim=1
                )

                # combine coarse and fine info and match shape
                densities = repeat(
                    F.sigmoid(raw_densities),
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )

                fine_disps = (fullres_disps + delta_disps).clamp(
                    1.0 / rearrange(far, "b v -> (v b) () () ()")[0:1],
                    1.0 / rearrange(near, "b v -> (v b) () () ()")[0:1],
                )
                depths = 1.0 / fine_disps
                depths = repeat(
                    depths,
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )
        
        else:
            raise ValueError(f"Unknown estimation_space: {estimation_space}")
        
        '''
        🇦​​​​​🇳​​​​​🇨​​​​​🇭​​​​​🇴​​​​​🇷​​​​​ 🇬​​​​​🇦​​​​​🇺​​​​​🇸​​​​​🇸​​​​​🇮​​​​​🇦​​​​​🇳​​​​​ 🇵​​​​​🇷​​​​​🇪​​​​​🇩​​​​​🇮​​​​​🇨​​​​​🇹​​​​​🇴​​​​​🇷​​​​​
        predict anchor gaussians which will be used to estimate neighboring gaussians with 
        '''
        # gaussians head

        anchor_feats = None
        if predict_only_canonical:
            if anchor_head_with_mlp:
                raw_gaussians_in = [feature_volume[:, self.disps_channels:]] #TODO: 127 - 32 = 95
                
                #* image wise super resolution
                raw_gaussians_in = rearrange(torch.cat(raw_gaussians_in, dim=1), "b c h w -> b h w c") #* shape: (2, 425, 224, 224)
                raw_gaussians = self.to_gaussians(raw_gaussians_in) #* 57 (depth 1, density 1, scale 3, rotation 4, sh 48) parameters for gaussian params and 32 for anchor features (total 89)
                
                raw_gaussians = rearrange(raw_gaussians, "b h w c -> b c h w")
                
                anchor_gaussians = raw_gaussians[:, :self.gaussian_raw_channels]
                anchor_feats = raw_gaussians[:, self.gaussian_raw_channels:]
                
                anchor_gaussians = rearrange(anchor_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b)
                anchor_feats = rearrange(anchor_feats, "(v b) c h w -> b v (h w) c", v=v, b=b)
            else:
                anchor_feats = [feature_volume[:, self.disps_channels:]] #TODO: 127 - 32 = 95
                anchor_feats = rearrange(torch.cat(anchor_feats, dim=1), "b c h w -> b h w c")
                anchor_gaussians = self.to_gaussians(anchor_feats)
                
                # anchor_feats_all_views = torch.cat([anchor_feats.expand(proj_feat_in_fullres.shape[0], -1, -1, -1), proj_feat_in_fullres], dim=1)
                anchor_feats = rearrange(anchor_feats, "(v b) h w c -> b v (h w) c", b=b)
                
            pred_pose = {"mats": pred_cam2world, "rays": rays, "cameras": pred_cam, 'rays_all': rays_all, 'pred_cam2world_all': pred_cam2world_all} if use_pred_cams else None
            return depths, densities, anchor_gaussians, anchor_feats, pred_pose, volume_geo
            
        else:
            anchor_feats = [feature_volume[:, self.disps_channels:]] #TODO: 127 - 32 = 95
            anchor_feats = rearrange(torch.cat(anchor_feats, dim=1), "b c h w -> b h w c")
            anchor_gaussians = self.to_gaussians(anchor_feats)
            
            # anchor_feats_all_views = torch.cat([anchor_feats.expand(proj_feat_in_fullres.shape[0], -1, -1, -1), proj_feat_in_fullres], dim=1)
            anchor_feats = rearrange(anchor_feats, "(v b) h w c -> b v (h w) c", b=b)
        
            pred_pose = {"mats": pred_cam2world, "rays": rays, "cameras": pred_cam, 'rays_all': rays_all, 'pred_cam2world_all': pred_cam2world_all } if use_pred_cams else None
            return depths, densities, anchor_gaussians, anchor_feats, pred_pose, volume_geo


    def extract_intrinsics(self,intrinsics):
        """
        Extracts the values of fx, fy, cx, and cy from intrinsic matrices.

        Args:
        intrinsics (numpy.ndarray): An array of shape (batch_size, num_view, 4, 4) containing intrinsic matrices.

        Returns:
        tuple: A tuple containing four lists (fx_list, fy_list, cx_list, cy_list) with shapes (batch_size, num_view).
        """
        batch_size, _, _, _ = intrinsics.shape

        fx_list = []
        fy_list = []
        cx_list = []
        cy_list = []

        for i in range(batch_size):
        
            fx_list.append(intrinsics[i, 0, 0, 0])
            fy_list.append(intrinsics[i, 0, 1, 1])
            cx_list.append(intrinsics[i, 0, 0, 2])
            cy_list.append(intrinsics[i, 0, 1, 2])
        
        fx_list = torch.stack(fx_list).reshape(batch_size, 1)
        fy_list = torch.stack(fy_list).reshape(batch_size, 1)
        cx_list = torch.stack(cx_list).reshape(batch_size, 1)
        cy_list = torch.stack(cy_list).reshape(batch_size, 1)
        
        return [fx_list, fy_list, cx_list, cy_list]
    
    def ray_to_plucker(self, rays):
        origin, direction = rays[...,:3], rays[...,3:6]
        # Normalize the direction vector to ensure it's a unit vector
        direction = F.normalize(direction, p=2.0, dim=-1)
        
        # Calculate the moment vector (M = O x D)
        moment = torch.cross(origin, direction, dim=-1)
        
        # Plucker coordinates are L (direction) and M (moment)
        return torch.cat((direction, moment),dim=-1)
    
    
    