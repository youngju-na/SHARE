from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple

import math
import tqdm
import torch
from einops import (rearrange, repeat)
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict
import numpy as np

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview, build_dino_backbone, BackboneOutBlock
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding

from ..ray_diffusion.model.dit import DiT, DiTCfg

from .costvolume.modules.encoder import CrossViewEncoder
from .costvolume.modules.depth_module import BackprojectDepth

from .mast3r.model import AsymmetricMASt3R as MAST3R
from .dust3r.image_pairs import make_pairs
from .dust3r.utils.device import to_cpu, collate_with_cat
from .dust3r.inference import check_if_same_size, loss_of_one_batch



import torch.nn.functional as F
from einops import (rearrange, repeat, reduce)

from src.visualization.drawing.cameras import compute_aabb

import random
random.seed(0)

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int

@dataclass
class LiftingCfg:
    lifting_switch: bool
    lifting_from: str
    latent_res: int
    radius: float
    feature_reduction: str
    
@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    lifting: LiftingCfg
    ray: DiTCfg
    gaussians_per_pixel: int
    gaussian_anchor_feats: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    pred_campose: bool
    gt_campose: bool
    predict_only_canonical: bool
    predict_offset: bool
    predict_anchor_gaussian: bool
    num_offset_gaussian: int
    use_monoN_loss: bool
    rendering_units: List[str]
    use_mast3r: bool
    mast3r_predict_pointmap: bool
    feature_padding: bool
    padding_size: int
    master_weights_path: str
    backbone: Literal["mvsplat", "dino", "mast3r", "mvsplat+mast3r"]
    estimation_space: Literal["depthmap", "pointmap"]
    backbone_fix: bool
    pose_estimation_warmup: int
    feature_fusion_strategy: Literal["both", "weighted_sum", "mean_variance"]
    
        
class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg, dataset_cfg) -> None:
        super().__init__(cfg)
        
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        
        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
            
   
        if "mvsplat" in self.cfg.backbone:
            self.backbone = BackboneMultiview(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
            )
            
            ckpt_path = cfg.unimatch_weights_path
            if get_cfg().mode == 'train':
                if cfg.unimatch_weights_path is None:
                    print("==> Init multi-view transformer backbone from scratch")
                else:
                    print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                    unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                    updated_state_dict = OrderedDict(
                        {
                            k: v
                            for k, v in unimatch_pretrained_model.items()
                            if k in self.backbone.state_dict()
                        }
                    )
                    # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                    is_strict_loading = not cfg.wo_backbone_cross_attn
                    self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)
            
                
        if "dino" in self.cfg.backbone: 
            self.backbone, self.down_rate, self.backbone_dim = build_dino_backbone()
            self.backbone_out_dim=self.backbone_dim #! change
            self.backbone_out = BackboneOutBlock(in_dim=self.backbone_dim, out_dim=self.backbone_out_dim)
            
        
        if "mast3r" in self.cfg.backbone:
            self.master = MAST3R.from_pretrained(self.cfg.master_weights_path)
            for param in self.master.parameters():
                param.requires_grad = False
            
        # self.encoder = CrossViewEncoder(encoder_layers=2, in_dim=cfg.d_feature)
        
        #! anchor gaussian을 예측하는 경우에는 모든 Gaussian parameter 수 만큼 예측하지만, 그렇지 않을 경우 오직 2개 (xy offset만 예측하도록 함)
        self.d_in = self.gaussian_adapter.d_in if cfg.predict_anchor_gaussian else 0
        
        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.d_in + 2) if cfg.estimation_space == "depthmap" else cfg.num_surfaces * (self.d_in + 1),
            gaussian_anchor_feats=cfg.gaussian_anchor_feats,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            predict_only_canonical=cfg.predict_only_canonical,
            estimation_space=cfg.estimation_space,
            pred_pose=cfg.pred_campose,
            pose_cfg=cfg.ray,
            lifting_cfg=cfg.lifting,
            backbone=cfg.backbone,
            pose_estimation_warmup=cfg.pose_estimation_warmup,
            feature_fusion_strategy=cfg.feature_fusion_strategy,
            feature_padding=cfg.feature_padding,
            padding_size=cfg.padding_size,
        )

        
        self.pixel_shuffle_2 = nn.PixelShuffle(2)     
        self.pixel_shuffle_4 = nn.PixelShuffle(4)
        
        mvsplat_feat_in = 128
        dino_feat_in = 768        
        mast3r_feat_in = 1024
        if self.cfg.backbone == "mvsplat":
            self.pre_conv = nn.Conv2d(mvsplat_feat_in, 128, 1)
        elif self.cfg.backbone == "dino":
            self.pre_conv = nn.Conv2d(dino_feat_in, 128, 1)
        elif self.cfg.backbone == "mast3r": 
            mast3r_feat_in += 128 if self.cfg.mast3r_predict_pointmap else 0
            self.pre_conv = nn.Conv2d(mast3r_feat_in // 16, mast3r_feat_in // 8, 1)
        elif self.cfg.backbone == "mvsplat+mast3r":
            self.pre_conv = nn.Conv2d(mvsplat_feat_in + mvsplat_feat_in, 128, 1)
        
        #MLP network that takes anchor features and predicts offsets
        # input channel: cfg.gaussian_anchor_feats, output_channels: 3 * num_neural_gaussian
        self.num_offset_gaussian = self.cfg.num_offset_gaussian
        self.to_offset = nn.Sequential(
            nn.Linear(cfg.gaussian_anchor_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * self.num_offset_gaussian),
        )
        
        self.offset_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(3), cfg.d_feature),
            )
        self.encoding_dim = cfg.d_feature
        self.gaussian_dims = 55 + 1 #* density
        d_lifting_ch = cfg.d_feature if cfg.lifting.lifting_switch else 0
        self.to_neighbor_gaussians = nn.Sequential(
            nn.Linear(cfg.gaussian_anchor_feats + self.encoding_dim + d_lifting_ch, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.gaussian_dims),
        )
        
        if "mast3r" in self.cfg.backbone:
            self.upsampler = nn.Sequential(
                nn.Conv2d(1024, 128, 3, 1, 1),
                nn.Upsample(
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.GELU(),
            )
            self.downsampler = nn.Sequential(
                nn.Conv2d(24+1, 128, 3, 1, 1),
                nn.MaxPool2d(16),
            )
        
        
    def extract_feature(self, x, return_h_w=False):
        b, _, h_origin, w_origin = x.shape
        out = self.backbone.get_intermediate_layers(x, n=1)[0]
        h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
        dim = out.shape[-1]
        out = out.reshape(b, h, w, dim).permute(0,3,1,2)
    
        return out
    
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def build_feature_volume(self, batch, source_imgs_feat):
        return self.feature_volume(source_imgs_feat, batch)
    
    def build_mvs_volume(self, batch, feature_volume, stage=1):
        return self.feature_volume(batch, feature_volume, stage=stage)
    
    def build_pairs(self, imgs, proj_mats, depth_values):
        
        #! imgs: B, N, 3, H, W
        N = imgs.shape[1]
        # if N == 3: [0, 1, 2], [1, 2, 0], [2, 0, 1]
        # if N == 4: [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2] 
        nums = [i for i in range(N)]
        all_combinations = []
        for i in range(N):
            all_combinations.append(nums[i:] + nums[:i])
        all_combinations = np.array(all_combinations)
        
        #indexing from images according to all_combinations using gather
        imgs = imgs[:, all_combinations]
        
        for stage in ['stage1', 'stage2', 'stage3']:
            proj_mats[stage] = rearrange(proj_mats[stage][:,all_combinations], "B N V dim2 dim4 dim4_2 -> (B N) V dim2 dim4 dim4_2", B=imgs.shape[0], N=N,V=N)
        
        imgs = rearrange(imgs, "B N V C H W -> (B N) V C H W", N=N)
        
        depth_values = depth_values.expand(imgs.shape[0], -1)
        return imgs, proj_mats, depth_values
    
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        render_video: bool = False,
        return_pointmap: bool = False,
        target=None,
    ):
        device = context["image"].device
        b, src_v, _, h, w = context["image"].shape
        v = src_v
                
        tgt_v = 0
        if target is not None:
            _, tgt_v, _, _, _ = target['image'].shape
            v = src_v + tgt_v
        
        if render_video:
            pass #TODO: fill in the code for rendering video

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        
        cnn_features = None
        backbone_input = context["image"] if target is None else torch.cat([context['image'], target['image']], dim=1)
        
        if "mvsplat" in self.cfg.backbone:
            features, cnn_features = self.backbone( #* BackboneMultiView
               backbone_input,
                attn_splits=self.cfg.multiview_trans_attn_split,
                return_cnn_features=True,
                epipolar_kwargs=epipolar_kwargs,
            )
        elif "dino" in self.cfg.backbone:
            # 2D per-view feature extraction
            imgs = rearrange(context['image'], 'b t c h w -> (b t) c h w')
            if self.cfg.backbone_fix:
                with torch.no_grad():
                    features = self.extract_feature(imgs) # [b*t,c=768,h,w]
            else:
                features = self.extract_feature(imgs)

            features = self.backbone_out(features)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
        
        #! additionally use mast3r features
        if  "mast3r" in self.cfg.backbone:
            if self.cfg.mast3r_predict_pointmap:
                duster_image = [{'img': context['image_normalized'][0][i:i+1]} for i in range(v)]
                pairs = make_pairs(duster_image, symmetrize=False) #! symmerize was True before
                
                # first, check if all images have the same size
                multiple_shapes = not (check_if_same_size(pairs))    
                batch_size = 1
                
                result = []
                for i in range(0, len(pairs), batch_size):
                    res = loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), self.master, None, device)
                    result.append(res)
                
                result = collate_with_cat(result)
            
                mast_can_feat = rearrange(result['feat1'].mean(dim=0, keepdim=True), 'b (h w) c -> b c h w', h=int(h/16), w=int(w/16)) #*shape: (1, c, 14, 14)
                msat_src_feats = rearrange(result['feat2'], 'b (h w) c -> b c h w', h=int(h/16), w=int(w/16)) #*shape: (v, c, 14, 14)        
                mast_feats = torch.cat([mast_can_feat, msat_src_feats], dim=0) #*shape: (v+1, c, 14, 14)

                mast_desc = torch.cat([result['pred1']['desc'].mean(dim=0, keepdim=True), result['pred2']['desc']], dim=0)
                mast_conf = torch.cat([result['pred1']['conf'].mean(dim=0, keepdim=True), result['pred2']['conf']], dim=0).unsqueeze(-1)

                features = rearrange(mast_feats, "(b v) c h w -> b v c h w", b=b, v=v)
                img_desc = rearrange(self.downsampler(rearrange(torch.cat([mast_desc, mast_conf], dim=-1), "v h w c -> v c h w")), "(b v) c h w -> b v c h w", b=b, v=v)
            
            else:
                mast_input = rearrange(context['image_normalized'], 'b v c h w -> (b v) c h w')
                if target is not None:
                    mast_input = torch.cat([mast_input, rearrange(target['image_normalized'], 'b v c h w -> (b v) c h w')], dim=0)
                mast_feats = self.master._encode_image(mast_input, true_shape=(h, w))[0]
                mast_feats = rearrange(mast_feats, "(b v) (h w) c -> (b v) c h w", b=b, v=v, h=int(h/16), w=int(w/16))
                
        '''
        ################################
        #||                            ||
        #||      Gaussian Adapter      ||
        #||                            ||
        ################################
        '''

        if self.cfg.estimation_space == "depthmap":
            if "mast3r" in self.cfg.backbone:
                mast_feats = self.upsampler(mast_feats) #* mast3r은 1/16로 downsample 되어 있으므로 4배 업샘플링
                mast_feats = rearrange(mast_feats, "(b v) c h w -> b v c h w", b=b, v=v)
                features = torch.cat([features, mast_feats], dim=2)
                
            features = self.pre_conv(rearrange(features, 'b v c h w -> (b v) c h w'))
            features = rearrange(features, '(b v) c h w -> b v c h w', v=v)  
            
            target_feats = None
            if target is None:
                in_feats = features
            else:
                in_feats = features[:, :src_v]
                target_feats = features[:, src_v:]
            
            
            extra_info = {}
            extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w").clone().detach()
            extra_info["scene_names"] = scene_names
            gpp = self.cfg.gaussians_per_pixel
            

            depths, densities, raw_gaussians, anchor_feats, pred_cam, vol_feat_3d = self.depth_predictor(
                in_feats,
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                gaussians_per_pixel=gpp,
                deterministic=deterministic,
                extra_info=extra_info,
                cnn_features=cnn_features[:, :src_v],
                use_pred_cams=True if self.cfg.pred_campose else False,
                use_gt_cams=True if self.cfg.gt_campose else False,
                predict_only_canonical=self.cfg.predict_only_canonical,
                estimation_space=self.cfg.estimation_space,
                src_view=v,
                context=context,
                global_step=global_step,
                target_feats=target_feats,
            )
            
            gaussians_all, gaussians_anchor, gaussians_offset = None, None, None
            if global_step < self.cfg.pose_estimation_warmup:
                return {"gaussians":gaussians_all, "gaussians_anchor": gaussians_anchor, "gaussians_offset": gaussians_offset, "pred_cam": pred_cam, 'mickey_out': None}
        
            # Convert the features and depths into Gaussians.
            if self.cfg.predict_only_canonical:
                v = 1
            _, feat_h, feat_w, _ = raw_gaussians.shape
            raw_gaussians = rearrange(raw_gaussians, '(b v) h w c -> b v (h w) c', b=b, v=v)
            gaussians = rearrange(
                raw_gaussians, #* (1, 1, 50176, 57)
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            ) #* (1, 1, 50176, 57)

            #TODO: increase offset_xy to make the number of gaussians bigger
            xy_ray, _ = sample_image_grid((feat_h, feat_w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((feat_w, feat_h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
            
            scale_factor = feat_h / h #* pixel-aligned feature resolution이 image resolution하고 다른 경우 intrinsic 변경을 위해 필요
            
            anchor_gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths, #* (1, 1, 50176, 1, 1)
                densities, #* 1, 1 50176, 1, 1
                rearrange(
                    gaussians[..., 2:], 
                    "b v r srf c -> b v r srf () c",
                ), #* (1, 1, 50176, 1, 1, 55)
                (h, w),
                use_pred_cams=True if self.cfg.pred_campose else False,
                pred_cams=rearrange(pred_cam['mats'], "b v i j -> b v () () () i j") if self.cfg.pred_campose else None,
                predict_only_canonical=self.cfg.predict_only_canonical,
                estimation_space=self.cfg.estimation_space,
                xyz_only=(not self.cfg.predict_anchor_gaussian), #* TRUE
                scale_factor=scale_factor,
                extra_info=extra_info,
                learn_sh_residual_from_canoncal_rgb=self.cfg.gaussian_adapter.learn_sh_residual_from_canoncal_rgb,
                gaussian_2d_scale=self.cfg.gaussian_adapter.gaussian_2d_scale,
            )

        '''
        ################################
        ||                            ||
        ||     OFFSET PREDICTION      ||
        ||                            ||
        ################################
        '''
        
        offset_xyz = None
        #TODO: offset 예측하기 (3 + 55 = 58)
        if self.cfg.predict_offset and self.cfg.predict_only_canonical:
            num_points = self.num_offset_gaussian
            # anchor_feats = rearrange(anchor_feats, "b v N c -> (b v) c N")
            anchor_feats = rearrange(anchor_feats, "b v (h w) c -> b () (v h w) c", h=feat_h, w=feat_w) #! correlation 더하기
            xyz_offsets = self.to_offset(anchor_feats) #* (b () vHW num_points*3)
            
            # if self.cfg.predict_anchor_gaussian:
            xyz_offsets = rearrange(xyz_offsets, "b () (v h w) (N xyz) -> b N (v h w) () xyz", h=feat_h, w=feat_w, N=num_points)
            anchor_feats = anchor_feats.unsqueeze(-2).expand(-1, num_points, -1, -1, -1) 
            # else:
            #     xyz_offsets = rearrange(xyz_offsets, "b n (v h w) xyz -> b (n v) (h w) () xyz", h=feat_h, w=feat_w)
            #     anchor_feats = rearrange(anchor_feats, 'b n (v h w) c -> b (n v) (h w) () c', h=feat_h, w=feat_w)
           
            if self.cfg.estimation_space == "depthmap":
                # offset_xy_n = xyz_offsets[..., :2].sigmoid() #* (b, (v N), HW, 2)
                # xy_ray = rearrange(xy_ray, "b v r srf xy -> b () (v r) srf xy")
                # depths = rearrange(depths, "b v r srf xyz -> b () (v r) srf xyz")
                # xy_ray_n = xy_ray + (offset_xy_n - 0.5) * pixel_size
                # depths_offset_n = depths + xyz_offsets[..., 2:3]
                
                #* rearrange extrinsic and intrinsic to match the shape of the num_points
                extrinsics_neighbor = repeat(context['extrinsics'][:, 0:1].clone().detach(), "b 1 i j -> b n i j", n=num_points*v)
                intrinsics_neighbor = repeat(context['intrinsics'][:, 0:1].clone().detach(), "b 1 i j -> b n i j", n=num_points*v)
                intrinsics_neighbor_org = repeat(context['intrinsics_org_scale'][:, 0:1].clone().detach(), "b 1 i j -> b n i j", n=num_points)
                
                offset_xyz = xyz_offsets[..., :3]
                offset_means_cam = anchor_gaussians.means_cam + offset_xyz #* world coordinate
                
            # positional encoding for the offsets xyz and input anchor features to predict the gaussian parameters for the offsets
            neighbor_gaussians = self.to_neighbor_gaussians(torch.cat([self.offset_encoding(offset_xyz), anchor_feats], dim=-1))
            
            anchor_offsets_depths = [anchor_gaussians.means_cam, offset_means_cam]
            
            offset_gaussians = self.gaussian_adapter.forward(
                rearrange(extrinsics_neighbor, "b n i j -> b n () () () i j"),
                rearrange(intrinsics_neighbor, "b n i j -> b n () () () i j"),
                None,
                offset_means_cam,
                neighbor_gaussians[..., 0:1].sigmoid(),
                rearrange(
                    neighbor_gaussians[..., 1:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                use_pred_cams=True if self.cfg.pred_campose else False,
                pred_cams=rearrange(pred_cam['mats'], "b v i j -> b v () () () i j") if self.cfg.pred_campose else None,
                predict_only_canonical=self.cfg.predict_only_canonical,
                xyz_only=False,
                estimation_space="cam_xyz",
                scale_factor=scale_factor,
                extra_info=extra_info,
                learn_sh_residual_from_canoncal_rgb=self.cfg.gaussian_adapter.learn_sh_residual_from_canoncal_rgb,
                gaussian_2d_scale=self.cfg.gaussian_adapter.gaussian_2d_scale,
            )
        
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                offset_gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                offset_gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
            
        #combined depths
        gaussians_all, gaussians_anchor, gaussians_offset = None, None, None
        gaussians_offsets_list = []
        if self.cfg.predict_offset:
            if "offset" in self.cfg.rendering_units:
                # depths = torch.cat([depths, depths_offset_n], dim=1)
                gaussians_offset = Gaussians(
                    rearrange(offset_gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                    rearrange(offset_gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                    rearrange(offset_gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                    rearrange(offset_gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
                    rearrange(offset_gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                    rearrange(offset_gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"),
                    rearrange(offset_gaussians.means_cam, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                )
                
                for i in range(self.cfg.num_offset_gaussian):
                    gaussians_offsets_list.append(Gaussians(
                        rearrange(offset_gaussians.means[:, i], "b r srf spp xyz -> b (r srf spp) xyz"),
                        rearrange(offset_gaussians.covariances[:, i], "b r srf spp i j -> b (r srf spp) i j"),
                        rearrange(offset_gaussians.harmonics[:, i], "b r srf spp c d_sh -> b (r srf spp) c d_sh"),
                        rearrange(offset_gaussians.opacities[:, i], "b r srf spp -> b (r srf spp)"),
                        rearrange(offset_gaussians.scales[:, i], "b r srf spp xyz -> b (r srf spp) xyz"),
                        rearrange(offset_gaussians.rotations[:, i], "b r srf spp xyzw -> b (r srf spp) xyzw"),
                        rearrange(offset_gaussians.means_cam[:, i], "b r srf spp xyz -> b (r srf spp) xyz"),
                    ))
                
                gaussians_all = self.update_gaussians_all(gaussians_all, gaussians_offset)
            
            if "anchor" in self.cfg.rendering_units and self.cfg.predict_anchor_gaussian:    
                gaussians_anchor = Gaussians(
                    rearrange(anchor_gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                    rearrange(anchor_gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                    rearrange(anchor_gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                    rearrange(anchor_gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
                    rearrange(anchor_gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                    rearrange(anchor_gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"),
                    rearrange(anchor_gaussians.means_cam, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                )
                gaussians_all = self.update_gaussians_all(gaussians_all, gaussians_anchor)
        
        else:
            
            assert anchor_gaussians is not None, "Anchor gaussians cannot be None"
            
            anchor_gaussians = Gaussians(
                    rearrange(anchor_gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                    rearrange(anchor_gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                    rearrange(anchor_gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                    rearrange(anchor_gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
                    rearrange(anchor_gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                    rearrange(anchor_gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"),
                    rearrange(anchor_gaussians.means_cam, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                )
            
            gaussians_all = self.update_gaussians_all(gaussians_all, anchor_gaussians)
            
        return {"gaussians":gaussians_all, "gaussians_anchor": gaussians_anchor, "gaussians_offset": gaussians_offset, "pred_cam": pred_cam, 'mickey_out': None, "gaussians_offsets_list": gaussians_offsets_list, "anchor_depth": depths, "offset": offset_xyz}
        
    def update_gaussians_all(self, gaussians_all, gaussians_new):
        
        
        if gaussians_all is not None:
            scales_all = torch.cat([gaussians_all.scales, gaussians_new.scales], dim=1)
            rotations_all = torch.cat([gaussians_all.rotations, gaussians_new.rotations], dim=1)
            covariances_all = torch.cat([gaussians_all.covariances, gaussians_new.covariances], dim=1)
            opacities_all = torch.cat([gaussians_all.opacities, gaussians_new.opacities], dim=1)
            means_all = torch.cat([gaussians_all.means, gaussians_new.means], dim=1)
            harmonics_all = torch.cat([gaussians_all.harmonics, gaussians_new.harmonics], dim=1)
            # if means_cam is an attribute of the class, use it
            if hasattr(gaussians_all, 'means_cam'):
                means_cam_all = torch.cat([gaussians_all.means_cam, gaussians_new.means_cam], dim=1)
            
            gaussians_all = Gaussians(
                means_all,
                covariances_all,
                harmonics_all,
                opacities_all,
                scales_all,
                rotations_all,
                means_cam_all,
            )
        else:
            gaussians_all = gaussians_new
            
        return gaussians_all
    
    def query_volume(
        self,
        positions_ndc: Float[Tensor, "*B p N srf 3"],
        feature_volume: Float[Tensor, "*B 3 Cp Hp Wp"],
        context: dict,
        aabb: Tuple,
        ):
        
        positions_ndc = rearrange(positions_ndc, "B p N srf xyz -> B (p N srf) xyz")
        batched = positions_ndc.ndim == 3
        
        if not batched:
            feature_volume = feature_volume[None, ...]
            positions_ndc = positions_ndc[None, ...]
        
        # # 고정된 delta_scale 사용
        delta_scale = 0.01
        
        # 각 차원별로 min, max 계산
        if aabb is None:
            aa = torch.min(positions_ndc, dim=1).values
            bb = torch.max(positions_ndc, dim=1).values
        else:
            aa, bb = aabb
            aa = aa.unsqueeze(0)
            bb = bb.unsqueeze(0)
        
        # 각 차원별로 범위 확장
        range_expanded = bb - aa
        aa = aa - delta_scale * range_expanded
        bb = bb + delta_scale * range_expanded
        
        aabb = torch.stack([aa, bb], dim=0) #* shape: (B, 2, 3)
        
        # 정규화
        coords = (positions_ndc - aa[:, None, :]) / (bb[:, None, :] - aa[:, None, :]) * 2.0 - 1.0
        
        coords_xyz = coords[..., [2, 1, 0]].unsqueeze(1).unsqueeze(1) # (B, 1, 1, N, 3)
        # coords_xyz = coords_xyz * 2.0 - 1.0 #* scale to -1 ~ 1
        # 3D feature map에서 좌표 샘플링
        out = F.grid_sample(feature_volume, coords_xyz, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2).squeeze(2) #* B Cp Np
        
        out = rearrange(out, "B Cp Np -> B Np Cp")
        
        if not batched:
            out = out.squeeze(0)

        return out
    
    def set_backproject(self):
        cfg = self.dataset_cfg
        backproject_depth = {}
        H = cfg.image_shape[0]
        W = cfg.image_shape[1]
        
        scale = 0
        h = H // (2 ** scale)
        w = W // (2 ** scale)
        if cfg.shift_rays_half_pixel == "zero":
            shift_rays_half_pixel = 0
        elif cfg.shift_rays_half_pixel == "forward":
            shift_rays_half_pixel = 0.5
        elif cfg.shift_rays_half_pixel == "backward":
            shift_rays_half_pixel = -0.5
        else:
            raise NotImplementedError
        backproject_depth = BackprojectDepth(
            1, 
            # backprojection can be different if padding was used
            h + 2 * self.dataset_cfg.padding_size, 
            w + 2 * self.dataset_cfg.padding_size,
            shift_rays_half_pixel=shift_rays_half_pixel
        )
        self.backproject_depth = backproject_depth

    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]
        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat
        

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
