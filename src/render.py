import torch
import numpy as np
# from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render
import torchvision
from .model.utils.general_utils import safe_state, poisson_mesh
from .model.utils.image_utils import psnr, depth2rgb, normal2rgb, depth2normal, match_depth, resample_points, mask_prune, grid_prune, depth2viewDir, img2video
from .model.utils.graphics_utils import getProjectionMatrix
from .model.utils.camera_utils import interpolate_camera
from argparse import ArgumentParser
from torchvision.utils import save_image
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
from torch.utils.cpp_extension import load
from einops import (rearrange, repeat)
import pymeshlab
import time

from jaxtyping import install_import_hook
import hydra
from pathlib import Path
from torch.utils.data import default_collate
from lightning_fabric.utilities.apply_func import apply_to_collection
from torch import Tensor

'''
This code is brought from gaussian_srufels
'''


# Configure beartype and jaxtyping.
# with install_import_hook(
#     ("src",),
#     ("beartype", "beartype"),
# ):
from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
from src.geometry.projection import homogenize_points, project
from src.global_cfg import set_cfg
from src.misc.image_io import save_image
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.decoder.cuda_splatting import render_cuda_orthographic
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper
from src.model.ply_export import export_ply
from src.visualization.color_map import apply_color_map_to_image
from src.visualization.drawing.cameras import unproject_frustum_corners
from src.visualization.drawing.lines import draw_lines
from src.visualization.drawing.points import draw_points


def render_set(model_path, example, gaussians, decoder, poisson_depth=10, write_image=True):
    
    scene = example['scene'][0]
    render_path = os.path.join(model_path, scene, "ours_{}".format(0), "renders")
    gts_path = os.path.join(model_path, scene, "ours_{}".format(0), "gt")
    info_path = os.path.join(model_path, scene, "ours_{}".format(0), "info")
    
    img_names = example['context']['index']

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(info_path, exist_ok=True)

    #TODO: encoder가 여기 들어가야 함. 
    bound = None
    # occ_grid, grid_shift, grid_scale, grid_dim = to_occ_grid(gaussians, 0.0, 512, bound)

    b, v, c, h, w = example['context']['image'].shape
    
    #TODO: resample?
    resampled = []
    psnr_all = []

    background = torch.zeros((3), dtype=torch.float32, device="cuda")
    
    #TODO: target-view rendering하는 부분? 
    color, normal, depth, opac = decoder.forward(
            gaussians,
            example["context"]["extrinsics_gt"],
            example["context"]["intrinsics"],
            example["context"]["near"],
            example["context"]["far"],
            (h, w),
            depth_mode='depth',
            rasterizer='surfel',
        )
    
    mask_gt = example['context']['mask']
    gt_image = example['context']['image']
    
    mask_vis = (opac.detach() > 1e-5)
    depth_range = [0, 20]
    mask_clip = (depth > depth_range[0]) * (depth < depth_range[1])

    normal = torch.nn.functional.normalize(normal, dim=1) * mask_vis
    
    gt_image = rearrange(gt_image, "b v c h w -> (b v) c h w")
    color = rearrange(color, "b v c h w -> (b v) c h w")
    depth = rearrange(depth, "b v d h w -> (b v) d h w")
    normal = rearrange(normal, "b v d h w -> (b v) d h w")
    opac = rearrange(opac, "b v d h w -> (b v) d h w")
    mask_vis = rearrange(mask_vis, "b v d h w -> (b v) d h w")
    mask_gt = rearrange(mask_gt, "b v d h w -> (b v) d h w")
    mask_clip = rearrange(mask_clip, "b v d h w -> (b v) d h w")
    intr = rearrange(example['context']['intrinsics'], "b v i j -> (b v) i j") #! org scale? or not?
    extr = rearrange(example['context']['extrinsics'], "b v i j -> (b v) i j")
    img_size = (h, w)
    d2ns = []
    
    
    
    for i in range(len(depth)):
        pts = resample_points(intr[i], extr[i], depth[i], normal[i], color[i], mask_vis[i] * mask_gt[i] * mask_clip[i], image_size=(h, w))
        # grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=1)
        # clean_mask = grid_mask #* mask_mask
        # pts = pts[clean_mask]
        resampled.append(pts.cpu())
        
        if write_image:
            d2n = depth2normal(depth[i], mask_vis[i], intr[i], image_size=(h, w))
            normal_wrt = normal2rgb(normal[i], mask_vis[i])
            depth_wrt = depth2rgb(depth[i], mask_vis[i])
            d2n_wrt = normal2rgb(d2n, mask_vis[i])
            normal_wrt += background[:, None, None] * (~mask_vis[i]).expand_as(color[i]) * mask_gt[i]
            depth_wrt += background [:, None, None]* (~mask_vis[i]).expand_as(color[i]) * mask_gt[i]
            d2n_wrt += background[:, None, None] * (~mask_vis[i]).expand_as(color[i]) * mask_gt[i]
            outofmask = mask_vis[i] * (1 - mask_gt[i])
            mask_vis_wrt = outofmask * (opac[i] - 1) + mask_vis[i]
            img_wrt = torch.cat([gt_image[i], color[i], normal_wrt, d2n_wrt, depth_wrt], 2)
            wrt_mask = torch.cat([opac[i] * mask_gt[i], mask_vis_wrt, mask_vis_wrt, mask_vis_wrt], 2)
            img_wrt = torch.cat([img_wrt, wrt_mask], 0)
            save_image(img_wrt.cpu(), os.path.join(info_path, '{}'.format(str(img_names[0][i])) + f".png"))
            save_image(color[i].cpu(), os.path.join(render_path, '{}'.format(str(img_names[0][i])) + ".png"))
            save_image((torch.cat([gt_image[i], mask_gt[i]], 0)).cpu(), os.path.join(gts_path, '{}'.format(str(img_names[0][i])) + ".png"))

    
    #TODO: poisson mesh 부분
    resampled = torch.cat(resampled, 0)
    mesh_path = f'{model_path}/poisson_mesh_{poisson_depth}'
    
    poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, 1 * 1e-4)
    
    # cd = eval_dtu(int(model_path.split('/')[-1].split('_')[0][4:]), mesh_path.split('.')[0] + '_pruned.ply')
    # cd = eval_bmvs(model_path.split('/')[-1].split('_')[0], mesh_path.split('.')[0] + '_pruned.ply')
    cd = 0
    with open(f"{model_path}/eval_result.txt", 'a') as f:
        f.write(f'CD: {cd}\n')

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def extract_mesh(cfg_dict):
    with torch.no_grad():
        cfg = load_typed_root_config(cfg_dict)
        set_cfg(cfg_dict)
        torch.manual_seed(cfg_dict.seed)
        device = torch.device("cuda:0")
        
        # Prepare the checkpoint for loading.
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

        encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
        decoder = get_decoder(cfg.model.decoder, cfg.dataset)
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path,
            optimizer_cfg=cfg.optimizer,
            test_cfg=cfg.test,
            train_cfg=cfg.train,
            encoder=encoder,
            encoder_visualizer=encoder_visualizer,
            decoder=decoder,
            losses=[],
            step_tracker=None,
            max_step=cfg.trainer.max_steps,
            strict=False
        )
        model_wrapper.eval()
        
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        for scan in [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]:
            
            scene = f"scan{scan}"
            print("Rendering scene: ", scene)
            
            # context_indices = example['context']['index']
            view_sampler_cfg = ViewSamplerArbitraryCfg(
                "arbitrary",
                2, #! num context
                3, #! num target
                context_views=cfg.dataset.mesh_ref_views, #! 33
                target_views=[0, 0],  # use [40, 80] for teaser
                )
            cfg.dataset.view_sampler = view_sampler_cfg
            
            cfg.dataset.overfit_to_scene = scene

            # Get the scene.
            dataset = get_dataset(cfg.dataset, "mesh", None)
            example = default_collate([next(iter(dataset))])
            example = apply_to_collection(example, Tensor, lambda x: x.to(device))
            
            # Generate the Gaussians.
            visualization_dump = {}
            encoder_out = encoder.forward(
                    example["context"], False, visualization_dump=visualization_dump
                )
            
            gaussians = encoder_out['gaussians']
            
            scene = example['scene'][0]
            model_path = os.path.join(cfg.test.output_path, "surface", scene)
            os.makedirs(model_path, exist_ok=True)
            render_set(model_path, example, gaussians, decoder, 10)
    
    
    
    
if __name__ == "__main__":
    # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--img", action="store_true")
    # parser.add_argument("--depth", default=10, type=int)
    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)
    
    # Initialize system state (RNG)
    # safe_state(args.quiet)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.img, args.depth)
    extract_mesh()
    
    