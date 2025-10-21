import io
import os
import os.path as osp

import ipdb  # noqa: F401
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from einops import rearrange

from src.model.ray_diffusion.utils.rays import (
    Rays,
    cameras_to_rays,
    rays_to_cameras,
    rays_to_cameras_homography,
)

cmap = plt.get_cmap("hsv")


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def plot_to_image(figure, dpi=100):
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="raw", dpi=dpi)
    plt.close(figure)
    buffer.seek(0)
    image = np.reshape(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8),
        newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1),
    )
    return image[..., :3]


def view_color_coded_images_from_tensor(images):
    num_frames = images.shape[0]
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_frames:
            axs[i].imshow(unnormalize_image(images[i]))
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()


def view_color_coded_images_from_path(image_dir):
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()

    def hidden(x):
        return not x.startswith(".")

    image_paths = sorted(os.listdir(image_dir))
    image_paths = list(filter(hidden, image_paths))
    image_paths = image_paths[0 : (min(len(image_paths), 8))]
    num_frames = len(image_paths)

    for i in range(num_rows * num_cols):
        if i < num_frames:
            img = np.asarray(Image.open(osp.join(image_dir, image_paths[i])))
            print(img.shape)
            axs[i].imshow(img)
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
    return fig, num_frames

def create_training_visualizations(
    pred_rays,
    images,
    device,
    cameras_gt,
    num_images,
    moments_rescale=1.0,
    calculate_intrinsics=False,
    intrinsics=None,
):
    
    v, c, num_patches_x, num_patches_y = pred_rays.shape
    
    # Prepare to store visualizations and predicted camera parameters
   
    # pred_cameras_batched = []

    # for index in range(b):  # Loop over batch size
    
    vis_images = []
    per_sample_images = []
    pred_cameras = []

    # Extract ground truth rays for all images in one go if possible
    rays_gt_batch = cameras_to_rays(
        cameras_gt,
        None,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )
    
    for ii in range(num_images):
        # Normalize and visualize different components
        image_vis = images[ii].cpu().permute(1, 2, 0).numpy()
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=100)
        
        components_to_visualize = [
            (rays_gt_batch.get_moments()[ii], "GT Moments", (0, 0)),
            (rearrange(pred_rays[ii, 3:], 'c h w -> (h w) c'), "Pred Moments", (1, 0)),
            (rays_gt_batch.get_directions()[ii], "GT Directions", (0, 1)),
            (rearrange(pred_rays[ii, :3],'c h w -> (h w) c'), "Pred Directions", (1, 1))
        ]
        
        cmap = plt.get_cmap('viridis')
        for data, title, pos in components_to_visualize:
            
            # data = rearrange(data, "c h w -> h w c", h=num_patches_y, w=num_patches_x)
            # vis = (torch.nn.functional.normalize(data, dim=-1) + 1) / 2.0 #! normalization
            normalized_data = (data - data.min()) / (data.max() - data.min() + 1e-6)
            
            # 감마 조정을 통한 대비 강화 (감마 값은 실험을 통해 최적화 가능)
            gamma = 0.5
            adjusted_data = torch.pow(normalized_data, gamma)
            
            # 데이터를 이미지 형태로 재구성 및 시각화
            vis_image = adjusted_data.reshape(num_patches_y, num_patches_x, 3).detach().cpu().numpy()
            r, c = pos
            axs[r, c].imshow(vis_image, cmap=cmap)
            axs[r, c].set_title(title)
            axs[r, c].axis('off')  # 축 정보 제거
        
        # Display the input image
        axs[0, 2].imshow(image_vis)
        axs[0, 2].set_title("Input Image")
        
        # Remove axes and set borders
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        img = plot_to_image(fig)  # Convert matplotlib plot to image format
        plt.close(fig)
        per_sample_images.append(img)
        
        # Calculate predicted cameras
        rays_camera = pred_rays[ii]
        rays = Rays.from_spatial(rays_camera, moments_rescale=moments_rescale)

        if calculate_intrinsics:
            pred_camera = rays_to_cameras_homography(
                rays=rays,
                device=device,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )
        else:
            focal_length = intrinsics[0, 0, [0, 1], [0, 1]]
            principal_point = intrinsics[0, 0, [0, 1], [2, 2]]

            pred_camera = rays_to_cameras(
                rays=rays,
                crop_parameters=None,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
                focal_lengths=focal_length,
                principal_points=principal_point
            )
        pred_cameras.append(pred_camera)
        
        vis_images.append(np.vstack(per_sample_images))

    return vis_images, pred_cameras


def create_plotly_cameras_visualization(cameras_gt, cameras_pred, num=0):
    num_frames = cameras_gt.R.shape[0]
    name = f"Vis GT vs Pred Cameras"
    camera_scale = 0.05

    # Cameras_pred is already a 2D list of unbatched cameras
    # But cameras_gt is a 1D list of batched cameras
    scenes = {f"Vis GT vs Pred Cameras": {}}
    for i in range(num_frames):
        scenes[name][f"Pred Camera {i}"] = PerspectiveCameras(
            R=cameras_pred[i].R, T=cameras_pred[i].T
        )
    for i in range(num_frames):
        scenes[name][f"GT Camera {i}"] = PerspectiveCameras(
            R=cameras_gt.R[i].unsqueeze(0), T=cameras_gt.T[i].unsqueeze(0)
        )

    fig = plot_scene(
        scenes,
        camera_scale=camera_scale,
    )
    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=800, width=800)

    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
        fig.data[i].line.width = 4
        fig.data[i + num_frames].line.dash = "dash"
        fig.data[i + num_frames].line.color = matplotlib.colors.to_hex(
            cmap(i / (num_frames))
        )
        fig.data[i + num_frames].line.width = 4

    return fig
