import json
import OpenEXR
import Imath
import torch
import argparse
import math
import numpy as np
import transformers

import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

from tqdm.auto import tqdm

import sys
from pvd_utils import random_sample_poses, world_point_to_obj, convert_opencv_to_nerf_c2w, convert_opencv_to_nerf_pointmap
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

from diffusers.utils import (
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

import diffusers
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

from PIL import Image
from torch.utils.data import Dataset
import os

from read_write_model import read_model
import glob
import cv2
import torch.nn.functional as F
from einops import rearrange
import imageio.v2 as imageio

from moge.model import import_model_class_by_version

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def save_tensor_image(tensor: torch.Tensor, save_path: str):
    assert tensor.ndim == 3 and tensor.shape[0] == 3, "Tensor shape must be (3, H, W)"
    img = (tensor + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = img.mul(255).byte()
    img = img.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(save_path)

def split_into_2x2_local_grids(x: torch.Tensor) -> torch.Tensor:
    H, W, C = x.shape
    assert H % 2 == 0 and W % 2 == 0, "Height and Width must be divisible by 2"
    x = x.view(H // 2, 2, W // 2, 2, C)
    x = x.permute(0, 2, 1, 3, 4)
    x = x.reshape(H // 2, W // 2, 4* C)
    return x

def project_to_pixel(xyz_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y, z = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.stack([u, v], axis=-1), z

def forward_splat(
    src_img: torch.Tensor,
    pix_coords: torch.Tensor,
    tgt_H: int,
    tgt_W: int,
    importance: torch.Tensor = None
):
    B, C, H, W = src_img.shape
    device = src_img.device
    y_src, x_src = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    x_src = x_src.flatten()
    y_src = y_src.flatten()
    tgt_img = torch.zeros((B, C, tgt_H, tgt_W), device=device)
    weight_img = torch.zeros((B, 1, tgt_H, tgt_W), device=device)
    for b in range(B):
        tgt_x = pix_coords[b, :, 0]
        tgt_y = pix_coords[b, :, 1]
        tgt_x_int = tgt_x.round().long().clamp(0, tgt_W - 1)
        tgt_y_int = tgt_y.round().long().clamp(0, tgt_H - 1)
        if importance is not None:
            weights = importance[b, :, 0]
        else:
            weights = torch.ones_like(tgt_x)
        src_vals = src_img[b, :, y_src, x_src]
        for c in range(C):
            tgt_img[b, c].index_put_(
                (tgt_y_int, tgt_x_int),
                src_vals[c] * weights,
                accumulate=True
            )
        weight_img[b, 0].index_put_(
            (tgt_y_int, tgt_x_int),
            weights,
            accumulate=True
        )
    eps = 1e-6
    tgt_img = tgt_img / (weight_img + eps)
    return tgt_img

class ViewChangerInference:
    def __init__(self, moge_checkpoint_path, transformer_checkpoint_path, flux_kontext_path):
        self.center_scale = 1.0
        self.elevation = 0.0
        self.use_fp16 = False
        self.fov_x_ = None
        self.resolution_level = 9
        self.num_tokens = None
        
        moge_model_version = "v2"
        self.moge_model = import_model_class_by_version(moge_model_version).from_pretrained(moge_checkpoint_path).to("cuda").eval()
        
        transformer = FluxTransformer2DModel.from_pretrained(transformer_checkpoint_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.pipe = FluxKontextPipeline.from_pretrained(flux_kontext_path, transformer=transformer, torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
    
    def process_single(self, image_path, phi, theta, r, output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            phi_list = [phi, phi]
            theta_list = [theta, theta]
            r_list = [r, r]
            
            image = cv2.imread(image_path)[:, :, ::-1]
            H, W = image.shape[:2]
            
            moge_input_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            moge_height, moge_width = moge_input_image.shape[:2]
            moge_input_image_tensor = torch.tensor(moge_input_image / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)
            moge_output = self.moge_model.infer(moge_input_image_tensor, fov_x=self.fov_x_, resolution_level=self.resolution_level, num_tokens=self.num_tokens, use_fp16=self.use_fp16)
            points, depth, mask, intrinsics = moge_output['points'].cpu().numpy(), moge_output['depth'].cpu().numpy(), moge_output['mask'].cpu().numpy(), moge_output['intrinsics'].cpu().numpy()
            
            depth[np.isinf(depth)] = np.max(depth[np.isfinite(depth)])
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth.squeeze(-1)
            
            c2w_tensor = torch.eye(4)
            cx, cy = W / 2, H / 2
            fx = intrinsics[0, 0] * W
            fy = intrinsics[1, 1] * H
            aspect_ratio = W / H
            
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
            
            H_news_W_news = [
                (image_height // 16, image_width // 16),
                (image_height // 8, image_width // 8),
                (image_height // 4, image_width // 4)
            ]
            
            x_range = np.arange(W)
            y_range = np.arange(H)
            xx, yy = np.meshgrid(x_range, y_range)
            x_cam = (xx - cx) * depth / fx
            y_cam = (yy - cy) * depth / fy
            pointmap = np.stack((x_cam, y_cam, depth), axis=-1)
            
            depth_avg = depth[H//2, W//2]
            radius = depth_avg * self.center_scale
            pointmap = torch.from_numpy(pointmap).float()
            
            c2ws, pcd = world_point_to_obj(poses=c2w_tensor.unsqueeze(0), points=pointmap.unsqueeze(0), k=-1, r=radius, elevation=self.elevation, device=c2w_tensor.device)
            
            frame_number = 1
            new_c2ws = random_sample_poses(c2ws, phi_list, theta_list, r_list, frame_number, c2w_tensor.device)
            
            new_w2cs = torch.inverse(new_c2ws)
            xyz_tgt_cam = (new_w2cs @ torch.cat([pcd.reshape(-1, 3), torch.ones(H*W, 1, device=new_w2cs.device)], dim=1).T).T[:, :3]
            xyz_tgt_cam = xyz_tgt_cam.reshape(H, W, 3).unsqueeze(0)
            
            K = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
            
            intr = torch.eye(4)
            intr[:3, :3] = torch.tensor(K)
            
            uv, z = project_to_pixel(xyz_tgt_cam, intr)
            
            z_min = z.min()
            z_max = z.max()
            
            if (z_max - z_min) > 1e-6:
                z_norm = (z - z_min) / (z_max - z_min)
            else:
                z_norm = torch.zeros_like(z) + 0.01
            
            pix_coords = np.concatenate([z_norm[..., None], uv], axis=-1)
            pix_coords_tensor = torch.from_numpy(pix_coords).float()
            pix_coords_tensor[..., 0] += 1.0
            pix_coords_tensor = pix_coords_tensor.permute(0, 3, 1, 2)
            
            pix_coords_downs = []
            grid_level = 0
            for H_new, W_new in H_news_W_news:
                pix_coords_down = F.interpolate(pix_coords_tensor, size=(H_new, W_new), mode='bilinear', align_corners=False)
                pix_coords_down = pix_coords_down.squeeze(0).permute(1, 2, 0)
                pix_coords_down[..., -2] = pix_coords_down[..., -2] / W * H_news_W_news[0][1]
                pix_coords_down[..., -1] = pix_coords_down[..., -1] / H * H_news_W_news[0][0]
                pix_coords_down = pix_coords_down[..., [0, 2, 1]]
                if grid_level > 0:
                    for _ in range(grid_level):
                        pix_coords_down = split_into_2x2_local_grids(pix_coords_down)
                pix_coords_down = pix_coords_down.view(-1, pix_coords_down.shape[-1])
                grid_level += 1
                pix_coords_downs.append(pix_coords_down)
            
            image = torch.from_numpy(image.copy()).unsqueeze(0)
            image = image.permute(0, 3, 1, 2).float()
            
            uv = torch.from_numpy(uv)
            uv = uv.view(1, -1, 2).float()
            warped_image = forward_splat(image, uv, H, W)
            warped_image = warped_image.float() / 255.0
            warped_image = (warped_image - 0.5) / 0.5
            
            image = F.interpolate(image, size=(image_height, image_width), mode='bilinear', align_corners=False)
            masked_image = image.float() / 255.0
            masked_image = (masked_image - 0.5) / 0.5
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_tensor_image(warped_image.squeeze(), os.path.join(output_dir, f"{base_name}_warped.png"))
            save_tensor_image(masked_image.squeeze(), os.path.join(output_dir, f"{base_name}_original.png"))
            
            autocast_ctx = nullcontext()
            with autocast_ctx:
                input_image = masked_image
                _, _, height, width = input_image.shape
                image = self.pipe(
                    image=input_image, height=height, width=width,
                    prompt="Ensure that the picture does not change in any way",
                    input_img_ids=pix_coords_downs,
                    use_multi_scale_position=True
                ).images[0]
                image.save(os.path.join(output_dir, f"{base_name}_output.png"))
            
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False

