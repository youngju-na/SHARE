# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from ..heads.postprocess import postprocess


class UNetDecoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetDecoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.upsample = nn.ConvTranspose2d(input_channels // 2, input_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.upsample(x)
        return x
    

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf

        self.proj = nn.Linear(net.dec_embed_dim, (3 + has_conf)*self.patch_size**2) #* 
        # self.gaussian = nn.Linear(net.dec_embed_dim, (3 + 3 + 1 + 48)*self.patch_size**2)  # 3 rotation, 3 scale, 1 opacity, 48 sh coeffs (total 55)
            
        
        # UNet-style decoder
        self.decoder1 = UNetDecoder(768, 384)
        self.decoder2 = UNetDecoder(384, 192)
        self.decoder3 = UNetDecoder(192, 96)
        self.decoder4 = UNetDecoder(96, 48)
        self.final_conv = nn.Conv2d(48, 56, kernel_size=3, stride=1, padding=1) #* density 1, rotation 4, scale 3 sh 48, total 56
        
        
    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        #! estimate gaussian parameters (means, rotation, scale, opacity, sh coeffs)
        gaussian = tokens.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        gaussian = self.decoder1(gaussian)  # B, 384, 28, 28
        gaussian = self.decoder2(gaussian)  # B, 192, 56, 56
        gaussian = self.decoder3(gaussian)  # B, 192, 112, 112
        gaussian = self.decoder4(gaussian)  # B, 96, 224, 224
        gaussian = self.final_conv(gaussian)  # B, 56, 224, 224
        
        
        # permute + norm depth
        return postprocess(feat, self.depth_mode, self.conf_mode), gaussian
