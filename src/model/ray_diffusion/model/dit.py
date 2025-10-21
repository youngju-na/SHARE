# Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

import math
from dataclasses import dataclass

import ipdb  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from ...ray_diffusion.model.memory_efficient_attention import MEAttention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A simplified DiT block without adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_xformers_attention=False,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        attn = MEAttention if use_xformers_attention else Attention
        self.attn = attn(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The simplified final layer of DiT without adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x




@dataclass
class DiTCfg:
    in_channels: int
    ray_dim: int
    width: int
    depth: int
    hidden_size: int
    max_num_images: int
    P: int
    num_heads: int
    mlp_ratio: float
    pose_embedding: bool
    view_embedding: bool

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone, adapted to not use additional input 'c'.
    """

    def __init__(
        self,
        cfg: DiTCfg,
    ):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.ray_dim
        self.width = cfg.width
        self.depth = cfg.depth
        self.hidden_size = cfg.hidden_size
        self.max_num_images = cfg.max_num_images
        self.P = cfg.P
        self.mlp_ratio = cfg.mlp_ratio

        cfg.num_patches_x = cfg.width
        cfg.num_patches_y = cfg.width
        
        self.x_embedder = PatchEmbed(
            img_size=self.width,
            patch_size=self.P,
            in_chans=cfg.in_channels,
            embed_dim=cfg.hidden_size,
            bias=True,
            flatten=False,
        )
        
        
        self.x_pos_enc = FeaturePositionalEncoding(
            self.max_num_images, self.hidden_size, self.width**2, P=self.P
        )
        
        # self.x_pos_enc_dyn = FeaturePositionalEncodingDynamic(
        #     self.hidden_size, P=self.P
        # )
        

        try:
            import xformers
            use_xformers_attention = True
        except ImportError:
            use_xformers_attention = False

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_xformers_attention=use_xformers_attention,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_layer = FinalLayer(self.hidden_size, self.P, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)

        # print("unpatchify", c, p, h, w, x.shape)
        # assert h * w == x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nhpwqc", x)
        imgs = x.reshape(shape=(x.shape[0], h * p, h * p, c))
        return imgs

    def forward(self, x):
        """

        Args:
            x: Image/Ray features (B, N, C, H, W).
            t: Timesteps (N,).

        Returns:
            (B, N, D, H, W)
        """
        B, N, c, h, w = x.shape
        P = self.P

        x = x.reshape((B * N, c, h, w))  # (B * N, C, H, W)
        x = self.x_embedder(x)  # (B * N, C, H / P, W / P)

        x = x.permute(0, 2, 3, 1)  # (B * N, H / P, W / P, C)
        # (B, N, H / P, W / P, C)
        x = x.reshape((B, N, h // P, w // P, self.hidden_size))
        x = self.x_pos_enc(x)  # (B, N, H * W / P ** 2, C)
        # TODO: fix positional encoding to work with (N, C, H, W) format.


        for i, block in enumerate(self.blocks):
            x = x.reshape((B, N * h * w // P**2, self.hidden_size))
            x = block(x)  # (N, T, D)

        # (B, N * H * W / P ** 2, D)
        x = self.final_layer(
            x
        )  # (B, N * H * W / P ** 2,  6 * P ** 2) or (N, T, patch_size ** 2 * out_channels)

        x = x.reshape((B * N, w * w // P**2, self.out_channels * P**2))
        x = self.unpatchify(x)  # (B * N, H, W, C)
        x = x.reshape((B, N) + x.shape[1:])
        x = x.permute(0, 1, 4, 2, 3)  # (B, N, C, H, W)
        return x


class FeaturePositionalEncoding(nn.Module):
    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(base, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def __init__(self, max_num_images=8, feature_dim=1152, num_patches=256, P=1):
        super().__init__()
        self.max_num_images = max_num_images
        self.feature_dim = feature_dim
        self.P = P
        self.num_patches = num_patches // self.P**2

        self.register_buffer(
            "image_pos_table",
            self._get_sinusoid_encoding_table(
                self.max_num_images, self.feature_dim, 10000
            ),
        )

        self.register_buffer(
            "token_pos_table",
            self._get_sinusoid_encoding_table(
                self.num_patches, self.feature_dim, 70007
            ),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_images = x.shape[1]

        x = x.reshape(batch_size, num_images, self.num_patches, self.feature_dim)

        # To encode image index
        pe1 = self.image_pos_table[:, :num_images].clone().detach()
        pe1 = pe1.reshape((1, num_images, 1, self.feature_dim))
        pe1 = pe1.repeat((batch_size, 1, self.num_patches, 1))

        # To encode patch index
        pe2 = self.token_pos_table.clone().detach()
        pe2 = pe2.reshape((1, 1, self.num_patches, self.feature_dim))
        pe2 = pe2.repeat((batch_size, num_images, 1, 1))

        x_pe = x + pe1 + pe2
        x_pe = x_pe.reshape(
            (batch_size, num_images * self.num_patches, self.feature_dim)
        )

        return x_pe

class FeaturePositionalEncodingDynamic(nn.Module):
    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(base, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def __init__(self, feature_dim=1152, P=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.P = P

    def forward(self, x):
        batch_size, num_images, h, w, feature_dim = x.shape
        assert feature_dim == self.feature_dim, (
            f"Feature dimension mismatch. Expected {self.feature_dim}, got {feature_dim}."
        )

        # Calculate number of patches dynamically
        num_patches = (h * w) // (self.P**2)

        # Generate dynamic sinusoidal tables for images and tokens
        image_pos_table = self._get_sinusoid_encoding_table(
            num_images, self.feature_dim, 10000
        ).to(x.device)

        token_pos_table = self._get_sinusoid_encoding_table(
            num_patches, self.feature_dim, 70007
        ).to(x.device)

        x = x.reshape(batch_size, num_images, num_patches, self.feature_dim)

        # To encode image index
        pe1 = image_pos_table[:, :num_images].clone().detach()
        pe1 = pe1.reshape((1, num_images, 1, self.feature_dim))
        pe1 = pe1.repeat((batch_size, 1, num_patches, 1))

        # To encode patch index
        pe2 = token_pos_table.clone().detach()
        pe2 = pe2.reshape((1, 1, num_patches, self.feature_dim))
        pe2 = pe2.repeat((batch_size, num_images, 1, 1))

        x_pe = x + pe1 + pe2
        x_pe = x_pe.reshape(
            (batch_size, num_images * num_patches, self.feature_dim)
        )

        return x_pe
    

import pytorch_lightning as L
class ModLN(L.LightningModule):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]