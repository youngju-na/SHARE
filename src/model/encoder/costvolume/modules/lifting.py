import torch.nn as nn
import torch
import torch.nn.functional as F
from .base_module.cross_attention import TransformerCrossAttnLayer
from .base_module.TransformerDecoder import TransformerDecoderLayerPermute
from .base_module.flash_attention.transformer import FlashTxDecoderLayer, FlashCrossAttnLayer
from einops import rearrange
import math

class lifting(nn.Module):
    def __init__(self, in_dim, latent_res) -> None:
        super(lifting, self).__init__()

        
        embedding_stdev = (1. / math.sqrt(in_dim))
        self.latent_res = latent_res
        self.geo_latent_res = latent_res // 2 
        self.latent_emb = nn.parameter.Parameter(
                            (torch.rand(self.latent_res, self.latent_res, self.latent_res, in_dim) * embedding_stdev))
        self.geo_emb = nn.parameter.Parameter(
                            (torch.rand(self.geo_latent_res, self.geo_latent_res, self.geo_latent_res, in_dim) * embedding_stdev))

        
        self.transformer = lifting_make_transformer_layers(in_dim=in_dim)

        
        self.latent_refine = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 256, 4, stride=2, padding=1),
            #nn.Conv3d(in_dim, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
        )
        
        self.geo_latent_refine = nn.Sequential(
            nn.ConvTranspose3d(in_dim, 256, 4, stride=2, padding=1),
            #nn.Conv3d(in_dim, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
        )
    

    def forward_latent(self, x):
        '''
        x: 2D features in shape [b,t,c,h,w]
        pe2d: 2D p.e. in shape [b,t,c,h,w]
        '''
        
        b, t, c, h, w = x.shape
        device = x.device
        
        latent = rearrange(self.latent_emb, 'd h w c -> (d h w) c').unsqueeze(0).repeat(b,1,1).to(device)  # [b,N=d*h*w,c]
        
        #! save latent torch as numpy
        # torch.save(latent.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/latent_level0.pt')
        # torch.save(x.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/2d_feature.pt')
        
        x = rearrange(x, 'b t c h w -> b (t h w) c')
    
        for block in self.transformer:
            #breakpoint()
            latent = block(latent, x)

        #! save latent torch as numpy
        # torch.save(latent.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/latent_level4.pt')
        
        # latent = rearrange(latent, 'b (xyz h w) c -> (b xyz) c h w', b=b, xyz=3, h=self.latent_res, w=self.latent_res)
        latent = rearrange(latent, 'b (d h w) c -> b c d h w', d=self.latent_res, h=self.latent_res, w=self.latent_res)
        latent = self.latent_refine(latent) #* 2D CNN으로 변경        
        
        # torch.save(latent.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/latent_fine.pt')
        
        return latent #* shape: (xyz, c, h, w), triplane latent feature
    
    def forward_cv(self, cost_volume, x):
        '''
        x: 2D features in shape [b,t,c,h,w]
        pe2d: 2D p.e. in shape [b,t,c,h,w]
        '''
        b, c, d, h, w = cost_volume.shape
        
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        b, t, c, h, w = x.shape
        device = x.device
        
        #x = x[:,:1]
        cost_volume = rearrange(cost_volume, 'b c d h w -> b (d h w) c')
        x = rearrange(x, 'b t c h w -> b (t h w) c')
    
        for block in self.transformer:
            #breakpoint()
            cost_volume = block(cost_volume, x)

        cost_volume = rearrange(cost_volume, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        
        return cost_volume

    def forward_geometry(self, x):
        '''
        x: 2D features in shape [b,t,c,h,w]
        pe2d: 2D p.e. in shape [b,t,c,h,w]
        '''
        
        b, t, c, h, w = x.shape
        device = x.device
        
        latent = rearrange(self.geo_emb, 'd h w c -> (d h w) c').unsqueeze(0).repeat(b,1,1).to(device)  # [b,N=d*h*w,c]
        
        #! save latent torch as numpy
        # torch.save(latent.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/latent_level0.pt')
        # torch.save(x.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/2d_feature.pt')
        
        x = rearrange(x, 'b t c h w -> b (t h w) c')
    
        for block in self.transformer:
            #breakpoint()
            latent = block(latent, x)

        #! save latent torch as numpy
        # torch.save(latent.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/latent_level4.pt')
        
        # latent = rearrange(latent, 'b (xyz h w) c -> (b xyz) c h w', b=b, xyz=3, h=self.latent_res, w=self.latent_res)
        latent = rearrange(latent, 'b (d h w) c -> b c d h w', d=self.geo_latent_res, h=self.geo_latent_res, w=self.geo_latent_res)
        latent = self.geo_latent_refine(latent) #* 2D CNN으로 변경        
        
        # torch.save(latent.clone().detach().cpu(), '/home/youngju/ssd/ufosplat/paper_analysis/latent_fine.pt')
        
        return latent #* shape: (xyz, c, h, w), triplane latent feature


def lifting_make_init_layer(config, in_dim):
    mlp_ratio = 4.0
    norm_first = config.model.norm_first

    if not config.model.use_flash_attn:
        latent_dim = int(mlp_ratio * in_dim)
        if not config.model.lifting_TXdecoder_permute:
            layer = torch.nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim,
                                            dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
        else:
            layer = TransformerDecoderLayerPermute(d_model=in_dim, nhead=8, dim_feedforward=latent_dim,
                                            dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
    else:
        layer = FlashTxDecoderLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=norm_first)
    return layer


def lifting_make_transformer_layers(num_layers=4, norm_first=False, use_flash_attn=False, in_dim=128):
    transformer = []
    mlp_ratio = 4.0

    if not use_flash_attn:
        latent_dim = int(mlp_ratio * in_dim)
        transformer = [torch.nn.TransformerDecoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim, 
                                                        dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
                       for _ in range(num_layers)]
    else:
        transformer = [FlashTxDecoderLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=norm_first)
                       for _ in range(num_layers)]
    transformer = nn.ModuleList(transformer)
    return transformer


def lifting_make_conv3d_layers(num_layers=4, lifting_use_conv3d=False, in_dim=128):
    
    if lifting_use_conv3d:
        conv3ds = [nn.Sequential(nn.Conv3d(in_dim, in_dim, 3, padding=1),
                             nn.BatchNorm3d(in_dim),
                             nn.LeakyReLU(inplace=True),) for _ in range(num_layers)]
    else:
        conv3ds = [nn.Identity() for _ in range(num_layers)]
    conv3ds = nn.ModuleList(conv3ds)
    return conv3ds