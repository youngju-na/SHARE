from typing import Optional

import torch
from .encoder import Encoder
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume

ENCODERS = {
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
}

EncoderCfg = EncoderCostVolumeCfg


def get_encoder(cfg: EncoderCfg, dataset_cfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg, dataset_cfg)
    
    if cfg.backbone=="transmvsnet" and cfg.transmvsnet.transmvsnet_ckpt:
        load_transmvsnet_checkpoint(encoder.transmvsnet, cfg.transmvsnet.transmvsnet_ckpt)
        print(f"loaded transmvsnet pretrained weight from {cfg.transmvsnet.transmvsnet_ckpt}.")

    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer

def load_transmvsnet_checkpoint(model_mvs, ckpt_path, gmflow_n_blocks=6):
    mvs_ckpt = torch.load(ckpt_path)
    mvs_weights = mvs_ckpt['model'] if 'model' in mvs_ckpt else mvs_ckpt
    
    model_mvs.load_state_dict(mvs_weights, strict=True)
    