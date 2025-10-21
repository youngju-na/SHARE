from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from einops import rearrange

@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: dict,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        _, context_v, _, _, _ = batch["context"]["image"].shape 
        
        delta = prediction['color'] - batch["target"]["image"]
        
        ssim_loss = self.ssim(
            rearrange(prediction['color'], 'b v c h w -> (b v) c h w'),
            rearrange(batch["target"]["image"], 'b v c h w -> (b v) c h w'),
        )
        total_loss = self.cfg.weight * (delta**2).mean() + (1-ssim_loss) * 0.1
        
        anchor_loss = None
        # if "color_anchor" in prediction.keys():
        #     delta_anchor = prediction['color_anchor'] - batch["context"]["image"][:, 0:1]
        #     ssim_loss_anchor = self.ssim(
        #         rearrange(prediction['color_anchor'], 'b v c h w -> (b v) c h w'),
        #         rearrange(batch["context"]["image"][:, 0:1], 'b v c h w -> (b v) c h w'),
        #     )
        #     anchor_loss = self.cfg.weight * (delta_anchor**2).mean() + (1-ssim_loss_anchor) * 0.1
        
        offset_loss = None
        # if "color_offset" in prediction.keys():
        #     delta_offset = prediction['color_offset'] - batch["context"]["image"][:, 1:context_v]
        #     ssim_loss_offset = self.ssim(
        #         rearrange(prediction['color_offset'], 'b v c h w -> (b v) c h w'),
        #         rearrange(batch["context"]["image"][:, 1:context_v], 'b v c h w -> (b v) c h w'),
        #     )
        #     offset_loss = self.cfg.weight * (delta_offset**2).mean() + (1-ssim_loss_offset) * 0.1
        
        return {"all":total_loss, "anchor":anchor_loss, "offset":offset_loss}