from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
# from .loss_sugar import LossSugar, LossSugarCfgWrapper
from .loss_camera import LossCameras, LossCamCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    # LossSugarCfgWrapper: LossSugar,
    LossCamCfgWrapper: LossCameras,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossCamCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
