from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"] #! added for regularization
    rotations: Float[Tensor, "batch gaussian 4"] #! added for regularization
    means_cam: torch.Tensor | None = None
