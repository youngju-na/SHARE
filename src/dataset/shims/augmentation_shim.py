import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews


def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    extrinsics_reflected = reflect @ extrinsics @ reflect

    R_reflected = extrinsics_reflected[:, :3, :3].clone()
    T_reflected = extrinsics_reflected[:, :3, 3].clone()

    return {"extrinsics": extrinsics_reflected, "R": R_reflected, "T": T_reflected}

def reflect_views(views: AnyViews) -> AnyViews:
    return_dict = {
        **views,
        "image": views["image"].flip(-1),
        "image_normalized": views["image_normalized"].flip(-1),
        "extrinsics": reflect_extrinsics(views["extrinsics"])["extrinsics"],
        "extrinsics_gt": reflect_extrinsics(views["extrinsics_gt"])["extrinsics"],
        "R": reflect_extrinsics(views["extrinsics_gt"])["R"],
        "T": reflect_extrinsics(views["extrinsics_gt"])["T"],
    }
    
    if "depth" in views:
        return_dict["depth"] = views["depth"].flip(-1)
    if "mono" in views:
        return_dict["mono"] = views["mono"].flip(-1)
        
    return return_dict


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }
