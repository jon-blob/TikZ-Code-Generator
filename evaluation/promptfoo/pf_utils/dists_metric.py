from functools import lru_cache
from pathlib import Path

import torch
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity

from pf_utils.device import get_device
from pf_utils.image_utils import image_tensor


@lru_cache(maxsize=1)
def metric():
    return DeepImageStructureAndTextureSimilarity(reduction="mean").to(get_device()).eval()


def compute_dists_distance(image_a: str | Path, image_b: str | Path) -> float:
    device = get_device()
    with torch.inference_mode():
        value = metric()(image_tensor(image_a).to(device), image_tensor(image_b).to(device))
    return float(value.item())


def distance_to_similarity(distance: float) -> float:
    return 1.0 / (1.0 + distance)
