from functools import lru_cache
from pathlib import Path

import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from pf_utils.device import get_device
from pf_utils.image_utils import image_tensor


@lru_cache(maxsize=1)
def metric():
    return MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(get_device()).eval()


def compute_image_ms_ssim(image_a: str | Path, image_b: str | Path) -> float:
    device = get_device()
    with torch.inference_mode():
        value = metric()(image_tensor(image_a).to(device), image_tensor(image_b).to(device))
    return float(value.item())
