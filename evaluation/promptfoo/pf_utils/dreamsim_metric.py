from functools import lru_cache
from pathlib import Path

from dreamsim import dreamsim
import torch

from config import MODELS, PATHS, RENDER
from pf_utils.device import get_device
from pf_utils.image_utils import load_rgb


@lru_cache(maxsize=1)
def model_and_processor():
    device = get_device()
    model, processor = dreamsim(
        dreamsim_type=MODELS.dreamsim,
        pretrained=True,
        normalize_embeds=True,
        device=str(device),
        cache_dir=str(PATHS.model_cache / "dreamsim"),
    )
    return model.eval(), processor, device


def compute_dreamsim_similarity(image_a: str | Path, image_b: str | Path) -> float:
    model, processor, device = model_and_processor()
    tensor_a = processor(load_rgb(image_a, RENDER.image_size)).to(device)
    tensor_b = processor(load_rgb(image_b, RENDER.image_size)).to(device)
    with torch.inference_mode():
        distance = float(model(tensor_a, tensor_b).item())
    return 1.0 - distance
