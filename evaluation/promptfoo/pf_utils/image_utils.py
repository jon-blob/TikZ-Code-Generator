from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageOps

from config import RENDER


def load_rgb(path: str | Path, size: int | None = None) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    image = background.convert("RGB")

    if size:
        difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
        box = difference.getbbox()
        if box:
            image = image.crop(box)
        image = ImageOps.pad(image, (size, size), color="white", method=Image.Resampling.LANCZOS)

    return image


def image_array(path: str | Path) -> np.ndarray:
    return np.asarray(load_rgb(path, RENDER.image_size))


def image_tensor(path: str | Path) -> torch.Tensor:
    array = image_array(path).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
