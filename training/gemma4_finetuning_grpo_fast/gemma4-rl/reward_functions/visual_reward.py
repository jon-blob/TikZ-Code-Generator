from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pf_utils.clip_siglip_metric import image_cosine_similarity
from pf_utils.dreamsim_metric import compute_dreamsim_similarity
from pf_utils.lpips_metric import compute_lpips_distance, distance_to_similarity


@dataclass
class VisualResult:
    score: float
    siglip: float
    lpips: float
    dreamsim: float
    reason: str


def clamp01(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} returned a non-finite value: {value!r}")
    return max(0.0, min(1.0, result))


def save_tmp_image(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path)


def visual_reward_func(
    cfg,
    input_image: Image.Image,
    rendered_image: Image.Image,
) -> VisualResult:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        input_path = tmp_dir / "input.png"
        rendered_path = tmp_dir / "rendered.png"

        save_tmp_image(input_image, input_path)
        save_tmp_image(rendered_image, rendered_path)

        siglip = image_cosine_similarity(
            input_path,
            rendered_path,
            model_key="siglip",
            train_mode=True,
        )
        lpips_distance = compute_lpips_distance(
            input_path,
            rendered_path,
        )

        lpips = distance_to_similarity(lpips_distance)

        dreamsim = compute_dreamsim_similarity(
            str(input_path),
            str(rendered_path),
        )

    siglip = clamp01(siglip, "SigLIP")
    lpips = clamp01(lpips, "LPIPS")
    dreamsim = clamp01(dreamsim, "DreamSim")

    score = (
        cfg.siglip_multiplier * siglip
        + cfg.lpips_multiplier * lpips
        + cfg.dreamsim_multiplier * dreamsim
    )

    return VisualResult(
        score=float(score),
        siglip=siglip,
        lpips=lpips,
        dreamsim=dreamsim,
        reason=(
            f"visual_score={score:.3f}; "
            f"siglip={siglip:.3f}; "
            f"lpips={lpips:.3f}; "
            f"dreamsim={dreamsim:.3f}"
        ),
    )
