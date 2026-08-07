from pathlib import Path

from skimage.metrics import structural_similarity

from pf_utils.image_utils import image_array


def compute_image_ssim(image_a: str | Path, image_b: str | Path) -> float:
    return float(
        structural_similarity(
            image_array(image_a),
            image_array(image_b),
            channel_axis=2,
            data_range=255,
        )
    )
