from assertions.common import image_assertion
from pf_utils.ms_ssim_metric import compute_image_ms_ssim


def get_assert(output: str, context: dict):
    return image_assertion(
        output,
        context,
        "ms_ssim",
        lambda reference, generated, config: (
            compute_image_ms_ssim(reference, generated),
            {},
            "",
        ),
    )
