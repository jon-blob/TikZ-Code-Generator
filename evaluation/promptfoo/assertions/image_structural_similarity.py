from assertions.common import image_assertion
from pf_utils.ssim_metric import compute_image_ssim


def get_assert(output: str, context: dict):
    return image_assertion(
        output,
        context,
        "ssim",
        lambda reference, generated, config: (
            compute_image_ssim(reference, generated),
            {},
            "",
        ),
    )
