from assertions.common import image_assertion
from pf_utils.dreamsim_metric import compute_dreamsim_similarity


def get_assert(output: str, context: dict):
    return image_assertion(
        output,
        context,
        "dreamsim",
        lambda reference, generated, config: (
            compute_dreamsim_similarity(reference, generated),
            {},
            "",
        ),
    )
