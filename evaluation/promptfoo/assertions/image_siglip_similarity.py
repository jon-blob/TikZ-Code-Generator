from assertions.common import image_assertion
from pf_utils.clip_siglip_metric import image_cosine_similarity


def get_assert(output: str, context: dict):
    return image_assertion(
        output,
        context,
        "siglip",
        lambda reference, generated, config: (
            image_cosine_similarity(reference, generated, "siglip"),
            {},
            "",
        ),
    )
