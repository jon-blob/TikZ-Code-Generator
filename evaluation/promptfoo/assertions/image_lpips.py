from assertions.common import image_assertion, setting
from pf_utils.lpips_metric import compute_lpips_distance, distance_to_similarity


def score(reference, generated, config):
    net_type = str(setting("lpips", "net_type", config))
    distance = compute_lpips_distance(reference, generated, net_type)
    return distance_to_similarity(distance), {"lpips_distance": distance}, f"net_type={net_type}"


def get_assert(output: str, context: dict):
    return image_assertion(output, context, "lpips", score)
