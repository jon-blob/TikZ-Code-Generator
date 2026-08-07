from assertions.common import image_assertion
from pf_utils.dists_metric import compute_dists_distance, distance_to_similarity


def score(reference, generated, config):
    distance = compute_dists_distance(reference, generated)
    return distance_to_similarity(distance), {"dists_distance": distance}, ""


def get_assert(output: str, context: dict):
    return image_assertion(output, context, "dists", score)
