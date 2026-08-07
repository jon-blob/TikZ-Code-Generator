from assertions.common import code_assertion
from pf_utils.ted_metric import compute_ted, distance_to_similarity


def score(reference: str, generated: str, config: dict):
    distance = compute_ted(generated, reference)
    return distance_to_similarity(distance), {"ted_distance": distance}, ""


def get_assert(output: str, context: dict):
    return code_assertion(output, context, "ted", score)
