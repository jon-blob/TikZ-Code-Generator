from assertions.common import renderable_assertion


def get_assert(output: str, context: dict):
    return renderable_assertion(output, context)
