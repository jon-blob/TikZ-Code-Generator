from assertions.common import code_assertion


def score(reference: str, generated: str, config: dict):
    reference_length = len(reference.strip())
    output_length = len(generated.strip())
    if not reference_length:
        raise ValueError("Reference code is empty")

    error = abs(output_length - reference_length) / reference_length
    similarity = 1.0 / (1.0 + error)
    named = {"output_length": output_length, "reference_length": reference_length}
    return similarity, named, f"relative_error={error:.4f}; ratio={output_length / reference_length:.4f}"


def get_assert(output: str, context: dict):
    return code_assertion(output, context, "relative_length", score)
