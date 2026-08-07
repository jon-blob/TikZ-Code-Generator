from pygments.lexers.markup import TexLexer
from pygments.token import Comment


def tokenize(text: str) -> list[str]:
    lexer = TexLexer()
    return [value.strip() for token_type, value in lexer.get_tokens(text) if value.strip() and token_type not in Comment]


def levenshtein(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def compute_ted(generated_code: str, reference_code: str) -> float:
    generated = tokenize(generated_code)
    reference = tokenize(reference_code)
    if not reference:
        raise ValueError("Reference code is empty")
    return levenshtein(generated, reference) / len(reference)


def distance_to_similarity(distance: float) -> float:
    return 1.0 / (1.0 + distance)
