from assertions.common import code_assertion, setting
from config import PATHS
from pf_utils.crystalbleu_metric import compute_crystalbleu_score


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def score(reference: str, generated: str, config: dict):
    k = int(setting("crystalbleu", "k", config))
    n = int(setting("crystalbleu", "n", config))
    smoothing = as_bool(
        setting("crystalbleu", "smoothing", config)
    )
    use_cache = as_bool(
        setting("crystalbleu", "use_cache", config)
    )

    value = compute_crystalbleu_score(
        reference_code=reference,
        generated_code=generated,
        corpus_dir=PATHS.crystalbleu_corpus,
        k=k,
        n=n,
        smoothing=smoothing,
        use_cache=use_cache,
    )

    return (
        value,
        {},
        f"k={k}; n={n}; smoothing={smoothing}",
    )


def get_assert(output: str, context: dict):
    return code_assertion(
        output,
        context,
        "crystalbleu",
        score,
    )