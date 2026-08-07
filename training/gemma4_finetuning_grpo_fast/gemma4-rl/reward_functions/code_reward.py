from __future__ import annotations

import math
from dataclasses import dataclass

from pf_utils.crystalbleu_metric import compute_crystalbleu_score
from pf_utils.ted_metric import compute_ted, distance_to_similarity


@dataclass
class CodeRewardResult:
    score: float
    ted: float
    crystalbleu: float
    reason: str


def _finite_float(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} returned a non-finite value: {value!r}")
    return result


def code_reward_func(
    generated_code: str,
    reference_code: str,
    cfg,
) -> CodeRewardResult:
    crystalbleu = _finite_float(
        compute_crystalbleu_score(
            reference_code=reference_code,
            generated_code=generated_code,
            corpus_dir=cfg.crystalbleu_corpus_dir,
            k=cfg.crystalbleu_k,
            n=cfg.crystalbleu_n,
            use_cache=cfg.crystalbleu_use_cache,
        ),
        "CrystalBLEU",
    )
    crystalbleu = max(0.0, min(1.0, crystalbleu))

    ted = _finite_float(
        compute_ted(
            generated_code=generated_code,
            reference_code=reference_code,
        ),
        "TED",
    )
    ted = max(0.0, ted)

    ted_sim = _finite_float(
        distance_to_similarity(ted),
        "TED similarity",
    )
    ted_sim = max(0.0, min(1.0, ted_sim))

    weight_sum = cfg.crystalbleu_weight + cfg.ted_weight
    if weight_sum <= 0:
        raise ValueError("Code reward weights must have a positive sum.")

    score = (
        cfg.crystalbleu_weight * crystalbleu
        + cfg.ted_weight * ted_sim
    ) / weight_sum

    return CodeRewardResult(
        score=float(score),
        ted=ted,
        crystalbleu=crystalbleu,
        reason=(
            f"code_score={score:.3f}; "
            f"crystalbleu={crystalbleu:.3f}; "
            f"ted_distance={ted:.3f}; "
            f"ted_sim={ted_sim:.3f}"
        ),
    )
