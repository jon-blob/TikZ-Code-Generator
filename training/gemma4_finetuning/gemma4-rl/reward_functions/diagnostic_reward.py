from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DiagnosticResult:
    score: float
    penalty: float
    reason: str


def _nonnegative_count(value, name: str) -> float:
    result = float(value or 0)
    if not math.isfinite(result):
        raise ValueError(f"{name} returned a non-finite value: {value!r}")
    return max(0.0, result)


def diagnostic_reward_func(cfg, errors: int, warnings: int, badboxes: int) -> DiagnosticResult:
    errors = _nonnegative_count(errors, "latex_errors")
    warnings = _nonnegative_count(warnings, "latex_warnings")
    badboxes = _nonnegative_count(badboxes, "latex_badboxes")

    penalty = (
        cfg.error_multiplier * errors
        + cfg.warning_multiplier * warnings
        + cfg.badboxes_multiplier * badboxes
    )

    score = cfg.diagnostic_base_max_score - penalty
    score = max(cfg.diagnostic_base_min_score, min(cfg.diagnostic_base_max_score, score))

    return DiagnosticResult(
        score=float(score),
        penalty=float(penalty),
        reason=(
            f"errors={errors:g}; warnings={warnings:g}; badboxes={badboxes:g}; "
            f"penalty={penalty:.3f}; score={score:.3f}"
        ),
    )
