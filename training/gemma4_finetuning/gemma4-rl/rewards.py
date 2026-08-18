from __future__ import annotations

import logging
import re
from typing import Any

from reward_functions.code_reward import code_reward_func
from reward_functions.diagnostic_reward import diagnostic_reward_func
from reward_functions.render_reward import is_renderable
from reward_functions.visual_reward import visual_reward_func

logger = logging.getLogger(__name__)


def extract_text(value: Any) -> str:
    """Extract text from TRL completion strings or conversational structures."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return extract_text(value["text"])
        for key in ("content", "text", "completion"):
            if key in value:
                return extract_text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return "".join(extract_text(item) for item in value)
    return str(value)


def clean_code(completion: Any) -> str:
    code = extract_text(completion).strip()
    code = re.sub(r"^```(?:latex|tex)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```$", "", code)

    start = r"\documentclass"
    end = r"\end{document}"
    if start in code:
        code = code[code.index(start) :]
    if end in code:
        code = code[: code.index(end) + len(end)]
    return code.strip()


def pick(values: Any, idx: int) -> Any:
    """Select metadata aligned one-to-one with a generated completion."""
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        if not values:
            return None
        if idx >= len(values):
            raise ValueError(
                "Reward metadata is shorter than the completion batch; "
                "image/answer alignment would be ambiguous."
            )
        return values[idx]
    return values


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


class TikZReward:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.__name__ = "tikz_reward"
        self.last_metrics: dict[str, float] = {}
        self.last_examples: list[dict[str, Any]] = []

    @staticmethod
    def _mean(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return float(sum(present) / len(present)) if present else None

    def _summarize_records(self, records: list[dict[str, Any]]) -> dict[str, float]:
        metrics = {
            "reward/render_ok_rate": self._mean([r["render_ok"] for r in records]),
            "reward/avg_total_score": self._mean([r["total_score"] for r in records]),
            "reward/avg_latex_errors": self._mean([r["latex_errors"] for r in records]),
            "reward/avg_latex_warnings": self._mean([r["latex_warnings"] for r in records]),
            "reward/avg_latex_badboxes": self._mean([r["latex_badboxes"] for r in records]),
            "reward/avg_diagnostic_score": self._mean([r["diagnostic_score"] for r in records]),
            "reward/visual_eval_rate": self._mean([r["visual_evaluated"] for r in records]),
            "reward/avg_visual_score": self._mean([r["visual_score"] for r in records]),
            "reward/avg_siglip": self._mean([r["siglip"] for r in records]),
            "reward/avg_lpips": self._mean([r["lpips"] for r in records]),
            "reward/avg_dreamsim": self._mean([r["dreamsim"] for r in records]),
            "reward/code_eval_rate": self._mean([r["code_evaluated"] for r in records]),
            "reward/avg_code_score": self._mean([r["code_score"] for r in records]),
            "reward/avg_crystalbleu": self._mean([r["crystalbleu"] for r in records]),
            "reward/avg_ted": self._mean([r["ted"] for r in records]),
        }
        return {key: value for key, value in metrics.items() if value is not None}

    @staticmethod
    def _empty_record() -> dict[str, Any]:
        return {
            "render_ok": 0.0,
            "total_score": None,
            "latex_errors": None,
            "latex_warnings": None,
            "latex_badboxes": None,
            "diagnostic_score": None,
            "visual_evaluated": 0.0,
            "visual_score": None,
            "siglip": None,
            "lpips": None,
            "dreamsim": None,
            "code_evaluated": 0.0,
            "code_score": None,
            "crystalbleu": None,
            "ted": None,
        }

    def __call__(
        self,
        completions,
        answer=None,
        image=None,
        images=None,
        reference_image=None,
        reference_images=None,
        **kwargs,
    ) -> list[float]:
        scores: list[float] = []
        records: list[dict[str, Any]] = []
        examples: list[dict[str, Any]] = []

        input_images = first_present(
            image, images, kwargs.get("image"), kwargs.get("images")
        )
        target_images = first_present(
            reference_image,
            reference_images,
            kwargs.get("reference_image"),
            kwargs.get("reference_images"),
        )
        reference_codes = first_present(answer, kwargs.get("answer"))

        for idx, completion in enumerate(completions):
            generated_code = clean_code(completion)
            reference_code = pick(reference_codes, idx)
            input_image = pick(input_images, idx)
            reference_image_for_reward = pick(target_images, idx)
            record = self._empty_record()

            render = is_renderable(generated_code)
            record["latex_errors"] = float(render.errors)
            record["latex_warnings"] = float(render.warnings)
            record["latex_badboxes"] = float(render.badboxes)

            if not render.ok:
                score = float(self.cfg.not_renderable_score)
                record["total_score"] = score
            else:
                record["render_ok"] = 1.0
                diagnostic = diagnostic_reward_func(
                    cfg=self.cfg,
                    errors=render.errors,
                    warnings=render.warnings,
                    badboxes=render.badboxes,
                )
                record["diagnostic_score"] = float(diagnostic.score)

                visual_score = 0.0
                if reference_image_for_reward is not None and render.image is not None:
                    try:
                        visual = visual_reward_func(
                            cfg=self.cfg,
                            input_image=reference_image_for_reward,
                            rendered_image=render.image,
                        )
                        visual_score = float(visual.score)
                        record.update(
                            visual_evaluated=1.0,
                            visual_score=visual_score,
                            siglip=float(visual.siglip),
                            lpips=float(visual.lpips),
                            dreamsim=float(visual.dreamsim),
                        )
                    except Exception:
                        logger.exception("Visual reward failed for completion %d", idx)
                        record["visual_score"] = 0.0

                code_score = 0.0
                if reference_code:
                    try:
                        code = code_reward_func(
                            generated_code=generated_code,
                            reference_code=str(reference_code),
                            cfg=self.cfg,
                        )
                        code_score = float(code.score)
                        record.update(
                            code_evaluated=1.0,
                            code_score=code_score,
                            crystalbleu=float(code.crystalbleu),
                            ted=float(code.ted),
                        )
                    except Exception:
                        logger.exception("Code reward failed for completion %d", idx)
                        record["code_score"] = 0.0

                score = float(
                    self.cfg.renderable_score
                    + diagnostic.score
                    + self.cfg.code_reward_multiplier * code_score
                    + self.cfg.visual_reward_multiplier * visual_score
                )
                record["total_score"] = score

            if len(examples) < self.cfg.log_examples_max:
                examples.append(
                    {
                        "input_image": input_image,
                        "reference_image": reference_image_for_reward,
                        "rendered_image": render.image,
                        "generated_code": generated_code,
                        "reference_code": reference_code,
                        "score": score,
                        "render_ok": bool(render.ok),
                        "render_reason": render.reason,
                    }
                )

            scores.append(score)
            records.append(record)

        self.last_metrics = self._summarize_records(records)
        self.last_examples = examples
        return scores
