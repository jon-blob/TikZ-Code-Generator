from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import math
import shutil
import traceback

from config import BENCHMARK, DEBUG, METRICS, PATHS, RENDER, ensure_directories
from pf_utils.clean_text_in_tex import replace_text_with_placeholders
from pf_utils.tikz_rendering import TikzRenderError, render_tex_to_png

ScoreResult = tuple[float, dict[str, float], str]


def context_data(context: dict | None) -> tuple[dict, dict]:
    context = context or {}
    return context.get("vars") or {}, context.get("config") or {}


def setting(metric: str, name: str, config: dict) -> Any:
    return config.get(name, METRICS[metric].get(name))


def path_value(value: Any) -> Path:
    text = str(value).removeprefix("file://")
    return Path(text).expanduser()


def required_path(vars_: dict, key: str) -> Path:
    value = vars_.get(key)
    if not value:
        raise ValueError(f"Missing vars.{key}")
    path = path_value(value)
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path


def metric_tex(tex: str) -> str:
    if BENCHMARK.text_replace_metrik:
        return replace_text_with_placeholders(tex)
    return tex


def failure(metric: str, error: Exception | str, named_scores: dict | None = None) -> dict:
    reason = f"{metric} failed: {error}"
    if DEBUG.tracebacks and isinstance(error, Exception):
        reason += "\n" + traceback.format_exc()

    result = {"pass": False, "score": 0.0, "reason": reason}
    if named_scores:
        result["namedScores"] = named_scores
    return result


def result(metric: str, score: float, threshold: float, details: str = "", named_scores: dict | None = None) -> dict:
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite score: {score}")
    score = min(1.0, max(0.0, score))

    reason = f"{metric}={score:.4f}; threshold={threshold:.4f}"
    if details:
        reason += f"; {details}"

    output = {"pass": score >= threshold, "score": score, "reason": reason}
    if named_scores:
        output["namedScores"] = {key: float(value) for key, value in named_scores.items()}
    return output


def render_cache_key(output: str) -> str:
    signature = "|".join((output, repr(RENDER)))
    return sha256(signature.encode("utf-8")).hexdigest()


def rendered_image(output: str) -> tuple[Path, dict]:
    ensure_directories()
    key = render_cache_key(output)
    image_path = PATHS.render_cache / f"{key}.png"
    metrics_path = PATHS.render_cache / f"{key}.json"

    if image_path.is_file() and image_path.stat().st_size > 0 and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics: dict[str, Any] = {}
        temporary = image_path.with_suffix(".tmp.png")
        render_tex_to_png(output, temporary, metrics=metrics)
        temporary.replace(image_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return image_path, metrics


def save_images(
    stem: str,
    reference: Path,
    generated: Path,
    reference_tex: str | None,
    generated_tex: str,
) -> None:
    if not DEBUG.save_images:
        return

    suffix = "_text_replaced" if BENCHMARK.text_replace_metrik else ""
    target = PATHS.generated_images
    reference_suffix = ".png" if reference_tex is not None else reference.suffix
    shutil.copyfile(reference, target / f"{stem}{suffix}_reference{reference_suffix}")
    shutil.copyfile(generated, target / f"{stem}{suffix}_generated.png")
    (target / f"{stem}{suffix}_output.tex").write_text(generated_tex, encoding="utf-8")
    if reference_tex is not None:
        (target / f"{stem}{suffix}_reference.tex").write_text(reference_tex, encoding="utf-8")


def image_assertion(
    output: str,
    context: dict,
    metric: str,
    scorer: Callable[[Path, Path, dict], ScoreResult],
) -> dict:
    vars_, config = context_data(context)
    threshold = float(setting(metric, "threshold", config))

    try:
        reference_image = required_path(vars_, "reference_image")
        generated_tex = metric_tex(output)
        generated, _ = rendered_image(generated_tex)
        reference_tex = None
        reference = reference_image

        if BENCHMARK.text_replace_metrik:
            reference_code = required_path(vars_, "reference_code").read_text(encoding="utf-8")
            reference_tex = metric_tex(reference_code)
            reference, _ = rendered_image(reference_tex)

        save_images(reference_image.stem, reference, generated, reference_tex, generated_tex)
        score, named_scores, details = scorer(reference, generated, config)
        return result(metric, score, threshold, details, named_scores)
    except Exception as error:
        return failure(metric, error)


def code_assertion(
    output: str,
    context: dict,
    metric: str,
    scorer: Callable[[str, str, dict], ScoreResult],
) -> dict:
    vars_, config = context_data(context)
    threshold = float(setting(metric, "threshold", config))

    try:
        reference = required_path(vars_, "reference_code").read_text(encoding="utf-8")
        score, named_scores, details = scorer(metric_tex(reference), metric_tex(output), config)
        return result(metric, score, threshold, details, named_scores)
    except Exception as error:
        return failure(metric, error)


def renderable_assertion(output: str, context: dict) -> dict:
    threshold = float(setting("renderable", "threshold", context_data(context)[1]))
    try:
        _, metrics = rendered_image(metric_tex(output))
        score = 1.0 if metrics.get("render_success") else 0.0
        named = {
            "latex_errors": metrics.get("latex_errors", 0),
            "latex_warnings": metrics.get("latex_warnings", 0),
            "latex_badboxes": metrics.get("latex_badboxes", 0),
        }
        details = f"engine={metrics.get('engine', 'unknown')}"
        return result("renderable", score, threshold, details, named)
    except TikzRenderError as error:
        metrics = error.metrics
        named = {
            "latex_errors": metrics.get("latex_errors", 0),
            "latex_warnings": metrics.get("latex_warnings", 0),
            "latex_badboxes": metrics.get("latex_badboxes", 0),
        }
        return failure("renderable", error, named)
    except Exception as error:
        return failure("renderable", error)
