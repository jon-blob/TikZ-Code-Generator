from dataclasses import dataclass
from pathlib import Path
import tempfile
import traceback

from PIL import Image


from pf_utils.tikz_rendering import (
    render_tex_to_png,
    TikzRenderError,
)


@dataclass
class RenderResult:
    ok: bool
    image: Image.Image | None
    errors: int
    warnings: int
    badboxes: int
    reason: str
    metrics: dict


def is_renderable(tex_code: str) -> RenderResult:
    metrics = {}

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_png = Path(tmp_dir) / "rendered.png"

            render_tex_to_png(
                tex_code=tex_code,
                output_path=output_png,
                metrics=metrics,
            )

            ok = output_png.exists() and output_png.stat().st_size > 0
            image = None
            if ok:
                with Image.open(output_png) as rendered:
                    image = rendered.convert("RGB").copy()

        errors = metrics.get("latex_errors", 0)
        warnings = metrics.get("latex_warnings", 0)
        badboxes = metrics.get("latex_badboxes", 0)

        return RenderResult(
            ok=ok,
            image=image,
            errors=errors,
            warnings=warnings,
            badboxes=badboxes,
            reason=f"renderable={ok}; errors={errors}; warnings={warnings}; badboxes={badboxes}",
            metrics=metrics,
        )

    except TikzRenderError as e:
        metrics.update(getattr(e, "metrics", {}))

        return RenderResult(
            ok=False,
            image=None,
            errors=metrics.get("latex_errors", 0),
            warnings=metrics.get("latex_warnings", 0),
            badboxes=metrics.get("latex_badboxes", 0),
            reason=f"not renderable: {e}",
            metrics=metrics,
        )

    except Exception as e:
        return RenderResult(
            ok=False,
            image=None,
            errors=metrics.get("latex_errors", 0),
            warnings=metrics.get("latex_warnings", 0),
            badboxes=metrics.get("latex_badboxes", 0),
            reason=f"not renderable: {e}\n{traceback.format_exc()}",
            metrics=metrics,
        )