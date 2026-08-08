"""Render LaTeX/TikZ source to a normalized PNG.

This module is based on the previous renderer, but all settings are explicit
function arguments. It does not depend on environment variables.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops


class TikzRenderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reason: str = "render_failed",
        metrics: dict | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.metrics = metrics or {}


def _resolve_command(name: str, preferred_bin_dir: Path | None = None) -> str | None:
    if preferred_bin_dir:
        candidate = preferred_bin_dir / name
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def _run(command: list[str], cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        return subprocess.CompletedProcess(
            command,
            returncode=124,
            stdout=stdout,
            stderr=f"TIMEOUT after {timeout_seconds}s",
        )


def count_latex_issues(log_text: str) -> dict[str, int]:
    errors = 0
    warnings = 0
    badboxes = 0

    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("!") or re.match(r"^\.?/?.*\.tex:\d+:", stripped):
            errors += 1
        if "Warning:" in stripped and not stripped.startswith(
            "Package rerunfilecheck Warning:"
        ):
            warnings += 1
        if stripped.startswith("Overfull \\") or stripped.startswith("Underfull \\"):
            badboxes += 1

    return {
        "latex_errors": errors,
        "latex_warnings": warnings,
        "latex_badboxes": badboxes,
    }


def get_pdf_page_count(
    pdf_path: Path,
    cwd: Path,
    *,
    timeout_seconds: int,
    preferred_bin_dir: Path | None = None,
) -> int | None:
    executable = _resolve_command("pdfinfo", preferred_bin_dir)
    if executable is None:
        return None

    result = _run([executable, str(pdf_path)], cwd, timeout_seconds)
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def disable_page_numbers(tex: str) -> str:
    if r"\begin{document}" not in tex:
        return tex

    tex = tex.replace(
        r"\begin{document}",
        r"\pagestyle{empty}" + "\n" + r"\begin{document}",
        1,
    )
    tex = tex.replace(
        r"\begin{document}",
        r"\begin{document}" + "\n" + r"\thispagestyle{empty}",
        1,
    )
    return tex


def crop_png(path: Path, padding: int = 4, tolerance: int = 10) -> None:
    with Image.open(path) as source:
        image = source.convert("RGB")
    background = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, background)
    difference = difference.point(lambda pixel: 255 if pixel > tolerance else 0)
    bounding_box = difference.getbbox()
    if bounding_box is None:
        return

    left, top, right, bottom = bounding_box
    left = max(left - padding, 0)
    top = max(top - padding, 0)
    right = min(right + padding, image.width)
    bottom = min(bottom + padding, image.height)
    image.crop((left, top, right, bottom)).save(path)


def normalize_canvas(path: Path, *, size: int, upscale: bool = True) -> None:
    with Image.open(path) as source:
        image = source.convert("RGBA")

    width, height = image.size
    scale = min(size / width, size / height)
    if not upscale:
        scale = min(scale, 1.0)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), "white")
    canvas.alpha_composite(
        image,
        ((size - new_width) // 2, (size - new_height) // 2),
    )
    canvas.convert("RGB").save(path)


def clean_markdown_tex(tex_code: str) -> str:
    if not tex_code:
        return ""
    tex_code = re.sub(
        r"^\s*```(?:latex|tex|tikz)?\s*\n?",
        "",
        tex_code,
        flags=re.IGNORECASE,
    )
    tex_code = re.sub(r"\n?\s*```\s*$", "", tex_code)
    return tex_code.strip()


def render_tex_to_png(
    tex_code: str,
    output_path: str | Path,
    metrics: dict | None = None,
    create_ds: bool = False,
    *,
    engines: Iterable[str] = ("pdflatex", "lualatex", "xelatex"),
    timeout_seconds: int = 45,
    dpi: int = 400,
    image_size: int = 512,
    preferred_bin_dir: str | Path | None = None,
    crop_pdf: bool = True,
    crop_png_enabled: bool = True,
    normalize: bool = True,
    upscale: bool = True,
    halt_on_error: bool = True,
    tolerant_fallback: bool = True,
    disable_pages: bool = True,
) -> Path:
    """Render one source file and return the created PNG path."""

    tex_code = clean_markdown_tex(tex_code)
    if not tex_code:
        raise TikzRenderError("The LaTeX source is empty.", reason="empty_code")

    output_path = Path(output_path)
    preferred = Path(preferred_bin_dir) if preferred_bin_dir else None
    metrics = metrics if metrics is not None else {}

    if disable_pages:
        tex_code = disable_page_numbers(tex_code)

    halt_modes = [halt_on_error]
    if halt_on_error and tolerant_fallback:
        halt_modes.append(False)

    errors: list[str] = []
    found_engine = False
    last_metrics: dict = {}

    for engine_name in engines:
        engine = _resolve_command(engine_name, preferred)
        if engine is None:
            continue
        found_engine = True

        for halt in halt_modes:
            with tempfile.TemporaryDirectory() as temporary_directory:
                temporary = Path(temporary_directory)
                tex_path = temporary / "figure.tex"
                pdf_path = temporary / "figure.pdf"
                cropped_pdf_path = temporary / "figure-crop.pdf"
                png_path = temporary / "figure.png"
                log_path = temporary / "figure.log"
                tex_path.write_text(tex_code, encoding="utf-8")

                command = [engine, "-interaction=nonstopmode", "-file-line-error"]
                if halt:
                    command.append("-halt-on-error")
                command.append(tex_path.name)

                result = _run(command, temporary, timeout_seconds)
                log_text = (
                    log_path.read_text(encoding="utf-8", errors="replace")
                    if log_path.exists()
                    else ""
                )
                issues = count_latex_issues(log_text)
                attempt_metrics = {
                    **issues,
                    "latex_returncode": result.returncode,
                    "pdf_created": pdf_path.exists(),
                    "engine": engine_name,
                    "halt_on_error": halt,
                    "pdf_pages": None,
                    "multipage_rejected": False,
                }
                last_metrics = attempt_metrics
                metrics.update(attempt_metrics)

                if result.returncode == 124:
                    errors.append(result.stderr)
                    continue
                if result.returncode != 0 and not pdf_path.exists():
                    errors.append((result.stdout + "\n" + result.stderr)[-4_000:])
                    continue
                if not pdf_path.exists():
                    errors.append("No PDF was created.")
                    continue

                if create_ds:
                    page_count = get_pdf_page_count(
                        pdf_path,
                        temporary,
                        timeout_seconds=timeout_seconds,
                        preferred_bin_dir=preferred,
                    )
                    metrics["pdf_pages"] = page_count
                    attempt_metrics["pdf_pages"] = page_count
                    if page_count is None:
                        errors.append("Could not determine the PDF page count.")
                        continue
                    if page_count != 1:
                        metrics["multipage_rejected"] = True
                        attempt_metrics["multipage_rejected"] = True
                        raise TikzRenderError(
                            f"Expected one page, got {page_count}.",
                            reason="multipage_pdf",
                            metrics=dict(metrics),
                        )

                pdf_to_render = pdf_path
                pdfcrop = _resolve_command("pdfcrop", preferred)
                if crop_pdf and pdfcrop:
                    crop_result = _run(
                        [pdfcrop, "--margins", "0", str(pdf_path), str(cropped_pdf_path)],
                        temporary,
                        timeout_seconds,
                    )
                    if crop_result.returncode == 0 and cropped_pdf_path.exists():
                        pdf_to_render = cropped_pdf_path

                pdftoppm = _resolve_command("pdftoppm", preferred)
                if pdftoppm is None:
                    raise TikzRenderError(
                        "pdftoppm is not installed.",
                        reason="missing_dependency",
                        metrics=dict(metrics),
                    )

                convert_result = _run(
                    [
                        pdftoppm,
                        "-png",
                        "-singlefile",
                        "-r",
                        str(dpi),
                        str(pdf_to_render),
                        str(temporary / "figure"),
                    ],
                    temporary,
                    timeout_seconds,
                )
                if convert_result.returncode != 0 or not png_path.exists():
                    errors.append("pdftoppm failed:\n" + convert_result.stderr[-4_000:])
                    continue

                if crop_png_enabled:
                    crop_png(png_path)
                if normalize:
                    normalize_canvas(png_path, size=image_size, upscale=upscale)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(png_path.read_bytes())
                metrics["render_success"] = True
                return output_path

    metrics.update(last_metrics)
    metrics["render_success"] = False
    if not found_engine:
        raise TikzRenderError(
            "No configured LaTeX engine was found.",
            reason="missing_dependency",
            metrics=dict(metrics),
        )

    reason = "latex_timeout" if any("TIMEOUT" in value for value in errors) else "render_failed"
    raise TikzRenderError(
        "Render failed:\n" + "\n\n".join(errors[-3:]),
        reason=reason,
        metrics=dict(metrics),
    )
