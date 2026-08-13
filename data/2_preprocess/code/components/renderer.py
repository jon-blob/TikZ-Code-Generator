"""TikZ rendering and render validation."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import config
from tikz_rendering import TikzRenderError, render_tex_to_png


class Renderer:
    def check_dependencies(self) -> None:
        def exists(command: str) -> bool:
            local = config.LATEX_BIN_DIR and (config.LATEX_BIN_DIR / command).exists()
            return bool(local) or shutil.which(command) is not None

        if not any(exists(engine) for engine in config.LATEX_ENGINES):
            raise RuntimeError("No LaTeX engine found")

        missing = [command for command in ("pdfinfo", "pdftoppm") if not exists(command)]
        if missing:
            raise RuntimeError(f"Missing commands: {', '.join(missing)}")

    def render(self, code: str) -> tuple[bytes, dict]:
        metrics: dict = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "render.png"
            try:
                render_tex_to_png(
                    tex_code=code,
                    output_path=output,
                    metrics=metrics,
                    create_ds=True,
                    engines=config.LATEX_ENGINES,
                    timeout_seconds=config.LATEX_TIMEOUT,
                    dpi=config.DPI,
                    image_size=config.IMAGE_SIZE,
                    preferred_bin_dir=config.LATEX_BIN_DIR,
                    crop_pdf=True,
                    crop_png_enabled=True,
                    normalize=True,
                    upscale=True,
                    halt_on_error=True,
                    tolerant_fallback=True,
                    disable_pages=True,
                )
            except TikzRenderError as error:
                raise RuntimeError(f"{error.reason}: {error}") from error

            if metrics.get("latex_errors", 0):
                raise RuntimeError("latex_error")
            if metrics.get("pdf_pages") != 1:
                raise RuntimeError(f"multipage_pdf: {metrics.get('pdf_pages')}")

            with Image.open(output) as image:
                ink_fraction = float(np.mean(np.asarray(image.convert("L")) < 250))
            if ink_fraction < config.MIN_INK_FRACTION:
                raise RuntimeError("blank_image")

            return output.read_bytes(), metrics
