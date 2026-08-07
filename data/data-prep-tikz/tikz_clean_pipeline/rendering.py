"""Rendering plus one-page and non-blank validation."""

from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

import config
from tikz_rendering import render_tex_to_png


class RenderRejected(RuntimeError):
    pass


class ValidatedRenderer:
    REQUIRED_COMMANDS = ("pdflatex", "pdftoppm", "pdfinfo")

    def check_dependencies(self) -> None:
        missing = [name for name in self.REQUIRED_COMMANDS if shutil.which(name) is None]
        if missing:
            raise RuntimeError(f"Missing system commands: {', '.join(missing)}")

    def render(self, tex: str) -> bytes:
        metrics: dict = {}

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.png"
            render_tex_to_png(
                tex_code=tex,
                output_path=image_path,
                metrics=metrics,
                create_ds=True,
            )

            latex_errors = int(metrics.get("latex_errors") or 0)
            latex_warnings = int(metrics.get("latex_warnings") or 0)
            latex_badboxes = int(metrics.get("latex_badboxes") or 0)

            if latex_errors:
                raise RenderRejected(
                    "LaTeX issues detected: "
                    f"errors={latex_errors}, "
                    f"warnings={latex_warnings}, "
                    f"badboxes={latex_badboxes}."
                )

            if metrics.get("pdf_pages") != 1:
                raise RenderRejected(
                    f"Expected exactly one PDF page, got {metrics.get('pdf_pages')}."
                )

            with Image.open(image_path) as image:
                gray = np.asarray(image.convert("L"))
                ink_fraction = float(np.mean(gray < config.WHITE_PIXEL_THRESHOLD))

            if ink_fraction < config.MIN_INK_FRACTION:
                raise RenderRejected(
                    f"Rendered image is blank: ink_fraction={ink_fraction:.6f}."
                )

            return image_path.read_bytes()
