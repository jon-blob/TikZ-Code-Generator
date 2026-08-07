import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops


class TikzRenderError(RuntimeError):
    def __init__(self, message, metrics=None):
        super().__init__(message)
        self.metrics = metrics or {}


def env_bool(name, default):
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def run(cmd, cwd):
    timeout = int(os.getenv("LATEX_TIMEOUT", "45"))

    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            stderr=f"TIMEOUT after {timeout}s",
        )


def count_latex_issues(log_text):
    errors = 0
    warnings = 0
    badboxes = 0

    for line in log_text.splitlines():
        s = line.strip()

        if s.startswith("!"):
            errors += 1

        elif re.match(r"^\.?/?.*\.tex:\d+:", s):
            errors += 1

        if "Warning:" in s and not s.startswith("Package rerunfilecheck Warning:"):
            warnings += 1

        if s.startswith("Overfull \\") or s.startswith("Underfull \\"):
            badboxes += 1

    return {
        "latex_errors": errors,
        "latex_warnings": warnings,
        "latex_badboxes": badboxes,
    }


def get_pdf_page_count(pdf_path: Path, cwd: Path) -> int | None:
    """
    Gibt die Seitenzahl eines PDFs zurück.
    Nutzt bevorzugt pdfinfo. Falls nicht vorhanden oder fehlerhaft: None.
    """
    if not shutil.which("pdfinfo"):
        return None

    result = run(["pdfinfo", str(pdf_path)], cwd)

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None

    return None


def disable_page_numbers(tex):
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


def crop_png(path, padding=4, tolerance=10):
    path = Path(path)

    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, "white")

    diff = ImageChops.difference(img, bg)
    diff = diff.point(lambda p: 255 if p > tolerance else 0)

    bbox = diff.getbbox()
    if bbox is None:
        return

    l, t, r, b = bbox
    l = max(l - padding, 0)
    t = max(t - padding, 0)
    r = min(r + padding, img.width)
    b = min(b + padding, img.height)

    img.crop((l, t, r, b)).save(path)


def normalize_canvas(path, size=384, upscale=True):
    path = Path(path)

    img = Image.open(path).convert("RGBA")
    w, h = img.size

    scale = min(size / w, size / h)
    if not upscale:
        scale = min(scale, 1.0)

    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), "white")
    canvas.alpha_composite(img, ((size - nw) // 2, (size - nh) // 2))
    canvas.convert("RGB").save(path)


def clean_markdown_tex(tex_code: str) -> str:
    if not tex_code:
        return ""

    tex_code = re.sub(
        r"^\s*```[a-zA-Z]*\s*\n?",
        "",
        tex_code,
        flags=re.IGNORECASE,
    )
    tex_code = re.sub(r"\n?\s*```\s*$", "", tex_code)

    return tex_code.strip()


def render_tex_to_png(tex_code, output_path, metrics=None, create_ds=False):
    tex_code = clean_markdown_tex(tex_code)

    if metrics is None:
        metrics = {}

    engines = [
        e.strip()
        for e in os.getenv("LATEX_ENGINES", "pdflatex,lualatex,xelatex").split(",")
        if e.strip()
    ]

    dpi = int(os.getenv("LATEX_DPI", "600"))
    size = int(os.getenv("REF_IMAGE_SIZE", "512"))

    crop_pdf = env_bool("LATEX_CROP_PDF", True)
    crop_png_enabled = env_bool("LATEX_CROP_PNG", True)
    normalize = env_bool("LATEX_NORMALIZE_CANVAS", True)
    upscale = env_bool("LATEX_UPSCALE_CANVAS", True)

    halt_on_error = env_bool("LATEX_HALT_ON_ERROR", True)
    tolerant_fallback = env_bool("LATEX_TOLERANT_FALLBACK", True)
    disable_pages = env_bool("LATEX_DISABLE_PAGE_NUMBERS", True)

    if disable_pages:
        tex_code = disable_page_numbers(tex_code)

    halt_modes = [halt_on_error]
    if halt_on_error and tolerant_fallback:
        halt_modes.append(False)

    errors = []
    last_metrics = {}

    for engine in engines:
        if not shutil.which(engine):
            continue

        for halt in halt_modes:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)

                tex = tmp / "figure.tex"
                pdf = tmp / "figure.pdf"
                pdf_crop = tmp / "figure-crop.pdf"
                png = tmp / "figure.png"
                log = tmp / "figure.log"

                tex.write_text(tex_code, encoding="utf-8")

                cmd = [engine, "-interaction=nonstopmode", "-file-line-error"]
                if halt:
                    cmd.append("-halt-on-error")
                cmd.append(tex.name)

                result = run(cmd, tmp)

                log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

                issues = count_latex_issues(log_text)

                attempt_metrics = {
                    **issues,
                    "latex_returncode": result.returncode,
                    "pdf_created": pdf.exists(),
                    "engine": engine,
                    "halt_on_error": halt,
                    "pdf_pages": None,
                    "multipage_rejected": False,
                }

                last_metrics = attempt_metrics
                metrics.update(attempt_metrics)

                if result.returncode != 0 and not pdf.exists():
                    errors.append(result.stdout[-3000:] + result.stderr[-3000:])
                    continue

                if not pdf.exists():
                    errors.append("No PDF created.")
                    continue

                # ====================================================
                # DATASET MODE: reject multi-page PDFs
                # ====================================================
                if create_ds:
                    page_count = get_pdf_page_count(pdf, tmp)
                    attempt_metrics["pdf_pages"] = page_count
                    metrics["pdf_pages"] = page_count

                    if page_count is not None and page_count != 1:
                        attempt_metrics["multipage_rejected"] = True
                        metrics["multipage_rejected"] = True
                        errors.append(f"Rejected multi-page PDF: pages={page_count}")
                        continue

                pdf_to_render = pdf

                if crop_pdf and shutil.which("pdfcrop"):
                    result = run(
                        ["pdfcrop", "--margins", "0", str(pdf), str(pdf_crop)],
                        tmp,
                    )
                    if result.returncode == 0 and pdf_crop.exists():
                        pdf_to_render = pdf_crop

                result = run(
                    [
                        "pdftoppm",
                        "-png",
                        "-singlefile",
                        "-r",
                        str(dpi),
                        str(pdf_to_render),
                        str(tmp / "figure"),
                    ],
                    tmp,
                )

                if result.returncode != 0 or not png.exists():
                    errors.append("pdftoppm failed:\n" + result.stderr[-3000:])
                    continue

                if crop_png_enabled:
                    crop_png(png)

                if normalize:
                    normalize_canvas(png, size=size, upscale=upscale)

                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(png.read_bytes())

                metrics["render_success"] = True
                return output_path

    metrics.update(last_metrics)
    metrics["render_success"] = False

    raise TikzRenderError(
        "Render failed:\n" + "\n\n".join(errors[-3:]),
        metrics=metrics,
    )