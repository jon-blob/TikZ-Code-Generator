from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from PIL import Image, ImageChops

from config import RENDER

FENCED_TEX_PATTERN = re.compile(
    r"`(?:latex|tex)?[ \t]*\r?\n?(.*?)`",
    flags=re.IGNORECASE | re.DOTALL,
)


class TikzRenderError(RuntimeError):
    def __init__(self, message: str, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics or {}


def executable(name: str, texlive: bool = False) -> str | None:
    if texlive:
        candidate = RENDER.texlive_bin / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RENDER.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        return subprocess.CompletedProcess(
            command,
            124,
            stdout,
            f"TIMEOUT after {RENDER.timeout_seconds}s",
        )


def latex_issues(log_text: str) -> dict[str, int]:
    errors = warnings = badboxes = 0
    for line in log_text.splitlines():
        text = line.strip()
        if text.startswith("!") or re.match(r"^.?/?.\*.tex:\d+:", text):
            errors += 1
        if "Warning:" in text and not text.startswith("Package rerunfilecheck Warning:"):
            warnings += 1
        if text.startswith(("Overfull \\", "Underfull \\")):
            badboxes += 1
    return {
        "latex_errors": errors,
        "latex_warnings": warnings,
        "latex_badboxes": badboxes,
    }


def clean_tex(code: str) -> str:
    if not code:
        return ""

    code = code.strip().lstrip("\ufeff")

    match = FENCED_TEX_PATTERN.search(code)
    if match:
        return match.group(1).strip()

    return code


def disable_page_numbers(code: str) -> str:
    marker = r"\begin{document}"
    if marker not in code:
        return code
    code = code.replace(marker, r"\pagestyle{empty}" + "\n" + marker, 1)
    return code.replace(marker, marker + "\n" + r"\thispagestyle{empty}", 1)


def crop_png(path: Path, padding: int = 4, tolerance: int = 10) -> None:
    image = Image.open(path).convert("RGB")
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    difference = difference.point(lambda value: 255 if value > tolerance else 0)
    box = difference.getbbox()
    if not box:
        return
    left, top, right, bottom = box
    image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    ).save(path)


def normalize_canvas(path: Path) -> None:
    image = Image.open(path).convert("RGBA")
    scale = min(RENDER.image_size / image.width, RENDER.image_size / image.height)
    if not RENDER.upscale_canvas:
        scale = min(scale, 1.0)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (RENDER.image_size, RENDER.image_size), "white")
    canvas.alpha_composite(image, ((RENDER.image_size - width) // 2, (RENDER.image_size - height) // 2))
    canvas.convert("RGB").save(path)


def pdf_page_count(pdf: Path, cwd: Path) -> int | None:
    pdfinfo = executable("pdfinfo")
    if not pdfinfo:
        return None
    result = run([pdfinfo, str(pdf)], cwd)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def render_tex_to_png(
    tex_code: str,
    output_path: str | Path,
    metrics: dict | None = None,
    create_ds: bool = False,
) -> Path:
    metrics = metrics if metrics is not None else {}
    code = clean_tex(tex_code)
    if RENDER.disable_page_numbers:
        code = disable_page_numbers(code)

    halt_modes = [RENDER.halt_on_error]
    if RENDER.halt_on_error and RENDER.tolerant_fallback:
        halt_modes.append(False)

    errors: list[str] = []
    last_metrics: dict = {}

    for engine in RENDER.engines:
        engine_path = executable(engine, texlive=True)
        if not engine_path:
            continue

        for halt in halt_modes:
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                tex = directory / "figure.tex"
                pdf = directory / "figure.pdf"
                cropped_pdf = directory / "figure-crop.pdf"
                png = directory / "figure.png"
                log = directory / "figure.log"
                tex.write_text(code, encoding="utf-8")

                command = [engine_path, "-interaction=nonstopmode", "-file-line-error"]
                if halt:
                    command.append("-halt-on-error")
                command.append(tex.name)
                completed = None
                run_outputs: list[str] = []

                for _ in range(max(1, RENDER.latex_runs)):
                    completed = run(command, directory)

                    run_outputs.append(
                        completed.stdout + "\n" + completed.stderr
                    )

                    if completed.returncode != 0 and not pdf.exists():
                        break

                assert completed is not None

                log_text = (
                    log.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    if log.exists()
                    else ""
                )

                attempt = {
                    **latex_issues(log_text),
                    "latex_returncode": completed.returncode,
                    "pdf_created": pdf.exists(),
                    "engine": engine,
                    "halt_on_error": halt,
                    "pdf_pages": None,
                    "multipage_rejected": False,
                }
                last_metrics = attempt
                metrics.update(attempt)

                if completed.returncode != 0 and not pdf.exists():
                    errors.append("\n".join(run_outputs)[-6000:])
                    continue
                if not pdf.exists():
                    errors.append("No PDF created.")
                    continue

                if create_ds:
                    pages = pdf_page_count(pdf, directory)
                    metrics["pdf_pages"] = pages
                    if pages is not None and pages != 1:
                        metrics["multipage_rejected"] = True
                        errors.append(f"Rejected multi-page PDF: pages={pages}")
                        continue

                pdf_to_render = pdf
                pdfcrop = executable("pdfcrop", texlive=True)
                if RENDER.crop_pdf and pdfcrop:
                    crop = run([pdfcrop, "--margins", "0", str(pdf), str(cropped_pdf)], directory)
                    if crop.returncode == 0 and cropped_pdf.exists():
                        pdf_to_render = cropped_pdf

                pdftoppm = executable("pdftoppm")
                if not pdftoppm:
                    errors.append("pdftoppm not found.")
                    continue

                converted = run(
                    [
                        pdftoppm,
                        "-png",
                        "-singlefile",
                        "-r",
                        str(RENDER.dpi),
                        str(pdf_to_render),
                        str(directory / "figure"),
                    ],
                    directory,
                )
                if converted.returncode != 0 or not png.exists():
                    errors.append("pdftoppm failed:\n" + converted.stderr[-3000:])
                    continue

                if RENDER.crop_png:
                    crop_png(png)
                if RENDER.normalize_canvas:
                    normalize_canvas(png)

                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(png.read_bytes())
                metrics["render_success"] = True
                return output

    metrics.update(last_metrics)
    metrics["render_success"] = False
    raise TikzRenderError("Render failed:\n" + "\n\n".join(errors[-3:]), metrics)
