"""Parallel TikZ rendering, validation and CLIP comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

import config
from first_stage_cleaner import Source, normalize_code
from output import StagingWriter
from tikz_rendering import TikzRenderError, render_tex_to_png


def image_bytes(value: Any) -> bytes:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, dict) and value.get("bytes") is not None:
        image = Image.open(BytesIO(value["bytes"])).convert("RGB")
    elif isinstance(value, dict) and value.get("path"):
        image = Image.open(value["path"]).convert("RGB")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(BytesIO(bytes(value))).convert("RGB")
    elif isinstance(value, np.ndarray):
        image = Image.fromarray(value).convert("RGB")
    else:
        image = Image.open(value).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class Reporter:
    def __init__(self) -> None:
        config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(value: str) -> str:
        return hashlib.sha1(value.encode()).hexdigest()[:16]

    @staticmethod
    def _append_csv(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def render_failure(self, result: dict) -> None:
        directory = config.FAILURE_DIR / "rendering" / result["dataset"] / self._safe(result["sample_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "original.tex").write_text(result["code"], encoding="utf-8")
        if result.get("original_image"):
            (directory / "original.png").write_bytes(result["original_image"])
        (directory / "metadata.json").write_text(
            json.dumps({key: value for key, value in result.items() if key not in {"original_image", "rendered_image"}}, indent=2, default=str),
            encoding="utf-8",
        )
        self._append_csv(config.FAILURE_DIR / "render_failures.csv", {
            "dataset": result["dataset"],
            "sample_id": result["sample_id"],
            "reason": result["reason"],
            "detail": result.get("detail", ""),
            "artifact_dir": str(directory),
        })

    def low_similarity(self, result: dict, similarity: float) -> None:
        directory = config.FAILURE_DIR / "low_similarity" / result["dataset"] / self._safe(result["sample_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "original.png").write_bytes(result["original_image"])
        (directory / "rendered.png").write_bytes(result["rendered_image"])
        (directory / "code.tex").write_text(result["code"], encoding="utf-8")
        self._append_csv(config.FAILURE_DIR / "low_similarity.csv", {
            "dataset": result["dataset"],
            "sample_id": result["sample_id"],
            "similarity": similarity,
            "threshold": config.CLIP_THRESHOLD,
            "artifact_dir": str(directory),
        })


class Renderer:
    def check_dependencies(self) -> None:
        def exists(command: str) -> bool:
            return bool(config.LATEX_BIN_DIR and (config.LATEX_BIN_DIR / command).exists()) or shutil.which(command) is not None

        if not any(exists(engine) for engine in config.LATEX_ENGINES):
            raise RuntimeError("No configured LaTeX engine was found")
        missing = [command for command in ("pdfinfo", "pdftoppm") if not exists(command)]
        if missing:
            raise RuntimeError(f"Missing commands: {', '.join(missing)}")

    def render(self, code: str) -> tuple[bytes, dict]:
        metrics: dict = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "render.png"
            try:
                render_tex_to_png(
                    tex_code=code,
                    output_path=path,
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

            if int(metrics.get("latex_errors") or 0) > 0:
                raise RuntimeError("latex_error")
            if metrics.get("pdf_pages") != 1:
                raise RuntimeError(f"multipage_pdf: {metrics.get('pdf_pages')}")

            with Image.open(path) as image:
                gray = np.asarray(image.convert("L"))
            if float(np.mean(gray < 250)) < config.MIN_INK_FRACTION:
                raise RuntimeError("blank_image")
            return path.read_bytes(), metrics


class ClipAnalyzer:
    def __init__(self) -> None:
        if config.CLIP_DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = config.CLIP_DEVICE
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL, cache_dir=str(config.CACHE_DIR / "clip")).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(config.CLIP_MODEL, cache_dir=str(config.CACHE_DIR / "clip"))

    def analyze(self, rows: list[dict]) -> list[tuple[float, list[float]]]:
        original = [Image.open(BytesIO(row["original_image"])).convert("RGB") for row in rows]
        rendered = [Image.open(BytesIO(row["rendered_image"])).convert("RGB") for row in rows]
        inputs = self.processor(images=original + rendered, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            embeddings = self.model.get_image_features(**inputs)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        size = len(rows)
        similarities = (embeddings[:size] * embeddings[size:]).sum(dim=-1)
        return [
            (float(similarities[index].cpu()), embeddings[size + index].cpu().tolist())
            for index in range(size)
        ]


class DatasetProcessor:
    def __init__(self) -> None:
        self.renderer = Renderer()
        self.clip = ClipAnalyzer()
        self.reporter = Reporter()

    def run(self, dataset_name: str, dataset, source: Source) -> dict:
        writer = StagingWriter(dataset_name)
        stats = {"input": len(dataset), "accepted": 0, "failed": 0, "low_similarity": 0}
        clip_rows: list[dict] = []

        def flush_clip() -> None:
            if not clip_rows:
                return
            try:
                analyzed = self.clip.analyze(clip_rows)
                pairs = list(zip(clip_rows, analyzed))
            except Exception:
                pairs = []
                for row in clip_rows:
                    try:
                        pairs.append((row, self.clip.analyze([row])[0]))
                    except Exception as error:
                        row.update(reason="clip_error", detail=repr(error))
                        self.reporter.render_failure(row)
                        stats["failed"] += 1

            for row, (similarity, embedding) in pairs:
                if similarity < config.CLIP_THRESHOLD:
                    stats["low_similarity"] += 1
                    self.reporter.low_similarity(row, similarity)
                writer.add({
                    "sample_id": row["sample_id"],
                    "reference_code": row["code"],
                    "source": row["source"],
                    "rendered_image": row["rendered_image"],
                    "clip_similarity": similarity,
                    "clip_embedding": embedding,
                })
                stats["accepted"] += 1
            clip_rows.clear()

        with ThreadPoolExecutor(max_workers=config.RENDER_WORKERS) as executor:
            block_size = max(config.RENDER_WORKERS * 4, 1)
            progress = tqdm(total=len(dataset), desc=f"Render {dataset_name}")
            for start in range(0, len(dataset), block_size):
                rows = [dataset[index] for index in range(start, min(start + block_size, len(dataset)))]
                results = executor.map(lambda row: self._process_row(dataset_name, row, source), rows)
                for result in results:
                    progress.update(1)
                    if result.get("reason"):
                        stats["failed"] += 1
                        self.reporter.render_failure(result)
                    else:
                        clip_rows.append(result)
                        if len(clip_rows) >= config.CLIP_BATCH_SIZE:
                            flush_clip()
            progress.close()

        flush_clip()
        stats["staging"] = writer.close()
        return stats

    def _process_row(self, dataset_name: str, row: dict, source: Source) -> dict:
        result = {
            "dataset": dataset_name,
            "sample_id": str(row["_sample_id"]),
            "code": normalize_code(row[str(source.columns["code"])]),
            "source": str(row.get(str(source.columns["source"])) or ""),
        }
        try:
            result["original_image"] = image_bytes(row[str(source.columns["image"])])
            result["rendered_image"], result["metrics"] = self.renderer.render(result["code"])
        except Exception as error:
            message = str(error)
            result["reason"] = message.split(":", 1)[0] or "render_error"
            result["detail"] = repr(error)
        return result
