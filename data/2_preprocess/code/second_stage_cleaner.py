"""Render TikZ samples, compare images with CLIP and write staging data."""

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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import traceback

import config
from first_stage_cleaner import Source, normalize_code
from output import StagingWriter
from tikz_rendering import TikzRenderError, render_tex_to_png


def to_png_bytes(value: Any) -> bytes:
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, dict) and value.get("bytes") is not None:
        image = Image.open(BytesIO(value["bytes"]))
    elif isinstance(value, dict) and value.get("path"):
        image = Image.open(value["path"])
    elif isinstance(value, (bytes, bytearray, memoryview)):
        image = Image.open(BytesIO(bytes(value)))
    elif isinstance(value, np.ndarray):
        image = Image.fromarray(value)
    else:
        image = Image.open(value)

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class Reporter:
    def __init__(self) -> None:
        config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _append(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _directory(category: str, result: dict) -> Path:
        sample_hash = hashlib.sha1(result["sample_id"].encode()).hexdigest()[:16]
        path = config.FAILURE_DIR / category / result["dataset"] / sample_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def failure(self, result: dict) -> None:
        path = self._directory("rendering", result)
        (path / "code.tex").write_text(result["code"], encoding="utf-8")

        if result.get("original_image"):
            (path / "original.png").write_bytes(result["original_image"])

        metadata = {
            key: value
            for key, value in result.items()
            if key not in {"original_image", "rendered_image"}
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

        self._append(
            config.FAILURE_DIR / "render_failures.csv",
            {
                "dataset": result["dataset"],
                "sample_id": result["sample_id"],
                "reason": result["reason"],
                "detail": result.get("detail", ""),
                "artifact_dir": str(path),
            },
        )

    def low_similarity(self, result: dict, similarity: float) -> None:
        path = self._directory("low_similarity", result)
        (path / "original.png").write_bytes(result["original_image"])
        (path / "rendered.png").write_bytes(result["rendered_image"])
        (path / "code.tex").write_text(result["code"], encoding="utf-8")

        self._append(
            config.FAILURE_DIR / "low_similarity.csv",
            {
                "dataset": result["dataset"],
                "sample_id": result["sample_id"],
                "similarity": similarity,
                "threshold": config.CLIP_THRESHOLD,
                "artifact_dir": str(path),
            },
        )


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


class ClipAnalyzer:
    def __init__(self) -> None:
        if config.CLIP_DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.CLIP_DEVICE

        self.device = torch.device(device)
        cache_dir = str(config.CACHE_DIR / "clip")

        self.model = (
            CLIPVisionModelWithProjection.from_pretrained(
                config.CLIP_MODEL,
                cache_dir=cache_dir,
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            config.CLIP_MODEL,
            cache_dir=cache_dir,
        )

    def _encode(self, images: list[Image.Image]) -> torch.Tensor:
        pixel_values = self.processor(
            images=images,
            return_tensors="pt",
        )["pixel_values"].to(self.device)

        with torch.inference_mode():
            vision_output = self.model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            pooled = vision_output.pooler_output

            if not isinstance(pooled, torch.Tensor):
                raise TypeError(
                    f"Expected pooler_output tensor, got {type(pooled).__name__}"
                )

            embeddings = self.model.visual_projection(pooled)

            if not isinstance(embeddings, torch.Tensor):
                raise TypeError(
                    f"Expected projection tensor, got {type(embeddings).__name__}"
                )

            return F.normalize(embeddings.to(torch.float32), dim=-1)

    def analyze(self, rows: list[dict]) -> list[tuple[float, list[float]]]:
        original = [
            Image.open(BytesIO(row["original_image"])).convert("RGB")
            for row in rows
        ]
        rendered = [
            Image.open(BytesIO(row["rendered_image"])).convert("RGB")
            for row in rows
        ]

        embeddings = self._encode(original + rendered)
        size = len(rows)

        original_embeddings = embeddings[:size]
        rendered_embeddings = embeddings[size:]
        similarities = (original_embeddings * rendered_embeddings).sum(dim=-1)

        return [
            (
                float(similarities[index].cpu()),
                rendered_embeddings[index].cpu().tolist(),
            )
            for index in range(size)
        ]


class DatasetProcessor:
    def __init__(self) -> None:
        self.renderer = Renderer()
        self.clip = ClipAnalyzer()
        self.reporter = Reporter()

    def run(self, dataset_name: str, dataset, source: Source) -> dict:
        writer = StagingWriter(dataset_name)
        stats = {
            "input": len(dataset),
            "accepted": 0,
            "failed": 0,
            "low_similarity": 0,
        }
        clip_rows: list[dict] = []

        def flush() -> None:
            if not clip_rows:
                return

            try:
                pairs = list(zip(clip_rows, self.clip.analyze(clip_rows)))
            except Exception:
                pairs = []
                for row in clip_rows:
                    try:
                        pairs.append((row, self.clip.analyze([row])[0]))
                    except Exception:
                        row.update(
                            reason="clip_error",
                            detail=traceback.format_exc(),
                        )
                        self.reporter.failure(row)
                        stats["failed"] += 1

            for row, (similarity, embedding) in pairs:
                if similarity < config.CLIP_THRESHOLD:
                    stats["low_similarity"] += 1
                    self.reporter.low_similarity(row, similarity)

                writer.add(
                    {
                        "sample_id": row["sample_id"],
                        "reference_code": row["code"],
                        "source": row["source"],
                        "rendered_image": row["rendered_image"],
                        "clip_similarity": similarity,
                        "clip_embedding": embedding,
                    }
                )
                stats["accepted"] += 1

            clip_rows.clear()

        block_size = max(config.RENDER_WORKERS * 4, 1)

        with ThreadPoolExecutor(max_workers=config.RENDER_WORKERS) as executor:
            with tqdm(total=len(dataset), desc=f"Render {dataset_name}") as progress:
                for start in range(0, len(dataset), block_size):
                    rows = [
                        dataset[index]
                        for index in range(start, min(start + block_size, len(dataset)))
                    ]
                    results = executor.map(
                        lambda row: self._process_row(dataset_name, row, source),
                        rows,
                    )

                    for result in results:
                        progress.update(1)

                        if result.get("reason"):
                            stats["failed"] += 1
                            self.reporter.failure(result)
                            continue

                        clip_rows.append(result)
                        if len(clip_rows) >= config.CLIP_BATCH_SIZE:
                            flush()

        flush()
        stats["staging"] = writer.close()
        return stats

    def _process_row(
        self,
        dataset_name: str,
        row: dict,
        source: Source,
    ) -> dict:
        result = {
            "dataset": dataset_name,
            "sample_id": str(row["_sample_id"]),
            "code": normalize_code(row[str(source.columns["code"])]),
            "source": str(row.get(str(source.columns["source"])) or ""),
        }

        try:
            result["original_image"] = to_png_bytes(
                row[str(source.columns["image"])]
            )
            result["rendered_image"], result["metrics"] = self.renderer.render(
                result["code"]
            )
        except Exception as error:
            result["reason"] = str(error).split(":", 1)[0] or "render_error"
            result["detail"] = repr(error)

        return result