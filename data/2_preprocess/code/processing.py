"""Render, validate and encode cleaned TikZ samples into staging Parquet."""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from cleaning import Source, normalize_code
from components import ClipAnalyzer, Renderer, Reporter
from output import StagingWriter


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


class RenderedDatasetProcessor:
    def __init__(self) -> None:
        self.renderer = Renderer()
        self.analyzer = ClipAnalyzer()
        self.reporter = Reporter()

    def run(self, dataset_name: str, dataset, source: Source) -> dict:
        writer = StagingWriter(dataset_name)
        stats = {
            "input": len(dataset),
            "accepted": 0,
            "failed": 0,
            "low_similarity": 0,
        }
        analysis_rows: list[dict] = []

        def flush() -> None:
            if not analysis_rows:
                return

            try:
                pairs = list(zip(analysis_rows, self.analyzer.analyze(analysis_rows)))
            except Exception:
                pairs = []
                for row in analysis_rows:
                    try:
                        pairs.append((row, self.analyzer.analyze([row])[0]))
                    except Exception:
                        row.update(
                            reason="image_encoder_error",
                            detail=traceback.format_exc(),
                        )
                        self.reporter.failure(row)
                        stats["failed"] += 1

            for row, (similarity, embedding) in pairs:
                if similarity < self.analyzer.threshold:
                    stats["low_similarity"] += 1
                    self.reporter.low_similarity(row, similarity, self.analyzer.threshold)

                writer.add(
                    {
                        "sample_id": row["sample_id"],
                        "reference_code": row["code"],
                        "source": row["source"],
                        "rendered_image": row["rendered_image"],
                        "image_encoder": self.analyzer.encoder_name,
                        "image_similarity": similarity,
                        "image_embedding": embedding,
                    }
                )
                stats["accepted"] += 1

            analysis_rows.clear()

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

                        analysis_rows.append(result)
                        if len(analysis_rows) >= config.IMAGE_BATCH_SIZE:
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