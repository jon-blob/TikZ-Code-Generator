"""Failure and low-similarity reporting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import config


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

    def low_similarity(self, result: dict, similarity: float, threshold: float) -> None:
        path = self._directory("low_similarity", result)
        (path / "original.png").write_bytes(result["original_image"])
        (path / "rendered.png").write_bytes(result["rendered_image"])
        (path / "code.tex").write_text(result["code"], encoding="utf-8")
        self._append(
            config.FAILURE_DIR / "low_similarity.csv",
            {
                "dataset": result["dataset"],
                "sample_id": result["sample_id"],
                "encoder": config.IMAGE_ENCODER,
                "similarity": similarity,
                "threshold": threshold,
                "artifact_dir": str(path),
            },
        )
