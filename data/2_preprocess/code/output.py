"""Parquet staging, final export and Hugging Face upload."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

import config


STAGING_SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("reference_code", pa.string()),
    ("source", pa.string()),
    ("rendered_image", pa.binary()),
    ("image_encoder", pa.string()),
    ("image_similarity", pa.float32()),
    ("image_embedding", pa.list_(pa.float32())),
])

IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
FINAL_SCHEMA = pa.schema([
    ("input_image", IMAGE),
    ("reference_image", IMAGE),
    ("reference_code", pa.string()),
    ("llm_description", pa.string()),
    ("llm_description_image", pa.string()),
    ("llm_description_image_code", pa.string()),
    ("type", pa.string()),
    ("source", pa.string()),
    ("class", pa.string()),
    ("repetition_class", pa.string()),
    ("token_len", pa.int32()),
])


class ShardWriter:
    def __init__(self, output_dir: Path, prefix: str, schema: pa.Schema, max_bytes: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.schema = schema
        self.max_bytes = max_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: list[dict[str, Any]] = []
        self.estimated_bytes = 0
        self.part = 0
        self.files: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any], estimated_bytes: int) -> None:
        self.buffer.append(row)
        self.estimated_bytes += estimated_bytes
        if self.estimated_bytes >= config.PARQUET_BUFFER_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        rows, self.buffer = self.buffer, []
        self.estimated_bytes = 0
        self._write(rows)

    def close(self) -> dict:
        self.flush()
        return {
            "rows": sum(item["rows"] for item in self.files),
            "files": self.files,
        }

    def _write(self, rows: list[dict[str, Any]]) -> None:
        candidate = self.output_dir / f".{uuid.uuid4().hex}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=self.schema),
            candidate,
            compression="zstd",
            compression_level=3,
        )
        size = candidate.stat().st_size
        if size <= self.max_bytes:
            final = self.output_dir / f"{self.prefix}-{self.part:05d}.parquet"
            candidate.replace(final)
            self.files.append({"name": final.name, "rows": len(rows), "bytes": size})
            self.part += 1
            return

        candidate.unlink(missing_ok=True)
        if len(rows) == 1:
            raise ValueError(f"One row exceeds {self.max_bytes} bytes")
        middle = len(rows) // 2
        self._write(rows[:middle])
        self._write(rows[middle:])


class StagingWriter:
    def __init__(self, dataset_name: str):
        self.writer = ShardWriter(
            config.STAGING_DIR / dataset_name,
            "part",
            STAGING_SCHEMA,
            config.MAX_PARQUET_BYTES,
        )
        self.rows_in_current_target = 0

    def add(self, row: dict[str, Any]) -> None:
        estimated = (
            len(row["rendered_image"])
            + len(row["reference_code"].encode("utf-8"))
            + 4 * len(row["image_embedding"])
            + 512
        )
        self.writer.add(row, estimated)
        self.rows_in_current_target += 1
        if self.rows_in_current_target >= config.STAGING_ROWS_PER_FILE:
            self.writer.flush()
            self.rows_in_current_target = 0

    def close(self) -> dict:
        return self.writer.close()


def parquet_files(dataset_name: str) -> list[Path]:
    files = sorted((config.STAGING_DIR / dataset_name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Missing staging files for {dataset_name}")
    return files


def iter_batches(
    dataset_name: str,
    columns: list[str] | None = None,
    batch_size: int = 2_048,
) -> Iterator[dict[str, list[Any]]]:
    for path in parquet_files(dataset_name):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            yield batch.to_pydict()


def iter_rows(dataset_name: str, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    for batch in iter_batches(dataset_name, columns):
        names = list(batch)
        for values in zip(*(batch[name] for name in names)):
            yield dict(zip(names, values))


def row_count(dataset_name: str) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files(dataset_name))


class FinalExporter:
    def run(self) -> dict:
        metadata_path = config.METADATA_DIR / "metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        if config.EXPORT_DIR.exists():
            if not config.OVERWRITE:
                raise FileExistsError(config.EXPORT_DIR)
            shutil.rmtree(config.EXPORT_DIR)

        metadata_table = pq.read_table(metadata_path)
        metadata = {
            sample_id: (
                split,
                class_name,
                repetition_class,
                token_len,
                description_code,
                description_image,
                description_image_code,
            )
            for (
                sample_id,
                split,
                class_name,
                repetition_class,
                token_len,
                description_code,
                description_image,
                description_image_code,
            ) in zip(
                metadata_table["sample_id"].to_pylist(),
                metadata_table["split"].to_pylist(),
                metadata_table["class"].to_pylist(),
                metadata_table["repetition_class"].to_pylist(),
                metadata_table["token_len"].to_pylist(),
                metadata_table["llm_description"].to_pylist(),
                metadata_table["llm_description_image"].to_pylist(),
                metadata_table["llm_description_image_code"].to_pylist(),
            )
        }

        data_dir = config.EXPORT_DIR / "data"
        writers = {
            split: ShardWriter(data_dir, split, FINAL_SCHEMA, config.MAX_PARQUET_BYTES)
            for split in ("benchmark", "train")
        }

        for dataset_name in ("datikz", "benchmark"):
            for row in iter_rows(dataset_name, ["sample_id", "reference_code", "source", "rendered_image"]):
                (
                    split,
                    class_name,
                    repetition_class,
                    token_len,
                    description_code,
                    description_image,
                    description_image_code,
                ) = metadata[row["sample_id"]]
                image = {"bytes": row["rendered_image"], "path": None}
                final_row = {
                    "input_image": image,
                    "reference_image": image,
                    "reference_code": row["reference_code"],
                    "llm_description": description_code,
                    "llm_description_image": description_image,
                    "llm_description_image_code": description_image_code,
                    "type": "normal",
                    "source": row["source"],
                    "class": class_name,
                    "repetition_class": repetition_class,
                    "token_len": token_len,
                }
                description_bytes = sum(
                    len((value or "").encode("utf-8"))
                    for value in (
                        description_code,
                        description_image,
                        description_image_code,
                    )
                )
                estimate = (
                    2 * len(row["rendered_image"])
                    + len(row["reference_code"].encode("utf-8"))
                    + description_bytes
                    + 512
                )
                writers[split].add(final_row, estimate)

        result = {split: writer.close() for split, writer in writers.items()}
        self._write_dataset_card(result)
        (config.EXPORT_DIR / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _write_dataset_card(result: dict) -> None:
        lines = "\n".join(f"- `{name}`: {value['rows']} rows" for name, value in result.items())
        text = f"""---
dataset_info:
  features:
  - name: input_image
    dtype: image
  - name: reference_image
    dtype: image
  - name: reference_code
    dtype: string
  - name: llm_description
    dtype: string
  - name: llm_description_image
    dtype: string
  - name: llm_description_image_code
    dtype: string
  - name: type
    dtype: string
  - name: source
    dtype: string
  - name: class
    dtype: string
  - name: repetition_class
    dtype: string
  - name: token_len
    dtype: int32
configs:
- config_name: default
  data_files:
  - split: benchmark
    path: data/benchmark-*.parquet
  - split: train
    path: data/train-*.parquet
---

# Cleaned TikZ dataset

{lines}
"""
        (config.EXPORT_DIR / "README.md").write_text(text, encoding="utf-8")


class HuggingFaceUploader:
    def run(self) -> dict:
        if config.HF_REPO_ID.startswith("your-user/"):
            raise ValueError("Set HF_REPO_ID in config.py")
        if not config.EXPORT_DIR.exists():
            raise FileNotFoundError(config.EXPORT_DIR)
        api = HfApi(token=config.HF_TOKEN)
        api.create_repo(
            repo_id=config.HF_REPO_ID,
            repo_type="dataset",
            private=config.HF_PRIVATE,
            exist_ok=True,
        )
        if hasattr(api, "upload_large_folder"):
            api.upload_large_folder(
                repo_id=config.HF_REPO_ID,
                repo_type="dataset",
                folder_path=str(config.EXPORT_DIR),
            )
        else:
            api.upload_folder(
                repo_id=config.HF_REPO_ID,
                repo_type="dataset",
                folder_path=str(config.EXPORT_DIR),
            )
        return {"repo_id": config.HF_REPO_ID}