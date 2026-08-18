"""Download the published dataset and add more Ollama descriptions."""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

import config

# Import the existing preprocessing client without importing the whole components package.
sys.path.insert(0, str(config.PREPROCESS_CODE_DIR / "components"))
from ollama_client import OllamaClient  # noqa: E402


Ref = tuple[Path, int]
Bucket = tuple[str, str, str, str]


def is_missing(value: object) -> bool:
    return value is None or not str(value).strip()


def target(repetition_class: str) -> int:
    return max(
        0,
        int(
            config.ADDITIONAL_DESCRIPTIONS_PER_REPETITION_CLASS.get(
                repetition_class,
                0,
            )
        ),
    )


def split_files(root: Path, split: str) -> list[Path]:
    files = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} Parquet files found in {root / 'data'}")
    return files


def download_dataset() -> Path:
    config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.HF_REPO_ID,
        repo_type="dataset",
        revision=config.HF_REVISION,
        local_dir=config.DOWNLOAD_DIR,
        token=config.HF_TOKEN,
        allow_patterns=["data/*.parquet", "README.md", "manifest.json"],
    )
    return config.DOWNLOAD_DIR


def prepare_output(source: Path) -> Path:
    if config.OUTPUT_DIR.exists():
        if not config.OVERWRITE_OUTPUT:
            raise FileExistsError(config.OUTPUT_DIR)
        shutil.rmtree(config.OUTPUT_DIR)

    shutil.copytree(
        source,
        config.OUTPUT_DIR,
        ignore=shutil.ignore_patterns(".cache"),
    )
    return config.OUTPUT_DIR


def collect_candidates(root: Path) -> dict[Bucket, list[Ref]]:
    enabled = tuple(dict.fromkeys(config.DESCRIPTION_TYPES))
    invalid = set(enabled).difference(config.DESCRIPTION_COLUMNS)
    if invalid:
        raise ValueError(f"Unknown DESCRIPTION_TYPES: {sorted(invalid)}")

    candidates: dict[Bucket, list[Ref]] = defaultdict(list)

    for split in config.SPLITS:
        for path in tqdm(split_files(root, split), desc=f"Scan {split}", unit="file"):
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema.names)
            required = {"class", "repetition_class"}
            missing = required.difference(names)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")

            existing_description_columns = [
                config.DESCRIPTION_COLUMNS[name]
                for name in enabled
                if config.DESCRIPTION_COLUMNS[name] in names
            ]
            columns = ["class", "repetition_class", *existing_description_columns]
            offset = 0

            for batch in parquet.iter_batches(columns=columns, batch_size=2_048):
                data = batch.to_pydict()
                size = batch.num_rows

                for i in range(size):
                    class_name = str(data["class"][i])
                    repetition_class = str(data["repetition_class"][i])
                    wanted = target(repetition_class)
                    if wanted <= 0:
                        continue

                    for description_type in enabled:
                        column = config.DESCRIPTION_COLUMNS[description_type]
                        value = data[column][i] if column in data else None
                        if not is_missing(value):
                            continue

                        bucket = (
                            split,
                            class_name,
                            repetition_class,
                            description_type,
                        )
                        limit = wanted * max(1, config.DESCRIPTION_CANDIDATE_FACTOR)
                        if len(candidates[bucket]) < limit:
                            candidates[bucket].append((path, offset + i))

                offset += size

    return candidates


def load_candidates(refs: list[Ref], description_type: str) -> list[tuple[Ref, str, bytes | None]]:
    by_file: dict[Path, list[int]] = defaultdict(list)
    for path, index in refs:
        by_file[path].append(index)

    loaded: dict[Ref, tuple[str, bytes | None]] = {}
    columns = ["reference_code"]
    if description_type in {"image", "image_code"}:
        columns.append("input_image")

    for path, indices in by_file.items():
        table = pq.read_table(path, columns=columns)
        codes = table["reference_code"]
        images = table["input_image"] if "input_image" in columns else None

        for index in indices:
            code = str(codes[index].as_py() or "")
            image_bytes = None
            if images is not None:
                image = images[index].as_py()
                if isinstance(image, dict):
                    image_bytes = image.get("bytes")
                elif isinstance(image, (bytes, bytearray)):
                    image_bytes = bytes(image)
            loaded[(path, index)] = (code, image_bytes)

    return [(ref, *loaded[ref]) for ref in refs]


def generate_descriptions(
    candidates: dict[Bucket, list[Ref]],
) -> tuple[dict[Path, dict[str, dict[int, str]]], dict]:
    client = OllamaClient()
    workers = max(1, int(config.OLLAMA_PARALLEL_REQUESTS))
    updates: dict[Path, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    failures: list[dict] = []
    generated = defaultdict(int)

    total = sum(min(target(bucket[2]), len(refs)) for bucket, refs in candidates.items())
    progress = tqdm(total=total, desc="Generate descriptions", unit="description")

    def request(description_type: str, code: str, image: bytes | None) -> str:
        return client.describe(description_type, code=code, image=image)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for bucket, refs in sorted(candidates.items()):
                split, class_name, repetition_class, description_type = bucket
                wanted = min(target(repetition_class), len(refs))
                if wanted <= 0:
                    continue

                rows = load_candidates(refs, description_type)
                next_index = 0
                success = 0
                pending = {}

                def submit_one() -> bool:
                    nonlocal next_index
                    if next_index >= len(rows):
                        return False
                    ref, code, image = rows[next_index]
                    next_index += 1
                    future = executor.submit(request, description_type, code, image)
                    pending[future] = ref
                    return True

                while len(pending) < min(workers, wanted) and submit_one():
                    pass

                while pending and success < wanted:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        path, row_index = pending.pop(future)
                        try:
                            text = future.result()
                            column = config.DESCRIPTION_COLUMNS[description_type]
                            relative = path.relative_to(config.DOWNLOAD_DIR)
                            updates[relative][column][row_index] = text
                            generated[description_type] += 1
                            success += 1
                            progress.update(1)
                        except Exception as error:
                            failures.append({
                                "split": split,
                                "class": class_name,
                                "repetition_class": repetition_class,
                                "description_type": description_type,
                                "file": str(path),
                                "row": row_index,
                                "reason": repr(error),
                            })

                        if success < wanted:
                            submit_one()

                        progress.set_postfix(
                            split=split,
                            class_name=class_name,
                            repetition=repetition_class,
                            type=description_type,
                            workers=workers,
                            failures=len(failures),
                        )
    finally:
        progress.close()

    return updates, {
        "generated": dict(generated),
        "total": sum(generated.values()),
        "failures": failures,
        "parallel_requests": workers,
    }


def apply_updates(output_root: Path, updates: dict[Path, dict[str, dict[int, str]]]) -> None:
    for relative, columns in tqdm(sorted(updates.items()), desc="Write Parquet", unit="file"):
        path = output_root / relative
        table = pq.read_table(path)

        for column, values_by_index in columns.items():
            if column in table.column_names:
                values = table[column].to_pylist()
                field = table.schema.field(column)
                index = table.schema.get_field_index(column)
            else:
                values = [None] * table.num_rows
                field = pa.field(column, pa.string())
                index = len(table.column_names)

            for row_index, text in values_by_index.items():
                values[row_index] = text

            array = pa.array(values, type=pa.string())
            if column in table.column_names:
                table = table.set_column(index, field, array)
            else:
                table = table.append_column(field, array)

        temporary = path.with_suffix(".tmp.parquet")
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=3,
        )
        temporary.replace(path)


def upload_dataset(output_root: Path) -> None:
    if not config.UPLOAD:
        return
    api = HfApi(token=config.HF_TOKEN)
    api.create_repo(
        repo_id=config.UPLOAD_REPO_ID,
        repo_type="dataset",
        private=config.HF_PRIVATE,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=config.UPLOAD_REPO_ID,
        repo_type="dataset",
        folder_path=str(output_root),
    )


def main() -> None:
    source = download_dataset()
    output = prepare_output(source)
    candidates = collect_candidates(source)
    updates, stats = generate_descriptions(candidates)
    apply_updates(output, updates)

    report = {
        "repo_id": config.HF_REPO_ID,
        "description_types": list(config.DESCRIPTION_TYPES),
        "additional_targets_per_repetition_class": dict(
            config.ADDITIONAL_DESCRIPTIONS_PER_REPETITION_CLASS
        ),
        **stats,
    }
    (output / "modification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    upload_dataset(output)

    print(f"Modified dataset: {output}")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
