from __future__ import annotations

import csv
import math
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


COLUMNS = [
    "input_image",
    "reference_image",
    "llm_description",
    "reference_code",
    "class",
    "source",
    "type",
]


@dataclass(frozen=True, slots=True)
class Config:
    input_dir: Path
    output_dir: Path
    benchmark_samples_per_class: int = 20
    val_samples_per_class: int = 20
    train_crystalbleu_size: int = 50_000
    train_size: int | None = None
    balance_tolerance: float = 0.10
    seed: int = 42
    overwrite: bool = False
    batch_size: int = 2_048


@dataclass(frozen=True, slots=True)
class Sample:
    file: int
    row: int
    class_name: str
    described: bool

    @property
    def key(self) -> tuple[int, int]:
        return self.file, self.row


def has_text(value: object) -> bool:
    return value is not None and bool(str(value).strip())


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "class"


def find_files(input_dir: Path, split: str) -> list[Path]:
    split_dir = input_dir / split

    if split_dir.is_dir():
        files = sorted(split_dir.rglob("*.parquet"))
    else:
        all_files = sorted(input_dir.rglob("*.parquet"))
        files = [
            path
            for path in all_files
            if split in str(path.relative_to(input_dir)).lower()
        ]
        files = files or all_files

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")

    return files


def scan(files: list[Path]) -> list[Sample]:
    samples: list[Sample] = []

    for file_index, path in enumerate(tqdm(files, desc="Scan metadata")):
        parquet = pq.ParquetFile(path)
        missing = set(COLUMNS) - set(parquet.schema_arrow.names)

        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        offset = 0

        for batch in parquet.iter_batches(
            columns=["class", "llm_description"],
            batch_size=65_536,
        ):
            classes = batch.column(0).to_pylist()
            descriptions = batch.column(1).to_pylist()

            for index, (class_name, description) in enumerate(
                zip(classes, descriptions, strict=True)
            ):
                samples.append(
                    Sample(
                        file=file_index,
                        row=offset + index,
                        class_name=str(class_name),
                        described=has_text(description),
                    )
                )

            offset += batch.num_rows

    return samples


def group_by_class(
    samples: Iterable[Sample],
) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)

    for sample in samples:
        groups[sample.class_name].append(sample)

    return dict(groups)


def choose_per_class(
    samples: Iterable[Sample],
    count: int,
    prefer_described: bool,
    rng: random.Random,
) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()

    for class_name, rows in sorted(group_by_class(samples).items()):
        preferred = [
            sample
            for sample in rows
            if sample.described == prefer_described
        ]
        fallback = [
            sample
            for sample in rows
            if sample.described != prefer_described
        ]

        rng.shuffle(preferred)
        rng.shuffle(fallback)
        chosen = (preferred + fallback)[:count]

        if len(chosen) < count:
            raise ValueError(
                f"{class_name}: requested {count}, available {len(rows)}"
            )

        selected.update(sample.key for sample in chosen)

    return selected


def balanced_counts(
    groups: dict[str, list[Sample]],
    total: int,
    tolerance: float,
    label: str = "selection",
) -> dict[str, int]:
    if not groups:
        raise ValueError("No classes available for balanced selection")

    available = {
        class_name: len(rows)
        for class_name, rows in groups.items()
    }
    smallest = min(available.values())
    maximum = math.floor(smallest * (1.0 + tolerance))
    capacities = {
        class_name: min(count, maximum)
        for class_name, count in available.items()
    }

    maximum_total = sum(capacities.values())

    if total > maximum_total:
        raise ValueError(
            f"Requested {label} size: {total:,}; "
            f"maximum balanced size: {maximum_total:,}"
        )

    counts = {
        class_name: 0
        for class_name in sorted(groups)
    }

    while total > 0:
        changed = False

        for class_name in sorted(
            counts,
            key=lambda name: (counts[name], name),
        ):
            if counts[class_name] >= capacities[class_name]:
                continue

            counts[class_name] += 1
            total -= 1
            changed = True

            if total == 0:
                break

        if not changed:
            raise RuntimeError("Could not allocate balanced sample counts")

    return counts


def choose_counts(
    groups: dict[str, list[Sample]],
    counts: dict[str, int],
    prefer_described: bool | None,
    rng: random.Random,
) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()

    for class_name, count in counts.items():
        rows = list(groups[class_name])

        if prefer_described is None:
            rng.shuffle(rows)
        else:
            preferred = [
                sample
                for sample in rows
                if sample.described == prefer_described
            ]
            fallback = [
                sample
                for sample in rows
                if sample.described != prefer_described
            ]

            rng.shuffle(preferred)
            rng.shuffle(fallback)
            rows = preferred + fallback

        selected.update(sample.key for sample in rows[:count])

    return selected


def without(
    samples: Iterable[Sample],
    keys: set[tuple[int, int]],
) -> list[Sample]:
    return [
        sample
        for sample in samples
        if sample.key not in keys
    ]


def balance_remainder(
    samples: list[Sample],
    tolerance: float,
    rng: random.Random,
) -> set[tuple[int, int]]:
    groups = group_by_class(samples)

    if not groups:
        return set()

    smallest = min(len(rows) for rows in groups.values())
    maximum = math.floor(smallest * (1.0 + tolerance))
    counts = {
        class_name: min(len(rows), maximum)
        for class_name, rows in groups.items()
    }

    return choose_counts(
        groups=groups,
        counts=counts,
        prefer_described=None,
        rng=rng,
    )


def rows_at(
    files: list[Path],
    keys: set[tuple[int, int]],
    columns: list[str],
    batch_size: int,
) -> Iterator[tuple[tuple[int, int], dict]]:
    per_file: dict[int, list[int]] = defaultdict(list)

    for file_index, row_index in keys:
        per_file[file_index].append(row_index)

    for file_index, wanted in sorted(per_file.items()):
        wanted.sort()
        pointer = 0
        offset = 0

        for batch in pq.ParquetFile(files[file_index]).iter_batches(
            columns=columns,
            batch_size=batch_size,
        ):
            end = offset + batch.num_rows
            positions: list[int] = []

            while pointer < len(wanted) and wanted[pointer] < end:
                positions.append(wanted[pointer] - offset)
                pointer += 1

            if positions:
                selected = batch.take(pa.array(positions))

                for position, row in zip(
                    positions,
                    selected.to_pylist(),
                    strict=True,
                ):
                    yield (file_index, offset + position), row

            offset = end

            if pointer == len(wanted):
                break


def to_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")

        path = value.get("path")

        if path:
            return Image.open(path).convert("RGB")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")

    return Image.open(value).convert("RGB")


def names_for(
    samples: Iterable[Sample],
) -> dict[tuple[int, int], str]:
    counters: dict[str, int] = defaultdict(int)
    names: dict[tuple[int, int], str] = {}

    for sample in sorted(
        samples,
        key=lambda item: (
            item.class_name,
            item.file,
            item.row,
        ),
    ):
        counters[sample.class_name] += 1
        names[sample.key] = (
            f"{slug(sample.class_name)}_"
            f"{counters[sample.class_name]:08d}"
        )

    return names


def export_split(
    files: list[Path],
    samples: list[Sample],
    output: Path,
    batch_size: int,
) -> None:
    folders = {
        name: output / name
        for name in (
            "input_image",
            "reference_image",
            "llm_description",
            "reference_code",
        )
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    names = names_for(samples)
    fields = ["sample_id", *COLUMNS]

    manifest_path = output / "manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        iterator = rows_at(
            files=files,
            keys=set(names),
            columns=COLUMNS,
            batch_size=batch_size,
        )

        for key, row in tqdm(
            iterator,
            total=len(names),
            desc=f"Export {output.name}",
        ):
            name = names[key]

            input_path = folders["input_image"] / f"{name}.png"
            reference_path = (
                folders["reference_image"] / f"{name}.png"
            )
            description_path = (
                folders["llm_description"] / f"{name}.txt"
            )
            code_path = (
                folders["reference_code"] / f"{name}.txt"
            )

            to_image(row["input_image"]).save(input_path, "PNG")
            to_image(row["reference_image"]).save(
                reference_path,
                "PNG",
            )
            description_path.write_text(
                str(row["llm_description"] or ""),
                encoding="utf-8",
            )
            code_path.write_text(
                str(row["reference_code"] or ""),
                encoding="utf-8",
            )

            writer.writerow(
                {
                    "sample_id": name,
                    "input_image": str(input_path.resolve()),
                    "reference_image": str(reference_path.resolve()),
                    "llm_description": str(
                        description_path.resolve()
                    ),
                    "reference_code": str(code_path.resolve()),
                    "class": row["class"],
                    "source": row["source"],
                    "type": row["type"],
                }
            )


def export_crystalbleu(
    files: list[Path],
    samples: list[Sample],
    output: Path,
    batch_size: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = names_for(samples)

    iterator = rows_at(
        files=files,
        keys=set(names),
        columns=["reference_code"],
        batch_size=batch_size,
    )

    for key, row in tqdm(
        iterator,
        total=len(names),
        desc=f"Export {output.name}",
    ):
        path = output / f"{names[key]}.txt"
        path.write_text(
            str(row["reference_code"] or ""),
            encoding="utf-8",
        )


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists")

        shutil.rmtree(path)

    path.mkdir(parents=True)