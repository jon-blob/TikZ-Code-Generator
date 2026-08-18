from __future__ import annotations

import csv
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from config import Config, IMPORTANCE_WEIGHTS


BASE_COLUMNS = [
    "input_image",
    "reference_image",
    "reference_code",
    "class",
    "repetition_class",
    "token_len",
    "source",
    "type",
]

DESCRIPTION_COLUMNS = {
    "llm_description",
    "llm_description_image",
    "llm_description_image_code",
}

IMPORTANCE_ALIASES = {
    "not_important": "not important",
    "a_bit_important": "a bit important",
    "really_important": "really important",
    "extremely_important": "extremely important",
    "extremly important": "extremely important",
    "extremly_important": "extremely important",
}


@dataclass(frozen=True, slots=True)
class Sample:
    file: int
    row: int
    class_name: str
    repetition_class: str
    described: bool

    @property
    def key(self) -> tuple[int, int]:
        return self.file, self.row


def has_text(value: object) -> bool:
    return value is not None and bool(str(value).strip())


def active_description_mix(
    mix: Mapping[str, float],
) -> dict[str, float]:
    unknown = set(mix) - DESCRIPTION_COLUMNS
    if unknown:
        raise ValueError(
            "Unknown description columns: " + ", ".join(sorted(unknown))
        )

    active = {
        column: float(weight)
        for column, weight in mix.items()
        if float(weight) > 0
    }

    if not active:
        raise ValueError("description_mix must contain at least one positive weight")

    if any(float(weight) < 0 for weight in mix.values()):
        raise ValueError("description_mix weights must be >= 0")

    return active


def normalize_importance(value: str) -> str:
    normalized = " ".join(
        str(value).strip().lower().replace("-", " ").replace("_", " ").split()
    )
    normalized = IMPORTANCE_ALIASES.get(normalized, normalized)

    if normalized not in IMPORTANCE_WEIGHTS:
        raise ValueError(
            f"Unknown class importance {value!r}. Valid levels: "
            + ", ".join(IMPORTANCE_WEIGHTS)
        )

    return normalized


def validate_class_importance(mapping: Mapping[str, str]) -> None:
    for class_name, level in mapping.items():
        if not str(class_name).strip():
            raise ValueError("train_class_importance contains an empty class name")
        normalize_importance(level)


def class_importance_level(
    class_name: str,
    mapping: Mapping[str, str],
) -> str:
    direct = {
        str(name).casefold(): level
        for name, level in mapping.items()
        if name != "*"
    }
    level = direct.get(class_name.casefold(), mapping.get("*", "none"))
    return normalize_importance(level)


def class_importance_weight(
    class_name: str,
    mapping: Mapping[str, str],
) -> float:
    return IMPORTANCE_WEIGHTS[class_importance_level(class_name, mapping)]


def valid_repetitions(values: Sequence[str]) -> set[str]:
    valid = {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }
    if not valid:
        raise ValueError("valid_repetition_classes must not be empty")
    return valid


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


def scan(
    files: list[Path],
    description_mix: Mapping[str, float],
) -> list[Sample]:
    samples: list[Sample] = []
    description_columns = list(active_description_mix(description_mix))
    required = set(BASE_COLUMNS) | set(description_columns)

    for file_index, path in enumerate(tqdm(files, desc="Scan metadata")):
        parquet = pq.ParquetFile(path)
        missing = required - set(parquet.schema_arrow.names)

        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        offset = 0
        scan_columns = ["class", "repetition_class", *description_columns]

        for batch in parquet.iter_batches(
            columns=scan_columns,
            batch_size=65_536,
        ):
            rows = batch.to_pylist()

            for index, row in enumerate(rows):
                described = any(
                    has_text(row[column])
                    for column in description_columns
                )
                samples.append(
                    Sample(
                        file=file_index,
                        row=offset + index,
                        class_name=str(row["class"]),
                        repetition_class=str(row["repetition_class"]).lower(),
                        described=described,
                    )
                )

            offset += batch.num_rows

    return samples


def is_noise(sample: Sample, noise_class: str) -> bool:
    return sample.class_name.casefold() == noise_class.casefold()


def repetition_candidates(
    samples: Iterable[Sample],
    valid_repetition_classes: Sequence[str] | None,
) -> list[Sample]:
    # None intentionally means no repetition filter. This is used by
    # CrystalBLEU so low/medium/high/very_high/critical are all eligible.
    if valid_repetition_classes is None:
        return list(samples)

    valid = valid_repetitions(valid_repetition_classes)
    return [
        sample
        for sample in samples
        if sample.repetition_class in valid
    ]


def ordered_rows(
    rows: Iterable[Sample],
    prefer_described: bool | None,
    rng: random.Random,
) -> list[Sample]:
    rows = list(rows)

    if prefer_described is None:
        rng.shuffle(rows)
        return rows

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
    return preferred + fallback


def _observed_class_names(
    samples: Iterable[Sample],
    noise_class: str,
) -> list[str]:
    names = {
        sample.class_name
        for sample in samples
        if not is_noise(sample, noise_class)
    }
    return sorted(names)


def train_class_names(
    samples: Iterable[Sample],
    class_importance: Mapping[str, str],
    noise_class: str,
) -> list[str]:
    validate_class_importance(class_importance)
    observed = _observed_class_names(samples, noise_class)

    names = {
        name
        for name in observed
        if class_importance_weight(name, class_importance) > 0
    }

    # Explicitly configured classes are kept even when a split has zero rows.
    # This matters for weighted targets and makes split behavior predictable.
    for name, level in class_importance.items():
        if name == "*" or name.casefold() == noise_class.casefold():
            continue
        if IMPORTANCE_WEIGHTS[normalize_importance(level)] > 0:
            names.add(str(name))

    return sorted(names)


def benchmark_class_names(
    samples: Iterable[Sample],
    requested: Sequence[str] | None,
    noise_class: str,
) -> list[str]:
    if requested is None:
        return _observed_class_names(samples, noise_class)

    result: list[str] = []
    seen: set[str] = set()
    for name in requested:
        text = str(name).strip()
        if not text or text.casefold() == noise_class.casefold():
            continue
        folded = text.casefold()
        if folded not in seen:
            result.append(text)
            seen.add(folded)
    return result


def group_selected_classes(
    samples: Iterable[Sample],
    class_names: Sequence[str],
) -> dict[str, list[Sample]]:
    lookup = {name.casefold(): name for name in class_names}
    groups = {name: [] for name in class_names}

    for sample in samples:
        configured_name = lookup.get(sample.class_name.casefold())
        if configured_name is not None:
            groups[configured_name].append(sample)

    return groups


def choose_per_class(
    samples: Iterable[Sample],
    class_names: Sequence[str],
    count: int,
    prefer_described: bool,
    rng: random.Random,
    valid_repetition_classes: Sequence[str],
) -> set[tuple[int, int]]:
    """Choose up to count real samples from every requested class."""
    if count <= 0:
        return set()

    eligible = repetition_candidates(samples, valid_repetition_classes)
    groups = group_selected_classes(eligible, class_names)
    selected: set[tuple[int, int]] = set()

    for class_name in class_names:
        rows = ordered_rows(groups[class_name], prefer_described, rng)
        selected.update(sample.key for sample in rows[:count])

    return selected


def _weighted_quota(
    weights: Mapping[str, float],
    total: int,
    rng: random.Random,
) -> dict[str, int]:
    positive = {
        name: float(weight)
        for name, weight in weights.items()
        if float(weight) > 0
    }
    counts = {name: 0 for name in weights}

    if total <= 0 or not positive:
        return counts

    weight_sum = sum(positive.values())
    raw = {
        name: total * weight / weight_sum
        for name, weight in positive.items()
    }
    for name, value in raw.items():
        counts[name] = math.floor(value)

    remaining = total - sum(counts.values())
    order = list(positive)
    rng.shuffle(order)
    order.sort(key=lambda name: raw[name] - counts[name], reverse=True)

    for name in order[:remaining]:
        counts[name] += 1

    # If total is large enough, keep every explicitly active class represented.
    zero_names = [name for name in positive if counts[name] == 0]
    if total >= len(positive):
        for name in zero_names:
            donors = [
                donor
                for donor in positive
                if counts[donor] > 1
            ]
            if not donors:
                break
            donor = max(
                donors,
                key=lambda item: (counts[item], positive[item]),
            )
            counts[donor] -= 1
            counts[name] += 1

    return counts


def _allocate_extra(
    counts: dict[str, int],
    capacities: Mapping[str, int],
    weights: Mapping[str, float],
    amount: int,
    minimum_weight: float,
    rng: random.Random,
) -> int:
    """Move shortage only to equally/more important classes."""
    remaining = amount

    while remaining > 0:
        candidates = {
            name: weights[name]
            for name in weights
            if (
                weights[name] >= minimum_weight
                and counts[name] < capacities[name]
            )
        }
        if not candidates:
            break

        free = {
            name: capacities[name] - counts[name]
            for name in candidates
        }
        batch_total = min(remaining, sum(free.values()))
        proposal = _weighted_quota(candidates, batch_total, rng)

        allocated = 0
        overflow = 0
        for name, proposed in proposal.items():
            take = min(proposed, free.get(name, 0))
            counts[name] += take
            allocated += take
            overflow += proposed - take

        # If rounding/capacity left something unallocated, fill one by one
        # from the most important classes that still have capacity.
        still_needed = batch_total - allocated
        while still_needed > 0:
            available = [
                name
                for name in candidates
                if counts[name] < capacities[name]
            ]
            if not available:
                break
            rng.shuffle(available)
            available.sort(key=lambda name: weights[name], reverse=True)
            for name in available:
                counts[name] += 1
                allocated += 1
                still_needed -= 1
                if still_needed == 0:
                    break

        if allocated == 0:
            break
        remaining -= allocated

    return remaining


def weighted_train_counts(
    groups: Mapping[str, list[Sample]],
    class_importance: Mapping[str, str],
    total: int,
    rng: random.Random,
) -> tuple[dict[str, int], int]:
    """
    Allocate a fixed target according to importance weights.

    A shortage in an important class is never moved down into a less important
    class. It is moved only to an equally/more important class; anything still
    missing is returned for random noise fallback.
    """
    weights = {
        name: class_importance_weight(name, class_importance)
        for name in groups
    }
    capacities = {name: len(rows) for name, rows in groups.items()}
    quotas = _weighted_quota(weights, total, rng)
    counts = {
        name: min(quotas[name], capacities[name])
        for name in groups
    }

    shortages = sorted(
        (
            (weights[name], quotas[name] - counts[name])
            for name in groups
            if quotas[name] > counts[name]
        ),
        reverse=True,
    )

    unresolved = 0
    for source_weight, missing in shortages:
        if missing <= 0:
            continue
        unresolved += _allocate_extra(
            counts=counts,
            capacities=capacities,
            weights=weights,
            amount=missing,
            minimum_weight=source_weight,
            rng=rng,
        )

    # _allocate_extra returns only the unresolved part of each shortage. Since
    # successful reallocations directly increment counts, the total deficit is
    # most robustly calculated from the requested total at the end.
    unresolved = max(0, total - sum(counts.values()))
    return counts, unresolved


def choose_uniform_with_noise_fallback(
    samples: Iterable[Sample],
    total: int,
    prefer_described: bool | None,
    rng: random.Random,
    valid_repetition_classes: Sequence[str] | None,
    noise_class: str,
    label: str,
) -> set[tuple[int, int]]:
    """
    Select up to ``total`` samples uniformly across all observed real classes.

    Small classes are exhausted first and their unused quota is redistributed
    among the remaining real classes. Noise is used only when all eligible
    non-noise rows together are insufficient.
    """
    if total <= 0:
        return set()

    eligible = repetition_candidates(samples, valid_repetition_classes)
    class_names = _observed_class_names(eligible, noise_class)
    groups = group_selected_classes(eligible, class_names)
    noise = [sample for sample in eligible if is_noise(sample, noise_class)]

    # Best-effort equal allocation with capacity-aware redistribution.
    counts = {name: 0 for name in class_names}
    remaining = min(total, sum(len(rows) for rows in groups.values()))

    active = [name for name in class_names if groups[name]]
    while remaining > 0 and active:
        rng.shuffle(active)
        share = max(1, remaining // len(active))
        progressed = 0
        next_active: list[str] = []

        for name in active:
            free = len(groups[name]) - counts[name]
            if free <= 0:
                continue
            take = min(share, free, remaining)
            counts[name] += take
            remaining -= take
            progressed += take
            if counts[name] < len(groups[name]):
                next_active.append(name)
            if remaining == 0:
                break

        if progressed == 0:
            break
        active = next_active

    selected: set[tuple[int, int]] = set()
    for class_name, count in counts.items():
        rows = ordered_rows(groups[class_name], prefer_described, rng)
        selected.update(sample.key for sample in rows[:count])

    missing = total - len(selected)
    if missing > len(noise):
        raise ValueError(
            f"Requested {label} size: {total:,}; remaining real classes provide "
            f"{len(selected):,} samples and only {len(noise):,} eligible noise "
            f"samples are available for the remaining {missing:,}."
        )

    if missing > 0:
        rng.shuffle(noise)
        selected.update(sample.key for sample in noise[:missing])

    return selected


def choose_weighted_with_noise_fallback(
    samples: Iterable[Sample],
    total: int,
    class_importance: Mapping[str, str],
    prefer_described: bool | None,
    rng: random.Random,
    valid_repetition_classes: Sequence[str],
    noise_class: str,
    label: str,
) -> set[tuple[int, int]]:
    if total <= 0:
        return set()

    eligible = repetition_candidates(samples, valid_repetition_classes)
    class_names = train_class_names(
        eligible,
        class_importance,
        noise_class,
    )
    groups = group_selected_classes(eligible, class_names)
    noise = [sample for sample in eligible if is_noise(sample, noise_class)]

    counts, missing = weighted_train_counts(
        groups=groups,
        class_importance=class_importance,
        total=total,
        rng=rng,
    )

    selected: set[tuple[int, int]] = set()
    for class_name, count in counts.items():
        rows = ordered_rows(groups[class_name], prefer_described, rng)
        selected.update(sample.key for sample in rows[:count])

    missing = total - len(selected)
    if missing > len(noise):
        raise ValueError(
            f"Requested {label} size: {total:,}; weighted real classes provide "
            f"{len(selected):,} samples and only {len(noise):,} eligible noise "
            f"samples are available for the remaining {missing:,}."
        )

    if missing > 0:
        # Noise fallback is deliberately random and does not use description
        # preference or class balancing.
        rng.shuffle(noise)
        selected.update(sample.key for sample in noise[:missing])

    return selected


def choose_benchmark_with_noise_fallback(
    samples: Iterable[Sample],
    class_names: Sequence[str],
    count_per_class: int,
    noise_samples: int,
    prefer_described: bool,
    rng: random.Random,
    valid_repetition_classes: Sequence[str],
    noise_class: str,
) -> set[tuple[int, int]]:
    """
    Select a fixed target for each benchmark class plus an explicit noise class.

    Real samples are preferred and description-bearing real samples come first.
    Missing slots of real benchmark classes are filled from the same random,
    non-reused noise pool. ``noise_samples`` then adds an explicit additional
    noise quota on top of those fallback rows.
    """
    if count_per_class <= 0 and noise_samples <= 0:
        return set()

    eligible = repetition_candidates(samples, valid_repetition_classes)
    groups = group_selected_classes(eligible, class_names)
    noise = [sample for sample in eligible if is_noise(sample, noise_class)]

    selected: set[tuple[int, int]] = set()
    total_missing = 0

    if count_per_class > 0:
        for class_name in class_names:
            rows = ordered_rows(groups[class_name], prefer_described, rng)
            chosen = rows[:count_per_class]
            selected.update(sample.key for sample in chosen)
            missing = count_per_class - len(chosen)
            total_missing += missing
            print(
                f"Benchmark {class_name}: {len(chosen):,}/{count_per_class:,} "
                f"real, noise fallback={missing:,}"
            )

    required_noise = total_missing + noise_samples
    if required_noise > len(noise):
        raise ValueError(
            f"Benchmark needs {required_noise:,} noise samples "
            f"({total_missing:,} fallback + {noise_samples:,} explicit) but only "
            f"{len(noise):,} eligible noise samples are available."
        )

    rng.shuffle(noise)
    selected.update(sample.key for sample in noise[:required_noise])

    if noise_samples > 0:
        print(
            f"Benchmark {noise_class}: {noise_samples:,} explicit samples "
            f"(+ {total_missing:,} used as class fallback)"
        )

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


def choose_description(
    row: dict,
    mix: Mapping[str, float],
    rng: random.Random,
) -> tuple[str, str]:
    """Randomly choose one configured description that exists for this row."""
    active = active_description_mix(mix)
    available = [
        (column, weight)
        for column, weight in active.items()
        if has_text(row.get(column))
    ]

    if not available:
        return "", ""

    columns = [column for column, _ in available]
    weights = [weight for _, weight in available]
    chosen = rng.choices(columns, weights=weights, k=1)[0]
    return str(row[chosen]).strip(), chosen


def export_split(
    files: list[Path],
    samples: list[Sample],
    output: Path,
    batch_size: int,
    description_mix: Mapping[str, float],
    seed: int,
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

    active_mix = active_description_mix(description_mix)
    names = names_for(samples)
    input_columns = [*BASE_COLUMNS, *active_mix]
    fields = [
        "sample_id",
        "input_image",
        "reference_image",
        "llm_description",
        "llm_description_source",
        "reference_code",
        "class",
        "repetition_class",
        "token_len",
        "source",
        "type",
    ]
    rng = random.Random(seed)
    description_counts: Counter[str] = Counter()

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
            columns=input_columns,
            batch_size=batch_size,
        )

        for key, row in tqdm(
            iterator,
            total=len(names),
            desc=f"Export {output.name}",
        ):
            name = names[key]

            input_path = folders["input_image"] / f"{name}.png"
            reference_path = folders["reference_image"] / f"{name}.png"
            description_path = folders["llm_description"] / f"{name}.txt"
            code_path = folders["reference_code"] / f"{name}.txt"

            description, description_source = choose_description(
                row=row,
                mix=active_mix,
                rng=rng,
            )
            description_counts[description_source or "missing"] += 1

            to_image(row["input_image"]).save(input_path, "PNG")
            to_image(row["reference_image"]).save(reference_path, "PNG")
            description_path.write_text(description, encoding="utf-8")
            code_path.write_text(
                str(row["reference_code"] or ""),
                encoding="utf-8",
            )

            writer.writerow(
                {
                    "sample_id": name,
                    "input_image": str(input_path.resolve()),
                    "reference_image": str(reference_path.resolve()),
                    "llm_description": str(description_path.resolve()),
                    "llm_description_source": description_source,
                    "reference_code": str(code_path.resolve()),
                    "class": row["class"],
                    "repetition_class": row["repetition_class"],
                    "token_len": row["token_len"],
                    "source": row["source"],
                    "type": row["type"],
                }
            )

    if names:
        summary = ", ".join(
            f"{name}={count:,}"
            for name, count in sorted(description_counts.items())
        )
        print(f"Description mix for {output}: {summary}")


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
