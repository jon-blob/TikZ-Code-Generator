"""Load, shuffle, filter and deduplicate the two input datasets."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import config


@dataclass(frozen=True)
class Source:
    name: str
    directory: Path
    columns: dict[str, str | None]

    @property
    def required_columns(self) -> set[str]:
        return {str(value) for value in self.columns.values() if value}


DATIKZ = Source("datikz", config.DATIKZ_DIR, config.DATIKZ_COLUMNS)
BENCHMARK = Source("benchmark", config.BENCHMARK_DIR, config.BENCHMARK_COLUMNS)


def normalize_code(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"^```(?:latex|tex|tikz)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def code_hash(value: Any) -> str:
    code = normalize_code(value)
    return hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""


class InputDatasetCleaner:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.TOKENIZER_PATH),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )

    def run(self) -> tuple[dict[str, Dataset], dict]:
        datikz = self._load(DATIKZ, config.SEED)
        benchmark = self._load(BENCHMARK, config.SEED + 1)
        input_rows = {"datikz": len(datikz), "benchmark": len(benchmark)}

        benchmark, date_stats = self._filter_dates(benchmark)
        datikz, benchmark, duplicate_stats = self._deduplicate(datikz, benchmark)
        datikz, datikz_token_stats = self._filter_tokens(datikz, DATIKZ)
        benchmark, benchmark_token_stats = self._filter_tokens(benchmark, BENCHMARK)

        stats = {
            "input": input_rows,
            "benchmark_date_filter": date_stats,
            "duplicates": duplicate_stats,
            "token_filter": {
                "datikz": datikz_token_stats,
                "benchmark": benchmark_token_stats,
            },
        }
        return {"datikz": datikz, "benchmark": benchmark}, stats

    @staticmethod
    def _load(source: Source, seed: int) -> Dataset:
        files = sorted(source.directory.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files found in {source.directory}")

        dataset = load_dataset(
            "parquet",
            data_files=[str(path) for path in files],
            split="train",
            cache_dir=str(config.CACHE_DIR / "datasets"),
        )
        missing = source.required_columns.difference(dataset.column_names)
        if missing:
            raise KeyError(f"Missing {source.name} columns: {sorted(missing)}")
        return dataset.select_columns(sorted(source.required_columns)).shuffle(seed=seed)

    @staticmethod
    def _filter_dates(dataset: Dataset) -> tuple[Dataset, dict]:
        date_column = str(BENCHMARK.columns["date"])
        source_column = str(BENCHMARK.columns["source"])
        dates = pd.to_datetime(dataset[date_column], errors="coerce", utc=True)
        origins = [str(value or "").lower() for value in dataset[source_column]]
        allowed = {value.lower() for value in config.BENCHMARK_ORIGINS}

        keep: list[int] = []
        invalid = outside = wrong_origin = 0
        for index, (timestamp, origin) in enumerate(zip(dates, origins)):
            if pd.isna(timestamp):
                invalid += 1
            elif allowed and origin not in allowed:
                wrong_origin += 1
            elif not (config.DATE_START <= timestamp.date() <= config.DATE_END):
                outside += 1
            else:
                keep.append(index)

        return dataset.select(keep), {
            "before": len(dataset),
            "after": len(keep),
            "invalid_dates": invalid,
            "outside_range": outside,
            "wrong_origin": wrong_origin,
        }

    def _deduplicate(
        self,
        datikz: Dataset,
        benchmark: Dataset,
    ) -> tuple[Dataset, Dataset, dict]:
        benchmark_hashes = [code_hash(value) for value in tqdm(
            benchmark[str(BENCHMARK.columns["code"])], desc="Hash benchmark"
        )]
        datikz_hashes = [code_hash(value) for value in tqdm(
            datikz[str(DATIKZ.columns["code"])], desc="Hash DaTikZ"
        )]

        benchmark_counts = Counter(value for value in benchmark_hashes if value)
        datikz_counts = Counter(value for value in datikz_hashes if value)
        benchmark_set = set(benchmark_counts)

        benchmark_keep = self._first_occurrences(benchmark_hashes)
        datikz_unique = self._first_occurrences(datikz_hashes)
        datikz_keep = [
            index for index in datikz_unique
            if not datikz_hashes[index] or datikz_hashes[index] not in benchmark_set
        ]

        benchmark = self._add_ids(
            benchmark.select(benchmark_keep),
            BENCHMARK,
            [benchmark_hashes[index] for index in benchmark_keep],
        )
        datikz = self._add_ids(
            datikz.select(datikz_keep),
            DATIKZ,
            [datikz_hashes[index] for index in datikz_keep],
        )

        cross = benchmark_set.intersection(datikz_counts)
        stats = {
            "benchmark": self._duplicate_stats(benchmark_counts),
            "datikz": self._duplicate_stats(datikz_counts),
            "cross_dataset": {
                "matching_hashes": len(cross),
                "datikz_rows_matching_before_internal_dedup": sum(datikz_counts[value] for value in cross),
                "datikz_unique_rows_removed": len(set(datikz_hashes[index] for index in datikz_unique).intersection(benchmark_set)),
            },
            "rows_after": {"datikz": len(datikz), "benchmark": len(benchmark)},
        }
        return datikz, benchmark, stats

    @staticmethod
    def _first_occurrences(hashes: list[str]) -> list[int]:
        seen: set[str] = set()
        keep: list[int] = []
        for index, value in enumerate(hashes):
            if not value or value not in seen:
                keep.append(index)
                if value:
                    seen.add(value)
        return keep

    @staticmethod
    def _duplicate_stats(counts: Counter[str]) -> dict:
        duplicates = [count for count in counts.values() if count > 1]
        return {
            "groups": len(duplicates),
            "rows_removed": sum(count - 1 for count in duplicates),
            "double": duplicates.count(2),
            "triple": duplicates.count(3),
            "quadruple": duplicates.count(4),
            "five_plus": sum(count >= 5 for count in duplicates),
            "max_multiplicity": max(duplicates, default=1),
        }

    @staticmethod
    def _add_ids(dataset: Dataset, source: Source, hashes: list[str]) -> Dataset:
        id_column = source.columns["id"]
        raw_ids = dataset[str(id_column)] if id_column else [None] * len(dataset)
        seen: Counter[str] = Counter()
        sample_ids: list[str] = []

        for index, (raw_id, value_hash) in enumerate(zip(raw_ids, hashes)):
            base = str(raw_id or f"row-{index:09d}").strip()
            candidate = f"{source.name}:{base}"
            seen[candidate] += 1
            if seen[candidate] > 1:
                candidate += f":{seen[candidate]}:{value_hash[:8]}"
            sample_ids.append(candidate)

        return dataset.add_column("_sample_id", sample_ids).add_column("_code_hash", hashes)

    def _filter_tokens(self, dataset: Dataset, source: Source) -> tuple[Dataset, dict]:
        code_column = str(source.columns["code"])
        keep: list[int] = []
        rejected: list[dict] = []
        max_seen = 0

        for start in tqdm(range(0, len(dataset), config.TOKEN_BATCH_SIZE), desc=f"Tokenize {source.name}"):
            stop = min(start + config.TOKEN_BATCH_SIZE, len(dataset))
            rows = dataset[start:stop]
            codes = [normalize_code(value) for value in rows[code_column]]
            encoded = self.tokenizer(
                codes,
                add_special_tokens=True,
                truncation=True,
                max_length=config.MAX_TOKENS + 1,
            )["input_ids"]

            for offset, (code, token_ids) in enumerate(zip(codes, encoded)):
                count = len(token_ids)
                max_seen = max(max_seen, count)
                index = start + offset
                if code and count <= config.MAX_TOKENS:
                    keep.append(index)
                else:
                    rejected.append({
                        "dataset": source.name,
                        "sample_id": rows["_sample_id"][offset],
                        "token_count": count,
                        "reason": "empty_code" if not code else "too_long",
                    })

        if rejected:
            path = config.REPORT_DIR / "token_rejections.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rejected[0].keys())
                if write_header:
                    writer.writeheader()
                writer.writerows(rejected)

        return dataset.select(keep), {
            "before": len(dataset),
            "after": len(keep),
            "rejected": len(rejected),
            "maximum_seen": max_seen,
        }
