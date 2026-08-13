"""Add repetition metadata, image clusters and selected Ollama descriptions."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import config
from components import Clusterer, OllamaClient, RepetitionClassifier
from output import iter_rows


DESCRIPTION_COLUMNS = {
    "code": "llm_description",
    "image": "llm_description_image",
    "image_code": "llm_description_image_code",
}


class DatasetEnricher:
    def __init__(self) -> None:
        invalid = set(config.DESCRIPTION_TYPES).difference(DESCRIPTION_COLUMNS)
        if invalid:
            raise ValueError(f"Unknown DESCRIPTION_TYPES: {sorted(invalid)}")
        self.description_types = tuple(dict.fromkeys(config.DESCRIPTION_TYPES))

    def run(self) -> dict:
        repetition, repetition_stats = RepetitionClassifier().run()
        assignments, clustering_stats = Clusterer().run()
        split_map = self._splits()
        descriptions, description_stats = self._descriptions(assignments, repetition)
        self._write_metadata(assignments, repetition, split_map, descriptions)
        return {
            "repetition": repetition_stats,
            "clustering": clustering_stats,
            "splits": dict(Counter(split_map.values())),
            "descriptions": description_stats,
        }

    @staticmethod
    def _splits() -> dict[str, str]:
        split_map = {
            row["sample_id"]: "train"
            for row in iter_rows("datikz", ["sample_id"])
        }
        split_map.update({
            row["sample_id"]: "benchmark"
            for row in iter_rows("benchmark", ["sample_id"])
        })
        return split_map

    @staticmethod
    def _description_target(repetition_class: str) -> int:
        return max(
            0,
            int(config.DESCRIPTIONS_PER_REPETITION_CLASS.get(repetition_class, 0)),
        )

    def _descriptions(
        self,
        assignments: dict[str, str],
        repetition: dict[str, tuple[str, int]],
    ) -> tuple[dict[str, dict[str, str]], dict]:
        descriptions = {name: {} for name in self.description_types}
        if not self.description_types:
            return descriptions, {"generated": {}, "total": 0}

        if not any(
            self._description_target(name) > 0
            for name in config.REPETITION_CLASSES
        ):
            return descriptions, {"generated": {}, "total": 0}

        candidates: dict[
            tuple[str, str, str],
            list[tuple[str, str, bytes]],
        ] = defaultdict(list)

        for dataset_name in config.CLUSTER_DATASETS:
            for row in iter_rows(
                dataset_name,
                ["sample_id", "reference_code", "rendered_image"],
            ):
                sample_id = row["sample_id"]
                class_name = assignments[sample_id]
                repetition_class = repetition[sample_id][0]
                target = self._description_target(repetition_class)
                if target <= 0:
                    continue

                key = (dataset_name, class_name, repetition_class)
                limit = target * config.DESCRIPTION_CANDIDATE_FACTOR
                if len(candidates[key]) < limit:
                    candidates[key].append((
                        sample_id,
                        row["reference_code"],
                        row["rendered_image"],
                    ))

        total_target = sum(
            min(self._description_target(key[2]), len(rows))
            * len(self.description_types)
            for key, rows in candidates.items()
        )
        progress = tqdm(
            total=total_target,
            desc="Generate descriptions",
            unit="description",
            dynamic_ncols=True,
        )

        client = OllamaClient()
        failures: list[dict] = []
        generated_groups = {name: {} for name in self.description_types}

        # One state per dataset × image class × repetition class × description type.
        # The first `wanted` candidates are submitted immediately. If one fails,
        # the next candidate from the same group is submitted as fallback.
        states: dict[tuple[str, str, str, str], dict] = {}
        for (dataset_name, class_name, repetition_class), rows in sorted(candidates.items()):
            wanted = min(self._description_target(repetition_class), len(rows))
            for description_type in self.description_types:
                states[(dataset_name, class_name, repetition_class, description_type)] = {
                    "rows": rows,
                    "wanted": wanted,
                    "next_index": 0,
                    "generated": 0,
                    "in_flight": 0,
                }

        workers = max(1, int(config.OLLAMA_PARALLEL_REQUESTS))

        def generate(
            description_type: str,
            row: tuple[str, str, bytes],
        ) -> str:
            _, code, image = row
            return client.describe(
                description_type,
                code=code,
                image=image,
            )

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = {}

                def submit_next(bucket: tuple[str, str, str, str]) -> bool:
                    state = states[bucket]
                    if state["next_index"] >= len(state["rows"]):
                        return False
                    if state["generated"] + state["in_flight"] >= state["wanted"]:
                        return False

                    row = state["rows"][state["next_index"]]
                    state["next_index"] += 1
                    state["in_flight"] += 1
                    future = executor.submit(generate, bucket[3], row)
                    pending[future] = (bucket, row)
                    return True

                for bucket, state in states.items():
                    while state["generated"] + state["in_flight"] < state["wanted"]:
                        if not submit_next(bucket):
                            break

                while pending:
                    done, _ = wait(
                        tuple(pending),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        bucket, row = pending.pop(future)
                        dataset_name, class_name, repetition_class, description_type = bucket
                        sample_id = row[0]
                        state = states[bucket]
                        state["in_flight"] -= 1

                        try:
                            descriptions[description_type][sample_id] = future.result()
                            generated_groups[description_type][sample_id] = (
                                dataset_name,
                                class_name,
                                repetition_class,
                            )
                            state["generated"] += 1
                            progress.update(1)
                        except Exception as error:
                            failures.append({
                                "dataset": dataset_name,
                                "sample_id": sample_id,
                                "class": class_name,
                                "repetition_class": repetition_class,
                                "description_type": description_type,
                                "reason": repr(error),
                            })

                        while state["generated"] + state["in_flight"] < state["wanted"]:
                            if not submit_next(bucket):
                                break

                        progress.set_postfix(
                            dataset=dataset_name,
                            class_name=class_name,
                            repetition=repetition_class,
                            workers=workers,
                            failures=len(failures),
                        )
        finally:
            progress.close()

        if failures:
            config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(failures),
                config.REPORT_DIR / "description_failures.parquet",
            )

        per_type = {
            name: len(values)
            for name, values in descriptions.items()
        }
        return descriptions, {
            "enabled_types": list(self.description_types),
            "parallel_requests": workers,
            "generated": per_type,
            "total": sum(per_type.values()),
            "failures": len(failures),
            "targets_per_repetition_class": dict(config.DESCRIPTIONS_PER_REPETITION_CLASS),
            "per_type_dataset_class_repetition": {
                description_type: dict(Counter(
                    f"{dataset_name}:{class_name}:{repetition_class}"
                    for dataset_name, class_name, repetition_class
                    in groups.values()
                ))
                for description_type, groups in generated_groups.items()
            },
        }

    @staticmethod
    def _write_metadata(
        assignments: dict[str, str],
        repetition: dict[str, tuple[str, int]],
        split_map: dict[str, str],
        descriptions: dict[str, dict[str, str]],
    ) -> None:
        path = config.METADATA_DIR / "metadata.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            ("sample_id", pa.string()),
            ("split", pa.string()),
            ("class", pa.string()),
            ("repetition_class", pa.string()),
            ("token_len", pa.int32()),
            ("llm_description", pa.string()),
            ("llm_description_image", pa.string()),
            ("llm_description_image_code", pa.string()),
        ])
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        buffer: list[dict] = []

        for dataset_name in config.CLUSTER_DATASETS:
            for row in iter_rows(dataset_name, ["sample_id"]):
                sample_id = row["sample_id"]
                repetition_class, token_len = repetition[sample_id]
                buffer.append({
                    "sample_id": sample_id,
                    "split": split_map[sample_id],
                    "class": assignments[sample_id],
                    "repetition_class": repetition_class,
                    "token_len": token_len,
                    "llm_description": descriptions.get("code", {}).get(sample_id),
                    "llm_description_image": descriptions.get("image", {}).get(sample_id),
                    "llm_description_image_code": descriptions.get("image_code", {}).get(sample_id),
                })
                if len(buffer) >= 2_048:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()

        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        writer.close()
