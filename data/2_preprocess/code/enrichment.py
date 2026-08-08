"""Cluster images, create splits and generate selected Ollama descriptions."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

import config
from output import iter_batches, iter_rows, row_count


class Clusterer:
    """Fit PCA and KMeans on the combined DaTikZ and benchmark embeddings."""

    def run(self) -> tuple[dict[str, str], dict]:
        datasets = config.CLUSTER_DATASETS
        rows_per_dataset = {name: row_count(name) for name in datasets}
        total = sum(rows_per_dataset.values())
        if total < config.N_CLUSTERS:
            raise ValueError(f"Not enough samples for {config.N_CLUSTERS} clusters: {total}")

        first = next(self._raw_embedding_batches(batch_size=1))[0]
        components = min(config.PCA_COMPONENTS, len(first), total - 1)
        pca = IncrementalPCA(
            n_components=components,
            batch_size=config.CLUSTER_BATCH_SIZE,
        )
        for vectors in tqdm(
            self._fit_batches(total, minimum=components),
            desc="Fit PCA on all datasets",
        ):
            pca.partial_fit(vectors)

        kmeans = MiniBatchKMeans(
            n_clusters=config.N_CLUSTERS,
            batch_size=config.CLUSTER_BATCH_SIZE,
            random_state=config.SEED,
            n_init=10,
        )
        initialized = False
        for vectors in tqdm(
            self._fit_batches(total, minimum=config.N_CLUSTERS),
            desc="Fit KMeans on all datasets",
        ):
            reduced = pca.transform(vectors)
            kmeans.partial_fit(reduced)
            initialized = True
        if not initialized:
            raise ValueError("MiniBatchKMeans could not be initialized")

        model_dir = config.METADATA_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, model_dir / "pca.joblib")
        joblib.dump(kmeans, model_dir / "kmeans.joblib")

        assignments: dict[str, str] = {}
        counts: dict[str, Counter] = {}
        for dataset_name in datasets:
            counter: Counter[str] = Counter()
            batches = iter_batches(
                dataset_name,
                ["sample_id", "clip_embedding"],
                config.CLUSTER_BATCH_SIZE,
            )
            for batch in tqdm(batches, desc=f"Assign {dataset_name}"):
                vectors = np.asarray(batch["clip_embedding"], dtype=np.float32)
                labels = kmeans.predict(pca.transform(vectors))
                for sample_id, label in zip(batch["sample_id"], labels):
                    name = config.class_name(int(label))
                    assignments[sample_id] = name
                    counter[name] += 1
            counts[dataset_name] = counter

        return assignments, {
            "fit_datasets": list(datasets),
            "fit_rows": total,
            "rows_per_dataset": rows_per_dataset,
            "pca_components": components,
            "classes_per_dataset": {
                name: dict(counts[name]) for name in datasets
            },
        }

    @staticmethod
    def _raw_embedding_batches(batch_size: int | None = None):
        """Yield embedding batches from every dataset in round-robin order."""
        size = batch_size or config.CLUSTER_BATCH_SIZE
        iterators = {
            name: iter(iter_batches(name, ["clip_embedding"], size))
            for name in config.CLUSTER_DATASETS
        }
        active = list(config.CLUSTER_DATASETS)
        while active:
            for name in active.copy():
                try:
                    batch = next(iterators[name])
                except StopIteration:
                    active.remove(name)
                    continue
                yield np.asarray(batch["clip_embedding"], dtype=np.float32)

    @classmethod
    def _fit_batches(cls, total: int, minimum: int):
        """Yield all embeddings in valid partial-fit batches without dropping a tail."""
        pending: list[np.ndarray] = []
        pending_rows = 0
        seen = 0

        for vectors in cls._raw_embedding_batches():
            pending.append(vectors)
            pending_rows += len(vectors)
            seen += len(vectors)
            remaining = total - seen

            if pending_rows >= config.CLUSTER_BATCH_SIZE and remaining >= minimum:
                yield np.concatenate(pending)
                pending = []
                pending_rows = 0

        if pending_rows:
            if pending_rows < minimum:
                raise ValueError(
                    f"Final clustering batch has {pending_rows} rows, but needs at least {minimum}"
                )
            yield np.concatenate(pending)


class OllamaClient:
    def __init__(self) -> None:
        self.prompt = config.PROMPT_PATH.read_text(encoding="utf-8").strip()

    def describe(self, code: str) -> str:
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": f"{self.prompt}\n\n```latex\n{code}\n```",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        error: Exception | None = None
        for attempt in range(config.OLLAMA_RETRIES + 1):
            try:
                response = requests.post(
                    f"{config.OLLAMA_URL.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=config.OLLAMA_TIMEOUT,
                )
                response.raise_for_status()
                text = str(response.json()["response"]).strip()
                if not text:
                    raise ValueError("Empty Ollama response")
                return text
            except Exception as current:
                error = current
                if attempt < config.OLLAMA_RETRIES:
                    time.sleep(2**attempt)
        raise RuntimeError(error)


class Enricher:
    def run(self) -> dict:
        assignments, clustering_stats = Clusterer().run()
        split_map = self._splits()
        descriptions, description_stats = self._descriptions(assignments, split_map)
        self._write_metadata(assignments, split_map, descriptions)
        return {
            "clustering": clustering_stats,
            "splits": dict(Counter(split_map.values())),
            "descriptions": description_stats,
        }

    @staticmethod
    def _splits() -> dict[str, str]:
        datikz_rows = row_count("datikz")
        if config.VAL_SIZE > datikz_rows:
            raise ValueError(f"VAL_SIZE={config.VAL_SIZE} exceeds DaTikZ rows={datikz_rows}")

        split_map: dict[str, str] = {}
        for index, row in enumerate(iter_rows("datikz", ["sample_id"])):
            split_map[row["sample_id"]] = "val" if index < config.VAL_SIZE else "train"
        for row in iter_rows("benchmark", ["sample_id"]):
            split_map[row["sample_id"]] = "benchmark"
        return split_map

    def _descriptions(self, assignments: dict[str, str], split_map: dict[str, str]) -> tuple[dict[str, str], dict]:
        if config.DESCRIPTIONS_PER_CLASS <= 0:
            return {}, {"generated": 0}

        limit = config.DESCRIPTIONS_PER_CLASS * config.DESCRIPTION_CANDIDATE_FACTOR
        candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for dataset_name in config.CLUSTER_DATASETS:
            for row in iter_rows(dataset_name, ["sample_id", "reference_code"]):
                class_name = assignments[row["sample_id"]]
                if len(candidates[class_name]) < limit:
                    candidates[class_name].append((row["sample_id"], row["reference_code"]))

        def generate_class(item: tuple[str, list[tuple[str, str]]]) -> tuple[str, dict[str, str], list[dict]]:
            class_name, rows = item
            client = OllamaClient()
            generated: dict[str, str] = {}
            failures: list[dict] = []
            for sample_id, code in rows:
                if len(generated) >= config.DESCRIPTIONS_PER_CLASS:
                    break
                try:
                    generated[sample_id] = client.describe(code)
                except Exception as error:
                    failures.append({"sample_id": sample_id, "class": class_name, "reason": repr(error)})
            return class_name, generated, failures

        descriptions: dict[str, str] = {}
        failures: list[dict] = []
        workers = max(1, min(config.OLLAMA_WORKERS, len(candidates)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _, generated, class_failures in executor.map(generate_class, sorted(candidates.items())):
                descriptions.update(generated)
                failures.extend(class_failures)

        if failures:
            table = pa.Table.from_pylist(failures)
            pq.write_table(table, config.REPORT_DIR / "description_failures.parquet")

        return descriptions, {
            "generated": len(descriptions),
            "failures": len(failures),
            "per_class": dict(Counter(assignments[sample_id] for sample_id in descriptions)),
        }

    @staticmethod
    def _write_metadata(assignments: dict[str, str], split_map: dict[str, str], descriptions: dict[str, str]) -> None:
        path = config.METADATA_DIR / "metadata.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            ("sample_id", pa.string()),
            ("split", pa.string()),
            ("class", pa.string()),
            ("llm_description", pa.string()),
        ])
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        buffer: list[dict] = []
        for dataset_name in config.CLUSTER_DATASETS:
            for row in iter_rows(dataset_name, ["sample_id"]):
                sample_id = row["sample_id"]
                buffer.append({
                    "sample_id": sample_id,
                    "split": split_map[sample_id],
                    "class": assignments[sample_id],
                    "llm_description": descriptions.get(sample_id),
                })
                if len(buffer) >= 2_048:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        writer.close()
