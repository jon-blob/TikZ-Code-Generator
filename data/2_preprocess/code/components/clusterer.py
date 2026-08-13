"""Joint image clustering with configurable reducer and clusterer."""

from __future__ import annotations

from collections import Counter

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

import config
from output import iter_batches, row_count


class Clusterer:
    """Cluster the shared configurable image-embedding space."""

    REDUCERS = {"pca", "umap"}
    ALGORITHMS = {"kmeans", "hdbscan"}

    def run(self) -> tuple[dict[str, str], dict]:
        reducer_name = config.CLUSTER_REDUCER.lower()
        algorithm_name = config.CLUSTER_ALGORITHM.lower()
        if reducer_name not in self.REDUCERS:
            raise ValueError(f"CLUSTER_REDUCER must be one of {sorted(self.REDUCERS)}")
        if algorithm_name not in self.ALGORITHMS:
            raise ValueError(f"CLUSTER_ALGORITHM must be one of {sorted(self.ALGORITHMS)}")

        datasets = config.CLUSTER_DATASETS
        self._validate_encoder()
        rows_per_dataset = {name: row_count(name) for name in datasets}
        total = sum(rows_per_dataset.values())
        if total < 2:
            raise ValueError("Not enough samples for clustering")
        if algorithm_name == "kmeans" and total < config.N_CLUSTERS:
            raise ValueError(f"Not enough samples for {config.N_CLUSTERS} clusters: {total}")

        first = next(self._raw_embedding_batches(batch_size=1))[0]
        embedding_dim = len(first)

        if reducer_name == "pca":
            reducer, components = self._fit_pca(total, embedding_dim)
            if algorithm_name == "kmeans":
                clusterer = self._fit_kmeans_streaming(reducer, total)
                assignments, counts = self._assign_streaming(reducer, clusterer)
                labels = None
            else:
                reduced, sample_ids, dataset_names = self._load_reduced(reducer.transform)
                clusterer = self._fit_hdbscan(reduced)
                labels = clusterer.labels_
                assignments, counts = self._assign_from_labels(sample_ids, dataset_names, labels)
        else:
            vectors, sample_ids, dataset_names = self._load_embeddings()
            reducer, reduced, components = self._fit_umap(vectors, total)
            if algorithm_name == "kmeans":
                clusterer = self._fit_kmeans(reduced)
                labels = clusterer.predict(reduced)
            else:
                clusterer = self._fit_hdbscan(reduced)
                labels = clusterer.labels_
            assignments, counts = self._assign_from_labels(sample_ids, dataset_names, labels)

        model_dir = config.METADATA_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, model_dir / f"{reducer_name}.joblib")
        joblib.dump(clusterer, model_dir / f"{algorithm_name}.joblib")

        if labels is None:
            noise_rows = 0
            discovered_clusters = config.N_CLUSTERS
        else:
            labels_array = np.asarray(labels)
            noise_rows = int(np.sum(labels_array < 0))
            discovered_clusters = len(set(int(label) for label in labels_array if label >= 0))

        stats = {
            "fit_datasets": list(datasets),
            "fit_rows": total,
            "rows_per_dataset": rows_per_dataset,
            "embedding": config.IMAGE_ENCODER.lower(),
            "reducer": reducer_name,
            "clusterer": algorithm_name,
            "components": components,
            "clusters": discovered_clusters,
            "noise_rows": noise_rows,
            "noise_fraction": noise_rows / total,
            "classes_per_dataset": {name: dict(counts[name]) for name in datasets},
        }
        if reducer_name == "pca":
            stats["pca_explained_variance"] = float(reducer.explained_variance_ratio_.sum())

        return assignments, stats


    @staticmethod
    def _validate_encoder() -> None:
        expected = config.IMAGE_ENCODER.lower()
        for dataset_name in config.CLUSTER_DATASETS:
            first_batch = next(iter_batches(dataset_name, ["image_encoder"], 1))
            actual = str(first_batch["image_encoder"][0]).lower()
            if actual != expected:
                raise ValueError(
                    f"Staging for {dataset_name} was created with {actual!r}, "
                    f"but IMAGE_ENCODER is {expected!r}. Re-run preprocessing."
                )

    def _fit_pca(self, total: int, embedding_dim: int):
        components = min(config.PCA_COMPONENTS, embedding_dim, total - 1)
        reducer = IncrementalPCA(
            n_components=components,
            batch_size=config.CLUSTER_BATCH_SIZE,
        )
        for vectors in tqdm(
            self._fit_batches(total, minimum=components),
            desc=f"Fit PCA ({components} components)",
        ):
            reducer.partial_fit(vectors)
        return reducer, components

    @staticmethod
    def _fit_umap(vectors: np.ndarray, total: int):
        try:
            from umap import UMAP
        except ImportError as error:
            raise ImportError(
                "UMAP clustering requires `umap-learn`. Install it with `pip install umap-learn`."
            ) from error

        components = min(config.UMAP_COMPONENTS, max(1, total - 2))
        n_neighbors = min(config.UMAP_N_NEIGHBORS, max(2, total - 1))
        reducer = UMAP(
            n_components=components,
            n_neighbors=n_neighbors,
            min_dist=config.UMAP_MIN_DIST,
            metric=config.UMAP_METRIC,
            random_state=config.SEED,
        )
        reduced = reducer.fit_transform(vectors).astype(np.float32, copy=False)
        return reducer, reduced, components

    @staticmethod
    def _fit_kmeans(reduced: np.ndarray):
        model = MiniBatchKMeans(
            n_clusters=config.N_CLUSTERS,
            batch_size=config.CLUSTER_BATCH_SIZE,
            random_state=config.SEED,
            n_init=10,
        )
        model.fit(reduced)
        return model

    def _fit_kmeans_streaming(self, reducer, total: int):
        model = MiniBatchKMeans(
            n_clusters=config.N_CLUSTERS,
            batch_size=config.CLUSTER_BATCH_SIZE,
            random_state=config.SEED,
            n_init=10,
        )
        initialized = False
        for vectors in tqdm(
            self._fit_batches(total, minimum=config.N_CLUSTERS),
            desc="Fit KMeans",
        ):
            model.partial_fit(reducer.transform(vectors))
            initialized = True
        if not initialized:
            raise ValueError("MiniBatchKMeans could not be initialized")
        return model

    @staticmethod
    def _fit_hdbscan(reduced: np.ndarray):
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as error:
            raise ImportError(
                "HDBSCAN requires a scikit-learn version that provides sklearn.cluster.HDBSCAN."
            ) from error

        model = HDBSCAN(
            min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=config.HDBSCAN_MIN_SAMPLES,
            cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
            cluster_selection_epsilon=config.HDBSCAN_CLUSTER_SELECTION_EPSILON,
            metric=config.HDBSCAN_METRIC,
            n_jobs=config.HDBSCAN_N_JOBS,
            copy=True,
        )
        model.fit(reduced)
        return model

    def _assign_streaming(self, reducer, clusterer):
        assignments: dict[str, str] = {}
        counts: dict[str, Counter] = {}
        for dataset_name in config.CLUSTER_DATASETS:
            counter: Counter[str] = Counter()
            batches = iter_batches(
                dataset_name,
                ["sample_id", "image_embedding"],
                config.CLUSTER_BATCH_SIZE,
            )
            for batch in tqdm(batches, desc=f"Assign {dataset_name}"):
                vectors = np.asarray(batch["image_embedding"], dtype=np.float32)
                labels = clusterer.predict(reducer.transform(vectors))
                for sample_id, label in zip(batch["sample_id"], labels):
                    name = config.class_name(int(label))
                    assignments[sample_id] = name
                    counter[name] += 1
            counts[dataset_name] = counter
        return assignments, counts

    @staticmethod
    def _assign_from_labels(sample_ids, dataset_names, labels):
        assignments: dict[str, str] = {}
        counts: dict[str, Counter] = {
            name: Counter() for name in config.CLUSTER_DATASETS
        }
        for sample_id, dataset_name, label in zip(sample_ids, dataset_names, labels):
            class_name = config.class_name(int(label))
            assignments[sample_id] = class_name
            counts[dataset_name][class_name] += 1
        return assignments, counts

    @staticmethod
    def _load_embeddings():
        vectors: list[np.ndarray] = []
        sample_ids: list[str] = []
        dataset_names: list[str] = []
        for dataset_name in config.CLUSTER_DATASETS:
            batches = iter_batches(
                dataset_name,
                ["sample_id", "image_embedding"],
                config.CLUSTER_BATCH_SIZE,
            )
            for batch in tqdm(batches, desc=f"Load embeddings: {dataset_name}"):
                part = np.asarray(batch["image_embedding"], dtype=np.float32)
                vectors.append(part)
                sample_ids.extend(batch["sample_id"])
                dataset_names.extend([dataset_name] * len(part))
        return np.concatenate(vectors), sample_ids, dataset_names

    @staticmethod
    def _load_reduced(transform):
        vectors: list[np.ndarray] = []
        sample_ids: list[str] = []
        dataset_names: list[str] = []
        for dataset_name in config.CLUSTER_DATASETS:
            batches = iter_batches(
                dataset_name,
                ["sample_id", "image_embedding"],
                config.CLUSTER_BATCH_SIZE,
            )
            for batch in tqdm(batches, desc=f"Reduce embeddings: {dataset_name}"):
                raw = np.asarray(batch["image_embedding"], dtype=np.float32)
                vectors.append(transform(raw).astype(np.float32, copy=False))
                sample_ids.extend(batch["sample_id"])
                dataset_names.extend([dataset_name] * len(raw))
        return np.concatenate(vectors), sample_ids, dataset_names

    @staticmethod
    def _raw_embedding_batches(batch_size: int | None = None):
        """Yield embedding batches from every dataset in round-robin order."""
        size = batch_size or config.CLUSTER_BATCH_SIZE
        iterators = {
            name: iter(iter_batches(name, ["image_embedding"], size))
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
                yield np.asarray(batch["image_embedding"], dtype=np.float32)

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

