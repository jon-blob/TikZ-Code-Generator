"""Edit this file to configure the complete pipeline."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Paths
DATIKZ_DIR = Path("/home/jonas/Datasets/TikZ/DatikZ-v4-raw-test")
BENCHMARK_DIR = Path("/home/jonas/Datasets/TikZ/benchmark-raw-test")
OUTPUT_DIR = Path("../../preprocessed")
TOKENIZER_PATH = Path("/home/jonas/models/gemma-4-31B-it-unsloth-bnb-4bit")
PROMPT_PATH = Path("../prompts/description_prompt.txt")

STAGING_DIR = OUTPUT_DIR / "staging"
METADATA_DIR = OUTPUT_DIR / "metadata"
REPORT_DIR = OUTPUT_DIR / "reports"
FAILURE_DIR = OUTPUT_DIR / "failures"
EXPORT_DIR = OUTPUT_DIR / "huggingface"
CACHE_DIR = OUTPUT_DIR / "cache"

# Input columns
DATIKZ_COLUMNS = {
    "id": "file_id",
    "code": "tikz_code",
    "image": "png_image",
    "source": "source",
    "date": None,
}
BENCHMARK_COLUMNS = {
    "id": "uri",
    "code": "code",
    "image": "image",
    "source": "origin",
    "date": "date",
}

# Cleaning
DATE_START = date(2025, 10, 1)
DATE_END = date(2026, 4, 30)
BENCHMARK_ORIGINS: tuple[str, ...] = ()
SEED = 42
MAX_TOKENS = 4096
TOKEN_BATCH_SIZE = 64

# Rendering and CLIP
IMAGE_SIZE = 512
DPI = 400
LATEX_TIMEOUT = 45
LATEX_BIN_DIR: Path | None = Path("/usr/local/texlive/2026/bin/x86_64-linux")
LATEX_ENGINES = ("pdflatex", "lualatex", "xelatex")
RENDER_WORKERS = max(1, min(8, os.cpu_count() or 1))
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_DEVICE = "auto"
CLIP_BATCH_SIZE = 64
CLIP_THRESHOLD = 0.85
MIN_INK_FRACTION = 0.002

# Clustering and descriptions
# The reducer and clusterer are fitted jointly on these staging datasets.
# Recommended setup from the analysis notebook: CLIP -> UMAP(20) -> HDBSCAN.
CLUSTER_DATASETS = ("datikz", "benchmark")
CLUSTER_REDUCER = "umap"       # "pca" or "umap"
CLUSTER_ALGORITHM = "hdbscan"  # "kmeans" or "hdbscan"

# Dimensionality reduction
PCA_COMPONENTS = 20
UMAP_COMPONENTS = 20
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

# KMeans (used only when CLUSTER_ALGORITHM == "kmeans")
N_CLUSTERS = 6

# HDBSCAN (used only when CLUSTER_ALGORITHM == "hdbscan")
HDBSCAN_MIN_CLUSTER_SIZE = 200
HDBSCAN_MIN_SAMPLES = 20
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0
HDBSCAN_METRIC = "euclidean"
HDBSCAN_N_JOBS = -1
HDBSCAN_NOISE_CLASS = "noise"

CLUSTER_BATCH_SIZE = 4_096
CLASS_NAMES: dict[int, str] = {}
DESCRIPTIONS_PER_CLASS = 10
DESCRIPTION_CANDIDATE_FACTOR = 3
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_TIMEOUT = 1_800
OLLAMA_RETRIES = 1
OLLAMA_WORKERS = 1

# Parquet and upload
MAX_PARQUET_BYTES = 134_000_000
PARQUET_BUFFER_BYTES = 110_000_000
STAGING_ROWS_PER_FILE = 10_000
HF_REPO_ID = "loss-boss/tikz-dataset-clean"
HF_PRIVATE = False
HF_TOKEN: str | None = "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"  # Uses the token from `hf auth login` when available.

# Execution
OVERWRITE = True
RUN_PREPROCESSING = True
RUN_ENRICHMENT = True
RUN_EXPORT = True
RUN_UPLOAD = True


def class_name(cluster_id: int) -> str:
    if cluster_id < 0:
        return HDBSCAN_NOISE_CLASS
    return CLASS_NAMES.get(cluster_id, f"class_{cluster_id + 1}")
