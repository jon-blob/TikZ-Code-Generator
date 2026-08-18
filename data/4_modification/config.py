"""Configuration for post-processing an already published dataset."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PREPROCESS_DIR = REPO_ROOT / "2_preprocess"
PREPROCESS_CODE_DIR = PREPROCESS_DIR / "code"
PROMPT_DIR = PREPROCESS_DIR / "prompts"

# Hugging Face dataset
HF_REPO_ID = "loss-boss/tikz-dataset-clean"
HF_REVISION = "main"
HF_TOKEN: str | None = None  # use `hf auth login`
DOWNLOAD_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "dataset_modified"
UPLOAD = False
UPLOAD_REPO_ID = "loss-boss/tikz-dataset-clean-extended"
HF_PRIVATE = False

# Description types: "code", "image", "image_code"
DESCRIPTION_TYPES = ["code", "image"]
DESCRIPTION_COLUMNS = {
    "code": "llm_description",
    "image": "llm_description_image",
    "image_code": "llm_description_image_code",
}

# Additional descriptions per split × image class × repetition class.
ADDITIONAL_DESCRIPTIONS_PER_REPETITION_CLASS = {
    "low": 300,
    "medium": 300,
    "high": 300,
    "very_high": 300,
    "critical": 300,
}
DESCRIPTION_CANDIDATE_FACTOR = 3
SPLITS = ("train", "benchmark")

# Reuse the preprocessing prompts.
DESCRIPTION_PROMPTS = {
    "code": PROMPT_DIR / "description_code_prompt.txt",
    "image": PROMPT_DIR / "description_image_prompt.txt",
    "image_code": PROMPT_DIR / "description_image_code_prompt.txt",
}

# Ollama settings used by 2_preprocess/code/components/ollama_client.py.
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TEXT_MODEL = "gemma4:31b-it-q8_0"
OLLAMA_VISION_MODEL = "gemma4:31b-it-q8_0"
OLLAMA_TIMEOUT = 1_800
OLLAMA_RETRIES = 1
OLLAMA_PARALLEL_REQUESTS = 5
OLLAMA_TEMPERATURE = 0.1
OLLAMA_NUM_PREDICT = 4096
OLLAMA_NUM_CTX = 8192

OVERWRITE_OUTPUT = True
