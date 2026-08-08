from pathlib import Path

from huggingface_hub import HfApi


# =========================
# CONFIG
# =========================
DATA_DIR = Path("../data")
REPO_ID = "loss-boss/tikz-train-split"
PRIVATE = True


README = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: "train/*.parquet"
  - split: val
    path: "val/*.parquet"
  - split: crystalbleu
    path: "crystalbleu-corpus/*.parquet"
---

# TikZ Training Dataset

The dataset contains the following splits:

- `train`
- `val`
- `crystalbleu`

All splits use the same columns:

- `images`
- `code`
- `llm_description`
- `type`
"""


def validate_data() -> None:
    split_directories = {
        "train": DATA_DIR / "train",
        "val": DATA_DIR / "val",
        "crystalbleu": DATA_DIR / "crystalbleu-corpus",
    }

    for split, directory in split_directories.items():
        parquet_files = list(directory.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found for split '{split}' in: {directory}"
            )

        print(f"{split}: {len(parquet_files)} shards")


def main() -> None:
    validate_data()

    # Define the splits for the Hugging Face dataset viewer and loader.
    (DATA_DIR / "README.md").write_text(README, encoding="utf-8")

    api = HfApi()

    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=PRIVATE,
        exist_ok=True,
    )

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(DATA_DIR),
    )

    print(f"Uploaded dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()