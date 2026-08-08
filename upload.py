"""Upload the raw local TikZ dataset to Hugging Face."""

from pathlib import Path

from datasets import Image, load_dataset


DATA_DIR = Path("/home/jonas/Datasets/TikZ/benchmark-not-clean")
REPO_ID = "loss-boss/tikz-benchmark"
PRIVATE = False
HF_TOKEN: str | None = "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"

EXPECTED_COLUMNS = [
    "caption",
    "code",
    "image",
    "pdf",
    "uri",
    "origin",
    "date",
]


def main() -> None:
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")

    dataset = load_dataset(
        "parquet",
        data_files=[str(path) for path in files],
        split="train",
    )

    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataset.column_names
    ]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Preserve every column and only mark the existing image column
    # as a Hugging Face image feature for the Dataset Viewer.
    dataset = dataset.select_columns(EXPECTED_COLUMNS)
    dataset = dataset.cast_column("image", Image(decode=False))

    dataset.push_to_hub(
        repo_id=REPO_ID,
        split="benchmark",
        private=PRIVATE,
        token=HF_TOKEN,
        max_shard_size="134MB",
        embed_external_files=True,
    )


if __name__ == "__main__":
    main()