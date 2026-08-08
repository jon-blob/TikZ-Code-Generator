from math import ceil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm


# =========================
# CONFIG
# =========================
REPO_ID = "loss-boss/tikz-train"
OUTPUT_DIR = Path("../data")

TRAIN_PERCENT = 90
VAL_PERCENT = 10
CRYSTALBLEU_SIZE = 50_000
SAMPLES_PER_SHARD = 10_000

RANDOM_SEED = 42
PARQUET_COMPRESSION = "zstd"


SOURCE_SCHEMAS = (
    {
        "type": "with_text",
        "images": "image_with_text",
        "code": "code_with_text",
        "llm_description": "llm_description_with_text",
    },
    {
        "type": "without_text",
        "images": "image_without_text_full",
        "code": "code_without_text_full",
        "llm_description": "llm_description_without_text_full",
    },
)

OUTPUT_COLUMNS = [
    "images",
    "code",
    "llm_description",
    "type",
]

IMAGE_TYPE = pa.struct(
    [
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ]
)

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("images", IMAGE_TYPE, nullable=False),
        pa.field("code", pa.string(), nullable=False),
        pa.field("llm_description", pa.string()),
        pa.field("type", pa.string(), nullable=False),
    ]
)


def load_data() -> pd.DataFrame:
    dataset_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="*.parquet",
        )
    )

    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    parts = []

    for file in tqdm(
        parquet_files,
        desc="Loading Parquet files",
        unit="file",
    ):
        available_columns = set(pq.read_schema(file).names)
        loaded_rows = 0

        for source in SOURCE_SCHEMAS:
            source_columns = [
                source["images"],
                source["code"],
                source["llm_description"],
            ]

            if not set(source_columns) <= available_columns:
                continue

            part = pd.read_parquet(
                file,
                columns=source_columns,
            )

            part = part.rename(
                columns={
                    source["images"]: "images",
                    source["code"]: "code",
                    source["llm_description"]: "llm_description",
                }
            )

            # Ignore rows where this complete source schema is unused.
            active_rows = part[
                ["images", "code", "llm_description"]
            ].notna().any(axis=1)

            part = part.loc[active_rows].copy()

            if part.empty:
                continue

            invalid_rows = (
                part["images"].isna()
                | part["code"].isna()
            )

            if invalid_rows.any():
                raise ValueError(
                    f"{file.name} contains "
                    f"{invalid_rows.sum():,} active "
                    f"{source['type']} rows without image or code."
                )

            # Preserve missing descriptions as Parquet null values.
            part["llm_description"] = part[
                "llm_description"
            ].where(
                part["llm_description"].notna(),
                None,
            )

            part["type"] = source["type"]
            part = part[OUTPUT_COLUMNS]

            parts.append(part)
            loaded_rows += len(part)

        if loaded_rows:
            tqdm.write(
                f"Loaded: {file.name} "
                f"({loaded_rows:,} rows)"
            )
        else:
            tqdm.write(
                f"Skipped: {file.name} "
                "(no supported rows found)"
            )

    if not parts:
        raise RuntimeError("No compatible Parquet data found.")

    dataframe = pd.concat(
        parts,
        ignore_index=True,
    )

    if dataframe["images"].isna().any():
        raise ValueError("The merged dataset contains missing images.")

    if dataframe["code"].isna().any():
        raise ValueError("The merged dataset contains missing code.")

    return dataframe


def create_splits(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TRAIN_PERCENT + VAL_PERCENT != 100:
        raise ValueError(
            "TRAIN_PERCENT + VAL_PERCENT must equal 100."
        )

    missing_description = (
        dataframe["llm_description"].isna()
        | dataframe["llm_description"]
        .astype("string")
        .str.strip()
        .eq("")
    )

    corpus_candidates = dataframe[missing_description]

    if CRYSTALBLEU_SIZE > len(corpus_candidates):
        raise ValueError(
            f"CRYSTALBLEU_SIZE is {CRYSTALBLEU_SIZE:,}, "
            f"but only {len(corpus_candidates):,} rows "
            "without descriptions are available."
        )

    corpus = corpus_candidates.sample(
        n=CRYSTALBLEU_SIZE,
        random_state=RANDOM_SEED,
    )

    remaining = dataframe.drop(
        index=corpus.index
    ).sample(
        frac=1,
        random_state=RANDOM_SEED,
    )

    train_end = int(
        len(remaining) * TRAIN_PERCENT / 100
    )

    train = remaining.iloc[:train_end]
    val = remaining.iloc[train_end:]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        corpus.reset_index(drop=True),
    )


def export_shards(
    dataframe: pd.DataFrame,
    split: str,
) -> None:
    split_dir = OUTPUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale shards from previous runs.
    for existing_file in split_dir.glob("*.parquet"):
        existing_file.unlink()

    shard_count = ceil(
        len(dataframe) / SAMPLES_PER_SHARD
    )

    for shard_index in tqdm(
        range(shard_count),
        desc=f"Writing {split}",
        unit="shard",
    ):
        start = shard_index * SAMPLES_PER_SHARD
        end = start + SAMPLES_PER_SHARD

        shard = dataframe.iloc[start:end]

        table = pa.Table.from_pandas(
            shard,
            schema=OUTPUT_SCHEMA,
            preserve_index=False,
            safe=True,
        )

        output_file = split_dir / (
            f"{split}-{shard_index:05d}"
            f"-of-{shard_count:05d}.parquet"
        )

        pq.write_table(
            table,
            output_file,
            compression=PARQUET_COMPRESSION,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_data()
    train, val, corpus = create_splits(dataframe)

    print(
        f"Train: {len(train):,} | "
        f"Validation: {len(val):,} | "
        f"CrystalBLEU: {len(corpus):,}"
    )

    export_shards(train, "train")
    export_shards(val, "val")
    export_shards(
        corpus,
        "crystalbleu-corpus",
    )

    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()