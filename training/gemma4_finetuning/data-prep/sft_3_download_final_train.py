from io import BytesIO
from pathlib import Path
import csv

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm


# =========================
# CONFIG
# =========================
REPO_ID = "loss-boss/tikz-train-split"
OUTPUT_DIR = Path("../data")
BATCH_SIZE = 256


def save_image(value, path):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = bytes(value["bytes"])
        else:
            value = value["path"]

    if isinstance(value, (bytes, bytearray, memoryview)):
        value = BytesIO(bytes(value))

    with Image.open(value) as image:
        image.convert("RGB").save(path, "PNG")


def find_files(dataset_dir, split):
    names = {
        "train": {"train"},
        "val": {"val"},
        "crystalbleu": {"crystalbleu", "crystalbleu-corpus"},
    }[split]

    files = [
        file
        for file in dataset_dir.rglob("*.parquet")
        if names.intersection(file.relative_to(dataset_dir).parts)
    ]

    if not files:
        raise FileNotFoundError(f"No Parquet files found for split: {split}")

    return sorted(files)


def export_split(dataset_dir, split):
    files = find_files(dataset_dir, split)
    split_dir = OUTPUT_DIR / split

    for folder in ("images", "references", "descriptions"):
        (split_dir / folder).mkdir(parents=True, exist_ok=True)

    total = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    manifest_path = OUTPUT_DIR / f"manifest_{split}.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.writer(manifest)
        writer.writerow(
            ["image_path", "code_path", "vlm_description_path"]
        )

        index = 0

        with tqdm(total=total, desc=f"Exporting {split}") as progress:
            for file in files:
                parquet = pq.ParquetFile(file)

                for batch in parquet.iter_batches(
                    batch_size=BATCH_SIZE,
                    columns=["images", "code", "llm_description"],
                ):
                    for sample in batch.to_pylist():
                        if sample["images"] is None or sample["code"] is None:
                            raise ValueError(
                                f"Missing image or code in {split}, row {index}"
                            )

                        name = f"{index:08d}"

                        image_path = Path(split, "images", f"{name}.png")
                        code_path = Path(split, "references", f"{name}.tex")
                        description_path = Path(
                            split, "descriptions", f"{name}.txt"
                        )

                        save_image(
                            sample["images"],
                            OUTPUT_DIR / image_path,
                        )

                        (OUTPUT_DIR / code_path).write_text(
                            sample["code"],
                            encoding="utf-8",
                        )

                        (OUTPUT_DIR / description_path).write_text(
                            sample["llm_description"] or "",
                            encoding="utf-8",
                        )

                        writer.writerow(
                            [
                                image_path.as_posix(),
                                code_path.as_posix(),
                                description_path.as_posix(),
                            ]
                        )

                        index += 1
                        progress.update()


def export_crystalbleu(dataset_dir):
    files = find_files(dataset_dir, "crystalbleu")
    corpus_dir = OUTPUT_DIR / "crystalbleu-corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    total = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    index = 0

    with tqdm(total=total, desc="Exporting crystalbleu") as progress:
        for file in files:
            parquet = pq.ParquetFile(file)

            for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE,
                columns=["code"],
            ):
                for sample in batch.to_pylist():
                    if sample["code"] is None:
                        raise ValueError(
                            f"Missing code in crystalbleu, row {index}"
                        )

                    (corpus_dir / f"{index:08d}.txt").write_text(
                        sample["code"],
                        encoding="utf-8",
                    )

                    index += 1
                    progress.update()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Completes the entire download before export starts.
    dataset_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=["*.parquet", "**/*.parquet"],
        )
    )

    export_split(dataset_dir, "train")
    export_split(dataset_dir, "val")
    export_crystalbleu(dataset_dir)

    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()