#!/usr/bin/env python3

# ============================== CONFIG ==============================
PARQUET_DIR = "/home/jonas/Datasets/TikZ/sketch/parquet"
OUTPUT_DIR = "/home/jonas/Datasets/TikZ/sketch"
BENCHMARK_SIZE = 200
VAL_SIZE = 250
ULTRASKETCH_RATIO = 0.50
DISPLACEMENT_RATIO = 0.50
SEED = 42
NUM_WORKERS = 12
# ====================================================================

import csv
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from tikz_clean_pipeline.tikz_rendering import render_tex_to_png

SIZE = 512
os.environ["REF_IMAGE_SIZE"] = str(SIZE)
os.environ["LATEX_DPI"] = "400"


def has_description(value):
    return value is not None and str(value).strip() != ""


def load_data():
    files = list(Path(PARQUET_DIR).rglob("*.parquet"))
    benchmark = [p for p in files if "benchmark" in p.name.lower()]
    regular = [p for p in files if p not in benchmark]

    data_files = {
        "ultrasketch": [str(p) for p in regular if "ultrasketch" in str(p).lower()],
        "displacement": [str(p) for p in regular if "displacement" in str(p).lower()],
        "benchmark": [str(p) for p in benchmark],
    }

    if not all(data_files.values()):
        raise FileNotFoundError(
            "Ultrasketch-, Displacement- oder Benchmark-Parquet fehlt."
        )

    return load_dataset("parquet", data_files=data_files)


def validate_config():
    if ULTRASKETCH_RATIO <= 0 or DISPLACEMENT_RATIO < 0:
        raise ValueError("Ungültiges Ultrasketch-/Displacement-Verhältnis.")
    if BENCHMARK_SIZE < 0 or VAL_SIZE < 0 or NUM_WORKERS < 1:
        raise ValueError("Ungültige Splitgröße oder Worker-Anzahl.")


def select_train_val(ds, rng):
    ultra = ds["ultrasketch"]
    displacement = ds["displacement"]

    n_displacement = round(
        len(ultra) * DISPLACEMENT_RATIO / ULTRASKETCH_RATIO
    )
    if n_displacement > len(displacement):
        raise ValueError(
            f"Benötigt: {n_displacement} Displacement-Samples, "
            f"vorhanden: {len(displacement)}."
        )

    displacement_ids = rng.sample(range(len(displacement)), n_displacement)
    samples = [
        ("ultrasketch", "ultrasketch", i) for i in range(len(ultra))
    ] + [
        ("displacement", "displacement", i) for i in displacement_ids
    ]
    rng.shuffle(samples)

    if VAL_SIZE > len(samples):
        raise ValueError("Validation ist größer als der Train/Val-Datenpool.")

    val = rng.sample(samples, VAL_SIZE)
    val_set = set(val)
    train = [sample for sample in samples if sample not in val_set]
    return train, val


def select_benchmark(ds, rng):
    benchmark = ds["benchmark"]
    total_ratio = ULTRASKETCH_RATIO + DISPLACEMENT_RATIO
    n_ultra = round(BENCHMARK_SIZE * ULTRASKETCH_RATIO / total_ratio)
    n_displacement = BENCHMARK_SIZE - n_ultra

    candidates = {"ultrasketch": [], "displacement": []}
    for i, (method, description) in enumerate(
        zip(benchmark["sketch_method"], benchmark["description"])
    ):
        method = str(method).lower()
        if method in candidates and has_description(description):
            candidates[method].append(i)

    required = {
        "ultrasketch": n_ultra,
        "displacement": n_displacement,
    }
    for method, count in required.items():
        if len(candidates[method]) < count:
            raise ValueError(
                f"Benchmark benötigt {count} {method}-Samples mit Description, "
                f"vorhanden: {len(candidates[method])}."
            )

    samples = []
    for method, count in required.items():
        samples += [
            ("benchmark", method, i)
            for i in rng.sample(candidates[method], count)
        ]

    rng.shuffle(samples)
    return samples


def save_512(image, path):
    image = image.convert("RGBA")
    image.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), "white")
    canvas.alpha_composite(
        image,
        ((SIZE - image.width) // 2, (SIZE - image.height) // 2),
    )
    canvas.convert("RGB").save(path)


def prepare_output(root):
    for split in ("benchmark", "train", "val"):
        folder = root / split
        if folder.exists():
            shutil.rmtree(folder)
        for name in ("images", "references", "descriptions", "input_images"):
            (folder / name).mkdir(parents=True, exist_ok=True)


def write_sample(ds, sample, folder):
    dataset_split, method, index = sample
    row = ds[dataset_split][index]
    stem = f"{method}_{index:08d}"

    image_path = folder / "images" / f"{stem}.png"
    input_path = folder / "input_images" / f"{stem}.png"
    code_path = folder / "references" / f"{stem}.txt"
    description_path = folder / "descriptions" / f"{stem}.txt"

    save_512(row["image"], input_path)
    code_path.write_text(row["code"], encoding="utf-8")
    description_path.write_text(row["description"] or "", encoding="utf-8")

    render_tex_to_png(row["code"], image_path, create_ds=True)
    with Image.open(image_path) as image:
        save_512(image.copy(), image_path)

    return {
        "input_image": str(input_path.resolve()),
        "reference_image": str(image_path.resolve()),
        "reference_code": str(code_path.resolve()),
        "llm_description": str(description_path.resolve()),
        "source_variant": row["source_variant"],
        "sketch_method": row["sketch_method"],
        "image_path": str(image_path.resolve()),
        "input_image_path": str(input_path.resolve()),
        "code_path": str(code_path.resolve()),
        "vlm_description_path": str(description_path.resolve()),
    }


def process_split(pool, ds, samples, folder, name):
    worker = partial(write_sample, ds, folder=folder)
    return list(
        tqdm(pool.map(worker, samples), total=len(samples), desc=name)
    )


def write_manifest(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row[column] for column in columns} for row in rows
        )


def main():
    validate_config()
    rng = random.Random(SEED)
    ds = load_data()

    benchmark = select_benchmark(ds, rng)
    train, val = select_train_val(ds, rng)

    output = Path(OUTPUT_DIR).expanduser().resolve()
    prepare_output(output)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        benchmark_rows = process_split(
            pool, ds, benchmark, output / "benchmark", "benchmark"
        )
        train_rows = process_split(
            pool, ds, train, output / "train", "train"
        )
        val_rows = process_split(
            pool, ds, val, output / "val", "val"
        )

    write_manifest(
        output / "benchmark" / "manifest.csv",
        benchmark_rows,
        [
            "input_image",
            "reference_image",
            "reference_code",
            "llm_description",
            "source_variant",
            "sketch_method",
        ],
    )

    columns = [
        "image_path",
        "input_image_path",
        "code_path",
        "vlm_description_path",
    ]
    write_manifest(
        output / "train" / "manifest_train.csv",
        train_rows,
        columns,
    )
    write_manifest(
        output / "val" / "manifest_val.csv",
        val_rows,
        columns,
    )

    print(f"Benchmark: {len(benchmark)}")
    print(f"Validation: {len(val)}")
    print(f"Train: {len(train)}")
    print(
        "Displacements im Benchmark: "
        f"{sum(method == 'displacement' for _, method, _ in benchmark)}"
    )


if __name__ == "__main__":
    main()