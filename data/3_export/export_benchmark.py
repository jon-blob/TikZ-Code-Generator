from __future__ import annotations

import random

from helpers import (
    Config,
    choose_per_class,
    export_crystalbleu,
    export_split,
    find_files,
    prepare_output,
    scan,
)


def run_benchmark(config: Config) -> None:
    files = find_files(config.input_dir, "benchmark")
    samples = scan(files)
    rng = random.Random(config.seed)

    selected_keys = choose_per_class(
        samples=samples,
        count=config.benchmark_samples_per_class,
        prefer_described=True,
        rng=rng,
    )

    selected = [
        sample
        for sample in samples
        if sample.key in selected_keys
    ]
    crystalbleu = [
        sample
        for sample in samples
        if sample.key not in selected_keys
    ]

    output = config.output_dir / "benchmark"
    prepare_output(output, config.overwrite)

    export_split(
        files=files,
        samples=selected,
        output=output,
        batch_size=config.batch_size,
    )
    export_crystalbleu(
        files=files,
        samples=crystalbleu,
        output=output / "crystalbleu",
        batch_size=config.batch_size,
    )

    print(
        f"Benchmark export complete: "
        f"{len(selected):,} selected, "
        f"{len(crystalbleu):,} CrystalBLEU"
    )
