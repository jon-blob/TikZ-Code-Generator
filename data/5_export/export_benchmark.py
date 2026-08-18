from __future__ import annotations

import random

from config import Config

from helpers import (
    benchmark_class_names,
    choose_benchmark_with_noise_fallback,
    export_crystalbleu,
    export_split,
    find_files,
    prepare_output,
    repetition_candidates,
    scan,
)


def run_benchmark(config: Config) -> None:
    files = find_files(config.input_dir, "benchmark")
    samples = scan(files, config.benchmark_description_mix)
    rng = random.Random(config.seed)

    eligible = repetition_candidates(
        samples,
        config.valid_repetition_classes,
    )
    class_names = benchmark_class_names(
        eligible,
        config.benchmark_classes,
        config.noise_class,
    )

    selected_keys = choose_benchmark_with_noise_fallback(
        samples=samples,
        class_names=class_names,
        count_per_class=config.benchmark_samples_per_class,
        noise_samples=config.benchmark_noise_samples,
        prefer_described=True,
        rng=rng,
        valid_repetition_classes=config.valid_repetition_classes,
        noise_class=config.noise_class,
    )

    selected = [sample for sample in samples if sample.key in selected_keys]

    # Benchmark CrystalBLEU uses the complete remaining benchmark dataset.
    # It is independent from benchmark_classes and from valid_repetition_classes,
    # so very_high/critical and image classes not selected for the benchmark
    # subset are retained as metric references.
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
        description_mix=config.benchmark_description_mix,
        seed=config.seed + 303,
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
