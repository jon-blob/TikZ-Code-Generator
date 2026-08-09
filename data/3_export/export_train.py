from __future__ import annotations

import random

from helpers import (
    Config,
    balance_remainder,
    balanced_counts,
    choose_counts,
    choose_per_class,
    export_crystalbleu,
    export_split,
    find_files,
    group_by_class,
    prepare_output,
    scan,
    without,
)


def run_train(config: Config) -> None:
    files = find_files(config.input_dir, "train")
    samples = scan(files)
    rng = random.Random(config.seed)

    val_keys = choose_per_class(
        samples=samples,
        count=config.val_samples_per_class,
        prefer_described=True,
        rng=rng,
    )
    after_val = without(samples, val_keys)

    crystal_counts = balanced_counts(
        groups=group_by_class(after_val),
        total=config.train_crystalbleu_size,
        tolerance=config.balance_tolerance,
        label="CrystalBLEU",
    )
    crystal_keys = choose_counts(
        groups=group_by_class(after_val),
        counts=crystal_counts,
        prefer_described=False,
        rng=rng,
    )

    train_candidates = without(after_val, crystal_keys)

    if config.train_size is None:
        train_keys = balance_remainder(
            samples=train_candidates,
            tolerance=config.balance_tolerance,
            rng=rng,
        )
    else:
        train_groups = group_by_class(train_candidates)
        train_counts = balanced_counts(
            groups=train_groups,
            total=config.train_size,
            tolerance=config.balance_tolerance,
            label="train",
        )
        train_keys = choose_counts(
            groups=train_groups,
            counts=train_counts,
            prefer_described=None,
            rng=rng,
        )

    val_samples = [
        sample
        for sample in samples
        if sample.key in val_keys
    ]
    crystalbleu_samples = [
        sample
        for sample in samples
        if sample.key in crystal_keys
    ]
    train_samples = [
        sample
        for sample in samples
        if sample.key in train_keys
    ]

    output = config.output_dir / "train"
    prepare_output(output, config.overwrite)

    export_split(
        files=files,
        samples=val_samples,
        output=output / "val",
        batch_size=config.batch_size,
    )
    export_crystalbleu(
        files=files,
        samples=crystalbleu_samples,
        output=output / "crystalbleu",
        batch_size=config.batch_size,
    )
    export_split(
        files=files,
        samples=train_samples,
        output=output / "train",
        batch_size=config.batch_size,
    )

    print(
        f"Train export complete: "
        f"{len(val_samples):,} validation, "
        f"{len(crystalbleu_samples):,} CrystalBLEU, "
        f"{len(train_samples):,} train"
    )