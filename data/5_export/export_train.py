from __future__ import annotations

import random
from collections import Counter

from config import Config

from helpers import (
    choose_per_class,
    choose_uniform_with_noise_fallback,
    choose_weighted_with_noise_fallback,
    export_crystalbleu,
    export_split,
    find_files,
    is_noise,
    prepare_output,
    repetition_candidates,
    scan,
    train_class_names,
    without,
)


def _print_train_distribution(samples, noise_class: str) -> None:
    counts = Counter(sample.class_name for sample in samples)
    print("Train distribution:")
    for class_name, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {class_name}: {count:,}")

    noise = sum(
        count
        for class_name, count in counts.items()
        if class_name.casefold() == noise_class.casefold()
    )
    print(f"Train noise fallback: {noise:,}")


def run_train(config: Config) -> None:
    files = find_files(config.input_dir, "train")
    samples = scan(files, config.description_mix)
    rng = random.Random(config.seed)

    eligible = repetition_candidates(
        samples,
        config.valid_repetition_classes,
    )
    class_names = train_class_names(
        eligible,
        config.train_class_importance,
        config.noise_class,
    )

    val_keys = choose_per_class(
        samples=samples,
        class_names=class_names,
        count=config.val_samples_per_class,
        prefer_described=True,
        rng=rng,
        valid_repetition_classes=config.valid_repetition_classes,
    )
    after_val = without(samples, val_keys)

    # Train has priority over CrystalBLEU. Select it immediately after
    # validation so the importance-weighted classes are not consumed by
    # the metric reference corpus first.
    train_candidates = after_val

    if config.train_size is None:
        eligible_train = repetition_candidates(
            train_candidates,
            config.valid_repetition_classes,
        )
        allowed = {name.casefold() for name in class_names}
        train_keys = {
            sample.key
            for sample in eligible_train
            if (
                not is_noise(sample, config.noise_class)
                and sample.class_name.casefold() in allowed
            )
        }
    else:
        train_keys = choose_weighted_with_noise_fallback(
            samples=train_candidates,
            total=config.train_size,
            class_importance=config.train_class_importance,
            prefer_described=True,
            rng=rng,
            valid_repetition_classes=config.valid_repetition_classes,
            noise_class=config.noise_class,
            label="train",
        )

    # CrystalBLEU is built only from the remainder after validation + train.
    # It is deliberately NOT importance-weighted. Real non-noise classes are
    # sampled as uniformly as availability allows, preferring rows without
    # descriptions. Noise is used only for any final shortfall.
    after_train = without(train_candidates, train_keys)
    crystal_keys = choose_uniform_with_noise_fallback(
        samples=after_train,
        total=config.train_crystalbleu_size,
        prefer_described=False,
        rng=rng,
        valid_repetition_classes=config.valid_repetition_classes,
        noise_class=config.noise_class,
        label="CrystalBLEU",
    )

    val_samples = [sample for sample in samples if sample.key in val_keys]
    crystalbleu_samples = [
        sample for sample in samples if sample.key in crystal_keys
    ]
    train_samples = [sample for sample in samples if sample.key in train_keys]

    output = config.output_dir / "train"
    prepare_output(output, config.overwrite)

    export_split(
        files=files,
        samples=val_samples,
        output=output / "val",
        batch_size=config.batch_size,
        description_mix=config.description_mix,
        seed=config.seed + 101,
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
        description_mix=config.description_mix,
        seed=config.seed + 202,
    )

    _print_train_distribution(train_samples, config.noise_class)
    print(
        f"Train export complete: "
        f"{len(val_samples):,} validation, "
        f"{len(crystalbleu_samples):,} CrystalBLEU, "
        f"{len(train_samples):,} train"
    )
