from __future__ import annotations

import csv
import random
from collections import Counter

from config import CONFIG, Config
from helpers import (
    choose_weighted_with_noise_fallback,
    export_split,
    find_files,
    prepare_output,
    scan,
    without,
)


# RL-specific repetition mix.
# Exactly 25% of the requested RL train size comes from `low`.
# The remaining 75% comes from the joint pool of `high` + `critical`.
RL_LOW_FRACTION = 0.25
RL_LOW_REPETITIONS = ("low",)
RL_HARD_REPETITIONS = ("high", "critical")



def _sort_manifest_by_repetition(manifest_path) -> None:
    # Curriculum order for RL: easiest repetition examples first, then harder ones.
    repetition_order = {
        "low": 0,
        "high": 1,
        "critical": 2,
    }

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        return

    # Python's sort is stable, so the original order is preserved inside each bucket.
    rows.sort(
        key=lambda row: repetition_order.get(
            str(row.get("repetition_class", "")),
            len(repetition_order),
        )
    )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _print_distribution(samples) -> None:
    repetition_counts = Counter(sample.repetition_class for sample in samples)
    class_counts = Counter(sample.class_name for sample in samples)

    print("RL repetition distribution:")
    for name, count in sorted(repetition_counts.items()):
        print(f"  {name}: {count:,}")

    print("RL image-class distribution:")
    for name, count in sorted(
        class_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {name}: {count:,}")


def run_rl(config: Config) -> None:
    if config.rl_train_size is None or config.rl_train_size <= 0:
        raise ValueError(
            "RL export requires CONFIG.rl_train_size to be a positive integer"
        )

    if not 0.0 <= RL_LOW_FRACTION <= 1.0:
        raise ValueError("RL_LOW_FRACTION must be between 0 and 1")

    files = find_files(config.input_dir, "train")
    samples = scan(files, config.train_description_mix)
    rng = random.Random(config.seed + 30_000)

    total = config.rl_train_size
    low_target = round(total * RL_LOW_FRACTION)
    hard_target = total - low_target

    # Same sampling logic as SFT inside each repetition bucket:
    # - same image-class importance mapping
    # - described samples preferred
    # - random noise only as final fallback
    low_keys = choose_weighted_with_noise_fallback(
        samples=samples,
        total=low_target,
        class_importance=config.train_class_importance,
        prefer_described=True,
        rng=rng,
        valid_repetition_classes=RL_LOW_REPETITIONS,
        noise_class=config.noise_class,
        label="RL low",
    )

    after_low = without(samples, low_keys)

    hard_keys = choose_weighted_with_noise_fallback(
        samples=after_low,
        total=hard_target,
        class_importance=config.train_class_importance,
        prefer_described=True,
        rng=rng,
        valid_repetition_classes=RL_HARD_REPETITIONS,
        noise_class=config.noise_class,
        label="RL high+critical",
    )

    train_keys = low_keys | hard_keys
    train_samples = [sample for sample in samples if sample.key in train_keys]

    if len(train_samples) != total:
        raise RuntimeError(
            f"RL selection produced {len(train_samples):,} samples, "
            f"expected exactly {total:,}"
        )

    # RL only exports its train split. CrystalBLEU is reused from the SFT export.
    output = config.rl_output_dir
    prepare_output(output, config.overwrite)

    train_output = output / "train"
    export_split(
        files=files,
        samples=train_samples,
        output=train_output,
        batch_size=config.batch_size,
        description_mix=config.train_description_mix,
        seed=config.seed + 30_101,
    )

    _sort_manifest_by_repetition(train_output / "manifest.csv")
    print("RL manifest order: low -> high -> critical")

    _print_distribution(train_samples)
    print(
        "RL export complete: "
        f"{len(train_samples):,} train "
        f"({low_target:,} low + {hard_target:,} high/critical)"
    )


if __name__ == "__main__":
    run_rl(CONFIG)
