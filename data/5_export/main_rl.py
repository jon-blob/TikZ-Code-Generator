from __future__ import annotations

import csv
import math
import random
from collections import Counter

from config import CONFIG, Config
from helpers import (
    choose_weighted_with_noise_fallback,
    export_split,
    find_files,
    is_noise,
    prepare_output,
    repetition_candidates,
    scan,
    without,
)


# RL-specific repetition mix.
# Exactly 25% of the requested RL train size comes from `low`.
# The remaining 75% comes from the joint pool of `high` + `critical`.
RL_LOW_FRACTION = 0.25
RL_LOW_REPETITIONS = ("low",)
RL_HARD_REPETITIONS = ("high", "critical")



def _allocate_counts(total: int, mix) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be >= 0")
    if total == 0:
        return {str(name): 0 for name in mix}

    cleaned = {str(name): float(weight) for name, weight in mix.items()}
    if not cleaned:
        raise ValueError(
            "CONFIG.rl_noise_repetition_mix must not be empty when "
            "CONFIG.rl_noise_samples > 0"
        )
    if any(weight < 0 for weight in cleaned.values()):
        raise ValueError("RL noise repetition percentages must be >= 0")

    weight_sum = sum(cleaned.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "CONFIG.rl_noise_repetition_mix must sum to 1.0; "
            f"got {weight_sum:.12g}"
        )

    raw = {name: total * weight for name, weight in cleaned.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())

    # Largest-remainder allocation guarantees that the integer quotas sum
    # exactly to rl_noise_samples while staying as close as possible to the
    # configured percentages.
    order = sorted(
        cleaned,
        key=lambda name: (-(raw[name] - counts[name]), name),
    )
    for name in order[:remainder]:
        counts[name] += 1

    return counts


def _sort_manifest_by_repetition(manifest_path, noise_class: str) -> None:
    # Curriculum order for RL: real low examples first, then high/critical.
    # Explicit/fallback noise rows are kept at the end, independent of their
    # configured repetition class.
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
        key=lambda row: (
            str(row.get("class", "")).casefold() == noise_class.casefold(),
            repetition_order.get(
                str(row.get("repetition_class", "")).strip().lower(),
                len(repetition_order),
            ),
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

    # Explicit RL noise is independent from the 25%/75% curriculum mix.
    # It is added on top of rl_train_size. Its repetition-class composition is
    # controlled independently through rl_noise_repetition_mix.
    base_keys = low_keys | hard_keys
    remaining = without(samples, base_keys)

    if config.rl_noise_samples < 0:
        raise ValueError("CONFIG.rl_noise_samples must be >= 0")

    noise_targets = _allocate_counts(
        config.rl_noise_samples,
        config.rl_noise_repetition_mix,
    )
    explicit_noise_keys = set()

    for repetition_class, target in noise_targets.items():
        if target == 0:
            continue

        candidates = [
            sample
            for sample in repetition_candidates(
                remaining,
                (repetition_class,),
            )
            if is_noise(sample, config.noise_class)
            and sample.key not in explicit_noise_keys
        ]

        if target > len(candidates):
            raise ValueError(
                f"RL explicit noise requires {target:,} {repetition_class!r} "
                f"samples, but only {len(candidates):,} unused samples are "
                "available for that repetition class."
            )

        rng.shuffle(candidates)
        explicit_noise_keys.update(
            sample.key for sample in candidates[:target]
        )

    if len(explicit_noise_keys) != config.rl_noise_samples:
        raise RuntimeError(
            f"RL explicit noise selected {len(explicit_noise_keys):,} samples, "
            f"expected {config.rl_noise_samples:,}."
        )

    train_keys = base_keys | explicit_noise_keys
    train_samples = [sample for sample in samples if sample.key in train_keys]

    expected_total = total + config.rl_noise_samples
    if len(train_samples) != expected_total:
        raise RuntimeError(
            f"RL selection produced {len(train_samples):,} samples, "
            f"expected exactly {expected_total:,} "
            f"({total:,} curriculum + {config.rl_noise_samples:,} explicit noise)"
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

    _sort_manifest_by_repetition(
        train_output / "manifest.csv",
        config.noise_class,
    )
    print("RL manifest order: low -> high -> critical -> noise")

    if config.rl_noise_samples:
        print("RL explicit noise repetition targets:")
        for name, count in noise_targets.items():
            print(f"  {name}: {count:,}")

    _print_distribution(train_samples)
    print(
        "RL export complete: "
        f"{len(train_samples):,} train "
        f"({low_target:,} low + {hard_target:,} high/critical "
        f"+ {config.rl_noise_samples:,} explicit noise)"
    )


if __name__ == "__main__":
    run_rl(CONFIG)
