from config import CONFIG
from export_benchmark import run_benchmark
from export_train import run_train
from helpers import (
    active_description_mix,
    validate_class_importance,
    valid_repetitions,
)


def main() -> None:
    if CONFIG.train_size is not None and CONFIG.train_size <= 0:
        raise ValueError("train_size must be positive or None")

    if CONFIG.train_crystalbleu_size < 0:
        raise ValueError("train_crystalbleu_size must be >= 0")

    if CONFIG.benchmark_samples_per_class < 0:
        raise ValueError("benchmark_samples_per_class must be >= 0")

    if CONFIG.benchmark_noise_samples < 0:
        raise ValueError("benchmark_noise_samples must be >= 0")

    if CONFIG.val_samples_per_class < 0:
        raise ValueError("val_samples_per_class must be >= 0")

    if CONFIG.val_noise_samples < 0:
        raise ValueError("val_noise_samples must be >= 0")

    active_description_mix(CONFIG.train_description_mix)
    active_description_mix(CONFIG.benchmark_description_mix)
    valid_repetitions(CONFIG.valid_repetition_classes)
    validate_class_importance(CONFIG.train_class_importance)

    run_benchmark(CONFIG)
    run_train(CONFIG)


if __name__ == "__main__":
    main()
