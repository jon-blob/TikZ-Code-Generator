from pathlib import Path

from export_benchmark import run_benchmark
from export_train import run_train
from helpers import Config


CONFIG = Config(
    input_dir=Path("../preprocessed/huggingface/data"),
    output_dir=Path("../preprocessed/dataset-exported"),
    benchmark_samples_per_class=0,
    val_samples_per_class=0,
    train_crystalbleu_size=0,
    train_size=500,
    balance_tolerance=100.0,
    seed=42,
    overwrite=True,
    batch_size=2_048,
    only_low=True,
)


def main() -> None:
    if CONFIG.balance_tolerance < 0:
        raise ValueError("balance_tolerance must be >= 0")

    if CONFIG.train_size is not None and CONFIG.train_size <= 0:
        raise ValueError("train_size must be positive or None")

    run_benchmark(CONFIG)
    run_train(CONFIG)


if __name__ == "__main__":
    main()
