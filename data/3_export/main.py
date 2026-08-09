from pathlib import Path

from export_benchmark import run_benchmark
from export_train import run_train
from helpers import Config


CONFIG = Config(
    input_dir=Path("../dataset"),
    output_dir=Path("../dataset-exported"),
    benchmark_samples_per_class=20,
    val_samples_per_class=20,
    train_crystalbleu_size=50_000,
    train_size=100,
    balance_tolerance=0.10,
    seed=42,
    overwrite=True,
    batch_size=2_048,
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
