"""Visible entry point for the complete pipeline."""

import config

config.apply_environment()

from pipeline import TikzDatasetPipeline  # noqa: E402


def main() -> None:
    pipe = TikzDatasetPipeline()
    pipe.check_dependencies()

    # For each split:
    # 1. Download the split once.
    # 2. Build one deterministic random sample order.
    # 3. Run every configured benchmark/train mode independently.
    # 4. For train=True, replace rejected rows until absolute_num is reached.
    # 5. Write one separate Parquet series per mode.
    # 6. Delete the split cache and continue with the next split.
    for split_name, mode_plans in config.SPLITS.items():
        pipe.run_split(split_name, mode_plans)


if __name__ == "__main__":
    main()
