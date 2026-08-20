from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


# Relative sampling weights for train classes.
# Example: 1 vs 50 means an "extremely important" class is targeted
# roughly 50x as often as a "not important" class, if enough samples exist.
IMPORTANCE_WEIGHTS = {
    "none": 0.0,
    "not important": 1.0,
    "a bit important": 5.0,
    "really important": 20.0,
    "extremely important": 50.0,
}


@dataclass(frozen=True, slots=True)
class Config:
    input_dir: Path
    output_dir: Path
    rl_output_dir: Path

    benchmark_samples_per_class: int = 20
    benchmark_noise_samples: int = 0
    val_samples_per_class: int = 20
    train_crystalbleu_size: int = 50_000
    train_size: int | None = None
    rl_train_size: int | None = None

    seed: int = 42
    overwrite: bool = False
    batch_size: int = 2_048
    noise_class: str = "noise"

    # Only these repetition classes are eligible anywhere in the export.
    valid_repetition_classes: Sequence[str] = ("low", "medium")

    # Train/validation class importance. "*" is the default for classes
    # that are not explicitly listed. Use "*": "none" for a whitelist.
    train_class_importance: Mapping[str, str] = field(
        default_factory=lambda: {"*": "really important"}
    )

    # Benchmark is independent from train.
    # None = all observed non-noise benchmark classes.
    benchmark_classes: Sequence[str] | None = None

    # Description selection is independent for train/validation and benchmark.
    train_description_mix: Mapping[str, float] = field(
        default_factory=lambda: {"llm_description": 1.0}
    )
    benchmark_description_mix: Mapping[str, float] = field(
        default_factory=lambda: {"llm_description": 1.0}
    )


CONFIG = Config(
    input_dir=Path("/home/jonas/Datasets/TikZ/tikz-dataset-clean/data"),
    output_dir=Path("/home/jonas/Datasets/TikZ/tikz-dataset-clean/sft-train"),
    rl_output_dir=Path("/home/jonas/Datasets/TikZ/tikz-dataset-clean/rl-train"),

    benchmark_samples_per_class=5,
    benchmark_noise_samples=50,
    val_samples_per_class=10,
    train_crystalbleu_size=50_000,
    train_size=200_000,
    rl_train_size=5000,

    seed=42,
    overwrite=True,
    batch_size=2_048,
    noise_class="noise",

    valid_repetition_classes=[
        "low",
        "medium",
    ],

    train_class_importance = {
        "*": "none",

        "class_1": "a bit important",
        "class_2": "none",         # noted: possibly exclude
        "class_3": "none",
        "class_4": "not important",      # annotation: definitions/theorems
        "class_5": "really important",
        "class_6": "not important",
        "class_7": "extremely important",
        "class_8": "not important",
        "class_9": "extremely important",
        "class_10": "not important",
        "class_11": "extremely important",
        "class_12": "none",
        "class_13": "extremely important",
        "class_14": "extremely important",
        "class_15": "not important",
        "class_16": "none",
        "class_17": "a bit important",
        "class_18": "really important",     # note says possibly extremely important
        "class_19": "extremely important",
        "class_20": "extremely important",
        "class_21": "not important",
        "class_22": "not important",
        "class_23": "none",                 # noted: possibly exclude
        "class_24": "none",                 # noted: possibly exclude
        "class_25": "extremely important",
        "class_26": "extremely important",
        "class_27": "none",                 # noted: possibly exclude
        "class_28": "none",                 # noted: possibly exclude
        "class_29": "none",
        "class_30": "a bit important",
        "class_31": "a bit important",
        "class_32": "a bit important",      # only arrows
        "class_33": "a bit important",      # only boxes
        "class_34": "a bit important",      # only lines
        "class_35": "not important",        # only black boxes
    },

    benchmark_classes=[
        "class_1",
        "class_4",
        "class_5",
        "class_6",
        "class_7",
        "class_8",
        "class_9",
        "class_10",
        "class_11",
        "class_13",
        "class_14",
        "class_15",
        "class_17",
        "class_18",
        "class_19",
        "class_20",
        "class_21",
        "class_22",
        "class_25",
        "class_26",
        "class_30",
        "class_31",
        "class_32",
        "class_33",
        "class_34",
        "class_35",
    ],

    train_description_mix={
        "llm_description_image": 0.5,
        "llm_description": 0.5,
        # "llm_description_image_code": 0.0,
    },

    benchmark_description_mix={
        "llm_description_image": 1.0,
        # "llm_description_image_code": 0.0,
    },
)
