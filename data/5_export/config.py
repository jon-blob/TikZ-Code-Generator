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

    benchmark_samples_per_class: int = 20
    val_samples_per_class: int = 20
    train_crystalbleu_size: int = 50_000
    train_size: int | None = None

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

    # Relative weights for the final exported description text.
    description_mix: Mapping[str, float] = field(
        default_factory=lambda: {"llm_description": 1.0}
    )


CONFIG = Config(
    input_dir=Path("../tikz-dataset-clean/data"),
    output_dir=Path("../tikz-dataset-clean/dataset-exported"),

    benchmark_samples_per_class=20,
    val_samples_per_class=10,
    train_crystalbleu_size=50_000,
    train_size=5_000,

    seed=42,
    overwrite=True,
    batch_size=2_048,
    noise_class="noise",

    valid_repetition_classes=[
        "low",
        "medium",
        "high",
    ],

    train_class_importance = {
        "*": "none",

        "class_1": "a bit important",
        "class_2": "not important",         # noted: possibly exclude
        "class_3": "a bit important",
        "class_4": "not important",      # annotation: definitions/theorems
        "class_5": "really important",
        "class_6": "not important",
        "class_7": "extremely important",
        "class_8": "not important",
        "class_9": "extremely important",
        "class_10": "not important",
        "class_11": "extremely important",
        "class_12": "not important",
        "class_13": "extremely important",
        "class_14": "extremely important",
        "class_15": "not important",
        "class_16": "not important",
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
        "class_29": "not important",
        "class_30": "a bit important",
        "class_31": "a bit important",
        "class_32": "a bit important",      # only arrows
        "class_33": "a bit important",      # only boxes
        "class_34": "a bit important",      # only lines
        "class_35": "not important",        # only black boxes
    },

    benchmark_classes=None,
    # Example:
    # benchmark_classes=["class_1", "class_4", "class_9"],

    description_mix={
        "llm_description_image": 1.0,
        # "llm_description": 0.5,
        # "llm_description_image": 0.5,
        # "llm_description_image_code": 0.0,
    },
)
