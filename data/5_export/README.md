# TikZ dataset export

Exports final TikZ Parquet shards into file-based `benchmark`, `val`, `train`, and CrystalBLEU datasets.

## Repetition classes

Define the repetition classes that are allowed anywhere in the export:

```python
valid_repetition_classes=[
    "low",
    "medium",
    "high",
]
```

Samples with any other `repetition_class` are excluded from train, validation, the selected benchmark subset, and their noise fallback. CrystalBLEU intentionally uses all repetition classes.

## Train image-class importance

Train and validation class selection is controlled independently from benchmark through `train_class_importance`.

The fixed importance ratios are:

```text
none                 = 0
not important        = 1
a bit important      = 5
really important     = 20
extremely important  = 50
```

Example:

```python
train_class_importance={
    "*": "none",
    "class_1": "not important",
    "class_2": "extremely important",
    "class_3": "really important",
}
```

With enough available samples, `class_1` and `class_2` therefore receive approximately a 1:50 target ratio. `none` excludes a class. `"*"` is the default for classes that are not explicitly listed.

For a fixed `train_size`, the exporter first computes a weighted target from these importance values. If a class does not contain enough eligible samples, its missing quota may only be transferred to classes with the same or higher importance. It is never transferred down to a less important class. Any remaining shortage is filled from random eligible `noise` samples.

This means a very large low-priority class cannot take over the training set simply because a high-priority class is small.

With `train_size=None`, every eligible non-noise sample from the active train classes is exported. Importance ratios only apply when a fixed target size is requested.

The same weighted allocation is used for a fixed-size train CrystalBLEU selection. CrystalBLEU prefers samples without descriptions; regular train/validation/benchmark selection prefers samples with a configured description.

## Independent benchmark classes

Benchmark does not reuse the train class list. Configure it separately:

```python
benchmark_classes=[
    "class_2",
    "class_7",
    "class_11",
]
```

Or use:

```python
benchmark_classes=None
```

to use every observed non-noise class in the benchmark split.

`benchmark_samples_per_class` remains a fixed target per requested real benchmark class. If a requested class has fewer real samples than this target, all available real samples are used and the missing slots are filled with random eligible `noise` samples. In addition, `benchmark_noise_samples` controls a separate explicit noise-class quota for benchmark only.

Example with `benchmark_samples_per_class=20`:

```text
class_1:  2 real samples -> 2 real + 18 random noise
class_2: 20 real samples -> 20 real + 0 noise
class_3:  1 real sample  -> 1 real + 19 random noise
```

Noise fallback samples are unique and are not reused between class deficits.

Train and benchmark are therefore fully independent: a train class does not need to exist in benchmark, and a benchmark class does not need to be active in train.

## Description selection

The exporter can use any mixture of:

```text
llm_description
llm_description_image
llm_description_image_code
```

For example:

```python
description_mix={
    "llm_description": 0.50,
    "llm_description_image": 0.50,
}
```

When selecting real samples, samples containing at least one configured description are preferred. After a sample has been selected, one of its available configured descriptions is chosen randomly using the configured relative weights.

If a selected sample has only one of the configured description types, that available description is used. If none exists, the exported description file is empty.

The manifest contains `llm_description_source`, which records the source description column selected for each sample.

Noise fallback is intentionally random and does not prioritize description availability.

## Configuration example

```python
CONFIG = Config(
    input_dir=Path("../tikz-dataset-clean/data"),
    output_dir=Path("../tikz-dataset-clean/dataset-exported"),

    benchmark_samples_per_class=20,
    benchmark_noise_samples=50,
    val_samples_per_class=10,
    val_noise_samples=50,
    train_crystalbleu_size=0,
    train_size=5000,

    seed=42,
    overwrite=True,
    batch_size=2_048,
    noise_class="noise",

    valid_repetition_classes=[
        "low",
        "medium",
        "high",
    ],

    train_class_importance={
        "*": "none",
        "class_1": "not important",
        "class_2": "extremely important",
        "class_3": "really important",
    },

    benchmark_classes=[
        "class_1",
        "class_2",
    ],

    description_mix={
        "llm_description": 0.50,
        "llm_description_image": 0.50,
    },
)
```

`extremly important` is also accepted as an alias for `extremely important`.

## Requirements

```bash
pip install pyarrow pillow tqdm
```

## Run

```bash
python main.py
```


## Configuration file

All user-editable export settings are centralized in `config.py`. In particular, `IMPORTANCE_WEIGHTS` controls the numeric ratios between `none`, `not important`, `a bit important`, `really important`, and `extremely important`. `CONFIG` controls paths, split sizes, valid repetition classes, per-class importance, benchmark classes, and the description mix.


## Current description and CrystalBLEU behavior

Train/validation and benchmark descriptions are configured independently with
`train_description_mix` and `benchmark_description_mix`.

`valid_repetition_classes` applies to validation, train, and the selected benchmark subset.
CrystalBLEU intentionally ignores this filter: all repetition classes, including
`very_high` and `critical`, are eligible. Train CrystalBLEU is selected from the
remainder after validation and train. Benchmark CrystalBLEU contains every remaining
benchmark row after the selected benchmark subset, regardless of image class or
repetition class.


## Explicit benchmark noise class

Benchmark can include `noise` as an explicit extra class without changing train behavior:

```python
benchmark_samples_per_class = 5
benchmark_noise_samples = 25
```

If 23 real benchmark image classes are active, the regular target is `23 * 5 = 115` rows.
The exporter then adds 25 explicit random eligible `noise` rows. Any noise rows needed to fill shortages of real benchmark classes are additional fallback rows and do not reduce the explicit 25-row noise quota. Train keeps its existing importance-weighted sampling and noise fallback unchanged.

## Separate RL export

Run:

```bash
python export_rl.py
```

The RL export reuses the SFT train settings (`train_size`, image-class importance,
description mix, noise fallback, seed/batch settings), but enforces this repetition
composition for the RL train split:

- 25% `low`
- 75% from the combined `high` + `critical` pool

The result is written to:

```text
dataset-exported/rl/
├── train/
│   ├── input_image/
│   ├── reference_image/
│   ├── llm_description/
│   ├── reference_code/
│   └── manifest.csv
└── crystalbleu/
```

`high` and `critical` are treated as one joint 75% pool; their internal ratio is
not fixed. CrystalBLEU is selected after RL train from the full remaining train
dataset and can use every repetition class.


## Explicit validation noise class

Validation now mirrors the benchmark sampling behavior for noise. Each active
real train class targets `val_samples_per_class` rows. If a real class contains
fewer eligible rows, its missing slots are filled with random, non-reused
eligible `noise` rows. In addition, `val_noise_samples` adds a separate explicit
noise quota for validation.

```python
val_samples_per_class = 10
val_noise_samples = 50
```

The fallback noise rows are additional to the explicit `val_noise_samples`
quota. Validation still uses `valid_repetition_classes`, prefers rows with a
configured train description, and is removed from the pool before the weighted
train selection. Train sampling itself is unchanged.
