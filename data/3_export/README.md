# TikZ dataset export

Exports final TikZ Parquet shards into file-based `benchmark`, `val`, `train`, and CrystalBLEU datasets.

## Files

```text
main.py               Configuration and entry point
export_benchmark.py   Benchmark and benchmark CrystalBLEU export
export_train.py       Validation, train CrystalBLEU, and train export
helpers.py            Shared Parquet, sampling, balancing, and file helpers
```

## Requirements

```bash
pip install pyarrow pillow tqdm
```

or 

```bash
conda activate data-tools
```


## Input data

The Parquet files must contain:

```text
input_image
reference_image
llm_description
reference_code
class
source
type
```

The input may use separate folders:

```text
input/
├── train/
└── benchmark/
```

It may also use one folder when the filenames contain `train` or `benchmark`:

```text
input/
├── train-00000.parquet
└── benchmark-00000.parquet
```

Do not use generic filenames such as only `part-00000.parquet` in one shared folder, because the exporter cannot distinguish the splits.

## Configuration

Edit `CONFIG` in `main.py`:

```python
CONFIG = Config(
    input_dir=Path("/path/to/final/parquet/root"),
    output_dir=Path("/path/to/export"),
    benchmark_samples_per_class=20,
    val_samples_per_class=20,
    train_crystalbleu_size=50_000,
    train_size=100_000,
    balance_tolerance=0.10,
    seed=42,
    overwrite=False,
)
```

`train_size` limits the final train split to an absolute number of samples. Set it to `None` to export the largest possible balanced remainder.

`balance_tolerance=0.10` allows larger train classes to contain up to 10% more samples than the smallest available class.

## Run

```bash
python main.py
```

The benchmark export runs first, followed by the train export.

## Selection logic

### Benchmark

- Selects a fixed number of samples per class.
- Prefers samples with an LLM description.
- Uses every remaining benchmark sample as CrystalBLEU code.

### Train

- Selects a fixed number of validation samples per class.
- Prefers validation samples with an LLM description.
- Creates a fixed-size, class-balanced CrystalBLEU corpus.
- Prefers CrystalBLEU samples without an LLM description.
- Selects exactly `train_size` remaining samples for train when a size is configured.
- Keeps the final train classes as uniform as possible within the configured tolerance.
- With `train_size=None`, exports the largest possible balanced remainder.

All groups are disjoint.

## Output

```text
export/
├── benchmark/
│   ├── input_image/
│   ├── reference_image/
│   ├── llm_description/
│   ├── reference_code/
│   ├── manifest.csv
│   └── crystalbleu/
└── train/
    ├── val/
    ├── train/
    └── crystalbleu/
```

Each regular split contains both image folders, description files, code files, and `manifest.csv`.

Manifest paths are absolute. Missing descriptions are exported as empty `.txt` files. CrystalBLEU folders contain only TikZ/LaTeX code files.
