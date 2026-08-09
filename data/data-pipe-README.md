# TikZ data pipeline

This package contains the complete workflow for preparing, inspecting, and exporting the TikZ dataset.

The required execution order is:

```text
Prepare the environment and input datasets
        ↓
Run `2_preprocess`
        ↓
Inspect the result with `1_analyze` if needed
        ↓
Run `3_export`
```

`1_analyze` is optional. The preprocessing stage must finish before the export stage is started.

## Package structure

```text
data-pipe/
├── README.md
├── environment.yaml
├── install_texlive_full.sh
├── 1_analyze/
│   ├── classify_dataset_clustering_clip_siglip_siglip2.ipynb
│   └── inspect_tikz_classes.ipynb
├── 2_preprocess/
│   ├── README.md
│   ├── code/
│   │   ├── config.py
│   │   ├── enrichment.py
│   │   ├── first_stage_cleaner.py
│   │   ├── output.py
│   │   ├── pipeline.py
│   │   ├── second_stage_cleaner.py
│   │   └── tikz_rendering.py
│   └── prompts/
│       └── description_prompt.txt
└── 3_export/
    ├── README.md
    ├── main.py
    ├── export_benchmark.py
    ├── export_train.py
    └── helpers.py
```

## 1. Prepare the environment

Create and activate the Conda environment from the package root:

```bash
conda env create -f environment.yaml
conda activate data-tools
```

The environment includes the Python packages used for Parquet processing, rendering support, embeddings, PCA, clustering, plotting, and export.

### Install TeX Live

The rendering pipeline requires a complete LaTeX installation.

On a supported Debian- or Ubuntu-based system, run:

```bash
sudo sh install_texlive_full.sh
```

The script installs the full TeX Live 2026 distribution under:

```text
/usr/local/texlive/2026
```

The installation is large and requires sufficient disk space.

### Install Ollama and the description model

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Download and test the configured Qwen coder model:

```bash
ollama run qwen3-coder:30b-a3b-q4_K_M
```

The Ollama service must be running while descriptions are generated.

### Prepare a local tokenizer

The preprocessing pipeline filters TikZ code by token length and therefore requires a local tokenizer.

Download or copy a compatible tokenizer and set its path in:

```text
2_preprocess/code/config.py
```

Example:

```python
TOKENIZER_PATH = Path("/path/to/local/tokenizer")
```

### Download the source datasets

Install or use the Hugging Face CLI and download both input datasets into separate directories:

```bash
hf download nllg/DaTikZ-V4 \
  --repo-type dataset \
  --local-dir ./dataset/DaTikZ-V4
```

```bash
hf download loss-boss/tikz-benchmark \
  --repo-type dataset \
  --local-dir ./dataset/tikz-benchmark
```

The two input datasets must remain separate during preprocessing.

## 2. Configure preprocessing

Edit:

```text
2_preprocess/code/config.py
```

At minimum, verify these paths:

```python
DATIKZ_DIR = Path("/path/to/dataset/DaTikZ-V4")
BENCHMARK_DIR = Path("/path/to/dataset/tikz-benchmark")
OUTPUT_DIR = Path("/path/to/preprocessed")
TOKENIZER_PATH = Path("/path/to/local/tokenizer")
PROMPT_PATH = Path("../prompts/description_prompt.txt")
```

Also verify the rendering and enrichment settings:

```python
LATEX_BIN_DIR = Path("/usr/local/texlive/2026/bin/x86_64-linux")
N_CLUSTERS = 4
PCA_COMPONENTS = 50
DESCRIPTIONS_PER_CLASS = 300
OLLAMA_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
```

The execution switches control which stages run:

```python
RUN_PREPROCESSING = True
RUN_ENRICHMENT = True
RUN_EXPORT = True
RUN_UPLOAD = False
```

Set `RUN_UPLOAD = True` only when the final Parquet dataset should be uploaded to Hugging Face.

Do not store a Hugging Face access token directly in a committed configuration file. Authenticate separately or provide the token through a secure local configuration.

### Disk-space preparation

Rendering, embeddings, staging Parquet files, reports, failures, and final shards can require substantial disk space.

Make sure that these locations point to a filesystem with enough free space:

```python
OUTPUT_DIR
CACHE_DIR
STAGING_DIR
EXPORT_DIR
```

## 3. Run preprocessing

Run the preprocessing pipeline before the standalone export scripts:

```bash
cd 2_preprocess/code
python pipeline.py
```

The pipeline performs the stages in this order:

```text
Load and independently shuffle DaTikZ-V4 and benchmark data
        ↓
Filter benchmark dates and optional origins
        ↓
Remove within-dataset duplicates
        ↓
Remove DaTikZ samples duplicated in benchmark
        ↓
Filter code by token length
        ↓
Render and validate TikZ samples
        ↓
Compute image embeddings and similarity reports
        ↓
Fit shared PCA and MiniBatchKMeans models
        ↓
Assign class labels
        ↓
Generate selected LLM descriptions
        ↓
Write final train and benchmark Parquet shards
        ↓
Optionally upload the dataset
```

The final Parquet files are written to:

```python
EXPORT_DIR = OUTPUT_DIR / "huggingface"
```

They should have names similar to:

```text
train-00000.parquet
train-00001.parquet
benchmark-00000.parquet
benchmark-00001.parquet
```

The final files must contain:

```text
input_image
reference_image
reference_code
llm_description
type
source
class
```

Do not use files from the preprocessing `staging` directory as input for `3_export`. Staging files may not yet contain final class labels or the final schema.

## 4. Optional analysis

The notebooks in `1_analyze` are not required for the pipeline.

Use them after or before preprocessing to:

- compare CLIP, SigLIP, and SigLIP2 embeddings;
- test PCA dimensions and cluster counts;
- inspect class distributions;
- display image matrices for the generated classes.

The notebooks should read the final Parquet shards, not intermediate staging files.

## 5. Configure the standalone export

After preprocessing has produced the final Parquet shards, edit:

```text
3_export/main.py
```

Example:

```python
CONFIG = Config(
    input_dir=Path("/path/to/preprocessed/huggingface"),
    output_dir=Path("/path/to/dataset-exported"),
    benchmark_samples_per_class=20,
    val_samples_per_class=20,
    train_crystalbleu_size=50_000,
    train_size=80_000,
    balance_tolerance=0.10,
    seed=42,
    overwrite=False,
    batch_size=2_048,
)
```

The `input_dir` may contain all final Parquet files in one directory, provided their names contain `train` or `benchmark`.

Valid example:

```text
huggingface/
├── train-00000.parquet
├── train-00001.parquet
├── benchmark-00000.parquet
└── benchmark-00001.parquet
```

Generic names such as only `part-00000.parquet` must not be mixed in one directory because the exporter cannot determine their split.

### Export settings

- `benchmark_samples_per_class`: benchmark samples selected per class.
- `val_samples_per_class`: validation samples selected per class.
- `train_crystalbleu_size`: total size of the class-balanced train CrystalBLEU corpus.
- `train_size`: absolute final train size. Use `None` for the largest possible balanced remainder.
- `balance_tolerance`: permitted class-size tolerance for train selection.
- `overwrite`: whether existing export directories may be deleted and replaced.

For example, `balance_tolerance=0.10` permits larger selected train classes to contain up to 10% more samples than the smallest available class.

## 6. Run the standalone export

Run the exporter only after preprocessing is complete:

```bash
cd 3_export
python main.py
```

`main.py` runs the exports in this order:

```text
1. Benchmark export
2. Train, validation, and train CrystalBLEU export
```

### Benchmark behavior

- Selects a fixed number of samples per class.
- Prefers samples with an LLM description.
- Uses every remaining benchmark sample as CrystalBLEU code.
- Does not balance the benchmark CrystalBLEU remainder.

### Train behavior

- Selects a fixed number of validation samples per class.
- Prefers validation samples with an LLM description.
- Builds a fixed-size, class-balanced CrystalBLEU corpus.
- Prefers CrystalBLEU samples without descriptions.
- Selects the requested absolute train size.
- Keeps final train classes as uniform as possible within the configured tolerance.

Validation, CrystalBLEU, and final train samples are disjoint.

## Export output

```text
dataset-exported/
├── benchmark/
│   ├── input_image/
│   ├── reference_image/
│   ├── llm_description/
│   ├── reference_code/
│   ├── manifest.csv
│   └── crystalbleu/
└── train/
    ├── val/
    │   ├── input_image/
    │   ├── reference_image/
    │   ├── llm_description/
    │   ├── reference_code/
    │   └── manifest.csv
    ├── train/
    │   ├── input_image/
    │   ├── reference_image/
    │   ├── llm_description/
    │   ├── reference_code/
    │   └── manifest.csv
    └── crystalbleu/
```

Manifest paths are absolute. A missing description is represented by an empty `.txt` file. CrystalBLEU directories contain only TikZ/LaTeX code files.

## Complete command order

```bash
# 1. Setup
conda env create -f environment.yaml
conda activate data-tools
sudo sh install_texlive_full.sh

# 2. Install and prepare Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama run qwen3-coder:30b-a3b-q4_K_M

# 3. Download the source datasets
hf download nllg/DaTikZ-V4 \
  --repo-type dataset \
  --local-dir ./dataset/DaTikZ-V4

hf download loss-boss/tikz-benchmark \
  --repo-type dataset \
  --local-dir ./dataset/tikz-benchmark

# 4. Edit 2_preprocess/code/config.py

# 5. Run preprocessing
cd 2_preprocess/code
python pipeline.py
cd ../..

# 6. Optionally inspect the final Parquet files with 1_analyze

# 7. Edit 3_export/main.py

# 8. Run the standalone export
cd 3_export
python main.py
```
