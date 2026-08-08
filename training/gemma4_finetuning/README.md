# Gemma 4 31B IT: SFT + GRPO (No Docker)

This project is completely Docker-free and uses only relative paths. It expects the following directory structure:

```text
project-root/
├── evaluation/
│   └── promptfoo/
└── training/
    └── gemma4_finetuning_grpo_fast/
        ├── data/
        ├── models/
        ├── latex/
        ├── gemma4-sft/
        └── gemma4-rl/
```

All configuration is defined directly in `config.py`. No `.env` file is used.

## 1. Create the Conda Environment

```bash
cd training/gemma4_finetuning_grpo_fast
conda env create -f environment.yaml
conda activate gemma4-finetuning
```

## 2. Install a Local TeX Live Distribution

```bash
./install_latex.sh
```

The script installs the complete TeX Live distribution into `./latex`. At runtime, `config.configure_runtime()` automatically adds the appropriate `latex/bin/<platform>` directory to `PATH`.

## 3. Model and Dataset

Place the original Gemma 4 31B IT model in:

```text
models/gemma-4-31B-it/
```

Store the dataset manifests in `data/`:

```text
data/manifest_train.csv
data/manifest_val.csv
data/manifest.csv
```

Required CSV columns:

```csv
image_path,code_path,vlm_description_path
```

Relative paths inside the CSV files are resolved relative to the `data/` directory.

## 4. Training

```bash
python gemma4-sft/train.py
python gemma4-rl/train.py
```

SFT saves the LoRA adapter to:

```text
models/sft/lora
```

GRPO automatically loads the SFT adapter and saves the resulting adapter to:

```text
models/grpo/lora
```

For faster GRPO generation, `gradient_checkpointing` is disabled by default, allowing the KV cache to remain enabled.

## 5. TensorBoard

```bash
tensorboard --logdir models --port 6006
```

## 6. Export to GGUF

```bash
python export_gguf.py
```

## Configuration

Adjust all training parameters, including batch sizes, sequence lengths, training steps, and reward weights, directly in `config.py`.
