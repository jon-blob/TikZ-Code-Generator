# TikZ VLM SFT

Minimal three-stage SFT pipeline for SigLIP2 NaFlex + Qwen Coder.

## Project structure

```text
custom_vlm_sft/
├── config.py
├── environment.yml
├── README.md
├── train.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── model.py
    ├── render.py
    └── validation.py
```

All settings are configured directly in `config.py`. No training YAML or
environment-variable based configuration is used.

## Data

Set both manifests in `config.py`:

```python
train_manifest = Path("/path/to/train/manifest.csv")
val_manifest = Path("/path/to/val/manifest.csv")
```

The configured CSV columns `input_image`, `reference_code`, and
`llm_description` are treated as file paths. Relative paths are resolved
against the directory containing the corresponding manifest unless
`data_root` is set.

```csv
input_image,reference_image,reference_code,llm_description,type,source,class
images/001.png,reference_images/001.png,codes/001.tex,descriptions/001.txt,...,...,...
```

## Shared prompt

All stages use the same `prompt`. Description usage is controlled separately:

```python
text_use_description = True
multimodal_use_description = False
```

With the defaults:

- `text_sft`: prompt + description -> reference code
- `projector`: image + prompt -> reference code
- `multimodal_sft`: image + prompt -> reference code

## Token limits

Prompt, description, and reference-code limits are configured separately:

```python
max_description_tokens = 2048
max_prompt_tokens = 4096
max_target_tokens = 4096
```

`max_description_tokens` truncates only the optional description before it is
added to the shared prompt. `max_prompt_tokens` is a safety limit for the full
Qwen chat-formatted instruction. `max_target_tokens` applies only to the
reference LaTeX/TikZ source; targets are never silently truncated. EOS is added
separately.

## Training stages

Change only `stage` in `config.py`.

### 1. Text SFT

```python
stage = "text_sft"
```

Qwen is trained with LoRA. With `text_use_description = True`, the description
file content is appended to the shared prompt.

```bash
accelerate launch train.py
```

The adapter is saved to `outputs/text_sft/adapter/`.

### 2. Projector alignment

```python
stage = "projector"
text_adapter_checkpoint = Path("outputs/text_sft/adapter")
```

Qwen and SigLIP2 are frozen. Only the vision projector is trained. The
projector is saved to `outputs/projector/projector.pt`.

### 3. Multimodal SFT

```python
stage = "multimodal_sft"
text_adapter_checkpoint = Path("outputs/text_sft/adapter")
projector_checkpoint = Path("outputs/projector/projector.pt")
```

SigLIP2 stays frozen. The projector and Qwen LoRA adapter are trainable.

## Validation

The complete validation manifest is used for teacher-forced validation loss.
The loss is weighted by the number of target tokens.

Qualitative generation uses fixed validation indices:

```python
log_validation_generations = True
validation_generation_indices = [0, 1, 2, 3]
generation_max_new_tokens = 4097
generation_do_sample = False
```

Generation is enabled for:

- `text_sft` when `text_use_description = True`
- `projector`
- `multimodal_sft`

For every configured validation sample, TensorBoard shows:

```text
validation/sample_0000/input_image
validation/sample_0000/generated_image
validation/sample_0000/generated_code
validation/sample_0000/reference_code
validation/sample_0000/render_success
```

If generated LaTeX fails to render, `render_error` is logged instead of
stopping training.

Rendered predictions are also written to:

```text
outputs/<stage>/validation_renders/epoch_XXXX/
```

## Rendering

`src/validation.py` imports the renderer directly:

```python
from src.render import render_tex_to_png
```

`train.py` only calls `log_validation_generations(...)` from `src.validation`.
The renderer uses the settings in `config.py` under `RENDER`. With
`texlive_bin = Path("")`, LaTeX executables are resolved through `PATH`.
A TeX Live installation with the packages required by your target documents
must be available. `pdftoppm` and `pdfinfo` are provided by Poppler.

## TensorBoard

```python
use_tensorboard = True
tensorboard_root = Path("runs")
tensorboard_project_name = "tikz_vlm"
```

Start TensorBoard with:

```bash
tensorboard --logdir runs
```

Scalar logs include:

```text
train/loss
train/lr
train/epoch
val/loss
val/epoch
```

## Environment

```bash
conda env create -f environment.yml
conda activate tikz-vlm
```
