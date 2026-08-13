from pathlib import Path

# -----------------------------------------------------------------------------
# Training stage
# -----------------------------------------------------------------------------
# One of: "text_sft", "projector", "multimodal_sft"
stage = "projector"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
train_manifest = Path("../../data/preprocessed/dataset-exported/train/train/manifest.csv")
val_manifest = Path("../../data/preprocessed/dataset-exported/train/val/manifest.csv")

# Every CSV cell in input_image_col, reference_code_col and description_col is
# treated as a file path. Relative paths are resolved against the directory of
# the corresponding manifest. Set data_root to a Path to use one shared root
# for train and validation instead.
data_root = None

input_image_col = "input_image"
reference_image_col = "reference_image"
reference_code_col = "reference_code"
description_col = "llm_description"
type_col = "type"
source_col = "source"
class_col = "class"

# Phase 1: append the description file content to the shared prompt when True.
text_use_description = True

# Final image -> TikZ training should normally stay image-only.
multimodal_use_description = False

# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------
prompt = """
As a LaTeX graphics expert, translate the image into TikZ code suitable for academic publications.
Focus on recreating geometric precision, typographic elements, and color schemes.
The code must be compilable, efficient, and maintain the original image's visual fidelity for professional document integration.
Return only the full LaTeX document.
Do not use markdown fences.
Do not add explanations.
""".strip()

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
vision_model_name = "/home/jonas/models/siglip2-base-patch16-naflex"
language_model_name = "/home/jonas/models/Qwen2.5-Coder-7B-Instruct"
attention_implementation = "sdpa"

# Checkpoints from earlier stages.
# Required for projector and multimodal_sft.
text_adapter_checkpoint = Path("outputs/text_sft/adapter")
# Required for multimodal_sft.
projector_checkpoint = Path("outputs/projector/projector.pt")

# SigLIP2 NaFlex patch budget. Larger values retain more visual detail but use
# more Qwen context and memory.
max_num_patches = 512

# -----------------------------------------------------------------------------
# Token limits
# -----------------------------------------------------------------------------
# Maximum number of tokens taken from the description file before it is added
# to the instruction.
max_description_tokens = 2048

# Maximum size of the complete instruction after applying the Qwen chat
# template. This includes the base prompt, optional description and chat tokens.
max_prompt_tokens = 4096

# Maximum number of tokens in the reference LaTeX/TikZ code.
# EOS is added separately.
max_target_tokens = 4096

# -----------------------------------------------------------------------------
# LoRA
# -----------------------------------------------------------------------------
lora_r = 16
lora_alpha = 32
lora_dropout = 0.0
lora_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------
seed = 42
num_epochs = 100
batch_size = 1
gradient_accumulation_steps = 5
num_workers = 4

text_sft_lr = 2e-4
projector_lr = 1e-3
multimodal_sft_lr = 1e-4
weight_decay = 0.01
warmup_ratio = 0.03
max_grad_norm = 1.0

# "bf16", "fp16", or "no". bf16 is recommended on recent NVIDIA GPUs.
mixed_precision = "bf16"
gradient_checkpointing = True

# -----------------------------------------------------------------------------
# Logging / validation
# -----------------------------------------------------------------------------
log_every_steps = 5
validate_every_steps = 20

use_tensorboard = True
tensorboard_root = Path("runs")
tensorboard_project_name = "tikz_vlm"

output_root = Path("outputs")

# -----------------------------------------------------------------------------
# Qualitative validation generation
# -----------------------------------------------------------------------------
# The same validation samples are generated after every validation run so their
# progression can be compared across epochs in TensorBoard.
log_validation_generations = True
validation_generation_indices = [0, 1, 2, 3]
generation_max_new_tokens = 4097
generation_do_sample = False

# Rendered predictions are also kept on disk in
# outputs/<stage>/validation_renders/epoch_XXXX/.

# -----------------------------------------------------------------------------
# LaTeX rendering
# -----------------------------------------------------------------------------
class RenderConfig:
    # Leave empty to use executables from PATH. If required, point this to the
    # TeX Live binary directory instead.
    texlive_bin = Path("")
    engines = ("pdflatex", "lualatex", "xelatex")
    timeout_seconds = 30
    latex_runs = 1
    halt_on_error = True
    tolerant_fallback = True
    disable_page_numbers = True
    crop_pdf = True
    crop_png = True
    normalize_canvas = True
    dpi = 200
    image_size = 512
    upscale_canvas = False


RENDER = RenderConfig()
