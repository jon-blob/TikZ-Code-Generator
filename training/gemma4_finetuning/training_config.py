from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR_SFT = Path("/home/jonas/Datasets/TikZ/tikz-dataset-clean/sft-train/train")
DATA_DIR_RL = Path("/home/jonas/Datasets/TikZ/tikz-dataset-clean/rl-train/train")
MODELS_DIR = Path("/home/jonas/models")
PROMPTFOO_DIR = ROOT.parent.parent / "evaluation" / "promptfoo"
LATEX_DIR = ROOT / "latex"
INSTRUCTION_FILE = ROOT / "prompts" / "instruction.txt"
USE_LLM_DESCRIPTION=True

def configure_runtime() -> None:
    """Expose the local Promptfoo utilities and local TeX Live installation."""
    import sys

    if PROMPTFOO_DIR.is_dir() and str(PROMPTFOO_DIR) not in sys.path:
        sys.path.insert(0, str(PROMPTFOO_DIR))

    latex_bins = sorted(LATEX_DIR.glob("bin/*"))
    if latex_bins:
        os.environ["PATH"] = f"{latex_bins[0]}:{os.environ.get('PATH', '')}"


@dataclass(slots=True)
class SFTConfig:
    model_name: str = str(MODELS_DIR / "gemma-4-12B-it")
    load_in_4bit: bool = False
    enable_thinking: bool = False
    use_llm_description: bool = USE_LLM_DESCRIPTION
    
    dataset_path: Path = DATA_DIR_SFT
    train_manifest: str = "train/manifest.csv"
    val_manifest: str = "val/manifest.csv"
    instruction_path: Path = INSTRUCTION_FILE
    image_column: str = "input_image"
    code_column: str = "reference_code"
    vlm_description_column: str = "llm_description"

    output_dir: Path = MODELS_DIR / "sft" / "checkpoints"
    lora_output_dir: Path = MODELS_DIR / "sft" / "lora"

    max_seq_length: int = 16384
    image_resize: str | int = "min"
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.3
    weight_decay: float = 0.2
    seed: int = 3407
    num_examples_train: int | None = None
    num_examples_val: int | None = None

    learning_rate: float = 1e-6
    epochs: float = 1.0
    max_steps: int = -1
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 1
    warmup_steps: int = 200
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    resume_from_checkpoint: bool = False

    # Endless-generation debugging
    debug_eval_at_step0: bool = True
    debug_max_new_tokens: int = 1024
    debug_label_audit_samples: int = 100
    debug_teacher_probe_eval_index: int = 0
    debug_train_sample_logging: bool = True

    image_paths_for_callback = [
            f"{DATA_DIR_SFT}/val/reference_image/class_1_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_2_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_4_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_5_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_6_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_7_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_8_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_9_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_10_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_25_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_18_00000001.png",
            f"{DATA_DIR_SFT}/val/reference_image/class_19_00000001.png",
        ]

    descripion_paths_for_callback = [
            f"{DATA_DIR_SFT}/val/reference_code/class_1_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_2_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_4_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_5_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_6_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_7_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_8_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_9_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_10_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_25_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_18_00000001.txt",
            f"{DATA_DIR_SFT}/val/reference_code/class_19_00000001.txt",
    ]



@dataclass(slots=True)
class GRPOConfigData:
    model_name: str = str(MODELS_DIR / "sft" / "checkpoints" / "checkpoint-2400")
    load_in_4bit: bool = True
    enable_thinking: bool = False
    gradient_checkpointing: bool = False
    use_llm_description: bool = USE_LLM_DESCRIPTION

    dataset_path: Path = DATA_DIR_RL
    manifest: str = "manifest.csv"
    instruction_path: Path = INSTRUCTION_FILE
    input_image_column: str = "input_image"
    reference_image_column: str = "reference_image"
    code_column: str = "reference_code"
    vlm_description_column: str = "llm_description"

    output_dir: Path = MODELS_DIR / "grpo" / "checkpoints"
    lora_output_dir: Path = MODELS_DIR / "grpo" / "lora"

    max_seq_length: int = 16384
    max_prompt_length: int = 1024
    max_completion_length: int = 4096
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.3
    weight_decay: float = 0.2
    seed: int = 3407
    num_examples: int | None = None

    learning_rate: float = 1e-6
    max_steps: int = -1
    save_steps: int = 50
    logging_steps: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    shuffle_dataset: bool = False

    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.05
    beta: float = 0.0

    log_examples_every: int = 1
    log_examples_max: int = 4
    max_logged_code_chars: int = 20000
    crystalbleu_corpus_dir: Path = DATA_DIR_SFT / "crystalbleu"
    crystalbleu_k: int = 500
    crystalbleu_n: int = 4
    crystalbleu_use_cache: bool = True
    crystalbleu_weight: float = 1.0
    ted_weight: float = 0.5
    ted_scale: float = 100.0

    not_renderable_score: float = -2.0
    renderable_score: float = 1.0
    code_reward_multiplier: float = 3.0
    visual_reward_multiplier: float = 0.5
    error_multiplier: float = 0.10
    warning_multiplier: float = 0.05
    badboxes_multiplier: float = 0.01
    diagnostic_base_max_score: float = 1.0
    diagnostic_base_min_score: float = -2.0
    siglip_multiplier: float = 0.15
    lpips_multiplier: float = 0.5
    dreamsim_multiplier: float = 0.35

    def __post_init__(self) -> None:
        if self.max_prompt_length + self.max_completion_length > self.max_seq_length:
            raise ValueError("Prompt and completion lengths exceed max_seq_length.")
        effective_batch = self.batch_size * self.gradient_accumulation_steps
        if effective_batch % self.num_generations:
            raise ValueError("Effective batch size must be divisible by num_generations.")
