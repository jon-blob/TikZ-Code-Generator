from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator

import config as cfg
from src.data import TikZCollator, TikZDataset
from src.model import TikZVLM
from src.render import render_tex_to_png


def _should_generate() -> bool:
    if not cfg.use_tensorboard or not cfg.log_validation_generations:
        return False
    if cfg.stage == "text_sft":
        return cfg.text_use_description
    return cfg.stage in {"projector", "multimodal_sft"}


def _code_block(code: str) -> str:
    return f"```latex\n{code.strip()}\n```"


@torch.no_grad()
def log_validation_generations(
    accelerator: Accelerator,
    model: TikZVLM,
    dataset: TikZDataset,
    collator: TikZCollator,
    tokenizer,
    epoch: int,
    global_step: int,
) -> None:
    """Generate fixed validation samples and log them to TensorBoard."""
    if not _should_generate():
        return

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        was_training = unwrapped_model.training
        unwrapped_model.eval()

        writer = accelerator.get_tracker("tensorboard", unwrap=True)
        render_dir = (
            Path(cfg.output_root)
            / cfg.stage
            / "validation_renders"
            / f"epoch_{epoch:04d}"
        )

        for index in cfg.validation_generation_indices:
            if index < 0 or index >= len(dataset):
                accelerator.print(
                    f"Skipping validation generation index {index}: "
                    f"dataset contains {len(dataset)} samples."
                )
                continue

            sample = dataset.get_generation_sample(index)
            generation_batch = collator.generation_batch(sample)
            generation_batch = {
                key: value.to(accelerator.device)
                for key, value in generation_batch.items()
            }

            with accelerator.autocast():
                generated_ids = unwrapped_model.generate(
                    **generation_batch,
                    max_new_tokens=cfg.generation_max_new_tokens,
                    do_sample=cfg.generation_do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_code = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            ).strip()
            reference_code = sample["code"].strip()
            tag = f"validation/sample_{index:04d}"

            writer.add_image(
                f"{tag}/input_image",
                np.asarray(sample["image"]),
                global_step=global_step,
                dataformats="HWC",
            )
            writer.add_text(
                f"{tag}/generated_code",
                _code_block(generated_code),
                global_step=global_step,
            )
            writer.add_text(
                f"{tag}/reference_code",
                _code_block(reference_code),
                global_step=global_step,
            )

            render_metrics: dict = {}
            render_path = render_dir / f"sample_{index:04d}.png"
            try:
                render_tex_to_png(
                    generated_code,
                    render_path,
                    metrics=render_metrics,
                )
                with Image.open(render_path) as rendered:
                    writer.add_image(
                        f"{tag}/generated_image",
                        np.asarray(rendered.convert("RGB")),
                        global_step=global_step,
                        dataformats="HWC",
                    )
                writer.add_scalar(
                    f"{tag}/render_success",
                    1,
                    global_step=global_step,
                )
            except Exception as error:
                writer.add_scalar(
                    f"{tag}/render_success",
                    0,
                    global_step=global_step,
                )
                writer.add_text(
                    f"{tag}/render_error",
                    str(error),
                    global_step=global_step,
                )

        writer.flush()
        if was_training:
            unwrapped_model.train()

    accelerator.wait_for_everyone()
