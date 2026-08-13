from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer, get_cosine_schedule_with_warmup

import config as cfg
from src.data import TikZCollator, TikZDataset
from src.model import TikZVLM
from src.validation import log_validation_generations


def _learning_rate() -> float:
    return {
        "text_sft": cfg.text_sft_lr,
        "projector": cfg.projector_lr,
        "multimodal_sft": cfg.multimodal_sft_lr,
    }[cfg.stage]


def _save(accelerator: Accelerator, model: TikZVLM) -> None:
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    model = accelerator.unwrap_model(model)
    output_dir = Path(cfg.output_root) / cfg.stage
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.stage in {"text_sft", "multimodal_sft"}:
        model.llm.save_pretrained(
            output_dir / "adapter",
            safe_serialization=True,
        )

    if model.projector is not None:
        projector_state = {
            key: value.detach().cpu()
            for key, value in model.projector.state_dict().items()
        }
        torch.save(projector_state, output_dir / "projector.pt")

    config_file = Path(__file__).resolve().with_name("config.py")
    shutil.copy2(config_file, output_dir / "config.py")


@torch.no_grad()
def _validate(
    accelerator: Accelerator,
    model: TikZVLM,
    dataloader: DataLoader,
) -> float:
    model.eval()

    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    token_count = torch.zeros((), device=accelerator.device, dtype=torch.float32)

    for batch in dataloader:
        with accelerator.autocast():
            outputs = model(**batch)

        target_tokens = (batch["labels"] != -100).sum().float()
        loss_sum += outputs.loss.detach().float() * target_tokens
        token_count += target_tokens

    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    token_count = accelerator.reduce(token_count, reduction="sum")

    model.train()

    if token_count.item() == 0:
        raise RuntimeError("Validation dataset contains no target tokens.")

    return (loss_sum / token_count).item()


def main() -> None:
    output_dir = Path(cfg.output_root) / cfg.stage
    tensorboard_dir = Path(cfg.tensorboard_root) / cfg.stage

    project_config = ProjectConfiguration(
        project_dir=str(output_dir),
        logging_dir=str(tensorboard_dir),
    )

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard" if cfg.use_tensorboard else None,
        project_config=project_config,
    )

    set_seed(cfg.seed)

    accelerator.print(
        f"device={accelerator.device} | "
        f"mixed_precision={accelerator.mixed_precision}"
    )

    if cfg.use_tensorboard:
        accelerator.init_trackers(
            project_name=cfg.tensorboard_project_name,
            config={
                "stage": cfg.stage,
                "train_manifest": str(cfg.train_manifest),
                "val_manifest": str(cfg.val_manifest),
                "vision_model": cfg.vision_model_name,
                "language_model": cfg.language_model_name,
                "learning_rate": _learning_rate(),
                "batch_size": cfg.batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "num_epochs": cfg.num_epochs,
                "max_num_patches": cfg.max_num_patches,
                "max_description_tokens": cfg.max_description_tokens,
                "max_prompt_tokens": cfg.max_prompt_tokens,
                "max_target_tokens": cfg.max_target_tokens,
                "text_use_description": cfg.text_use_description,
                "multimodal_use_description": cfg.multimodal_use_description,
                "log_validation_generations": cfg.log_validation_generations,
                "generation_max_new_tokens": cfg.generation_max_new_tokens,
                "log_every_steps": cfg.log_every_steps,
                "validate_every_steps": cfg.validate_every_steps,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.language_model_name)
    tokenizer.padding_side = "right"

    image_processor = None
    if cfg.stage != "text_sft":
        image_processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)

    train_dataset = TikZDataset(cfg.train_manifest)
    val_dataset = TikZDataset(cfg.val_manifest)
    collator = TikZCollator(tokenizer, image_processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    model = TikZVLM()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")

    optimizer = AdamW(
        trainable,
        lr=_learning_rate(),
        weight_decay=cfg.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler,
    )

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    accelerator.print(
        f"stage={cfg.stage} | "
        f"train={len(train_dataset)} | "
        f"val={len(val_dataset)} | "
        f"trainable_params={trainable_count:,} | "
        f"total_steps={total_steps}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0

    accumulated_loss = torch.zeros(
        (),
        device=accelerator.device,
        dtype=torch.float32,
    )
    accumulated_tokens = torch.zeros(
        (),
        device=accelerator.device,
        dtype=torch.float32,
    )

    for epoch in range(cfg.num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss

                accelerator.backward(loss)

                target_tokens = (batch["labels"] != -100).sum().float()
                accumulated_loss += loss.detach().float() * target_tokens
                accumulated_tokens += target_tokens

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        cfg.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                loss_sum = accelerator.reduce(
                    accumulated_loss,
                    reduction="sum",
                )
                token_count = accelerator.reduce(
                    accumulated_tokens,
                    reduction="sum",
                )

                train_loss = (loss_sum / token_count).item()

                accumulated_loss.zero_()
                accumulated_tokens.zero_()

                if global_step % cfg.log_every_steps == 0:
                    lr = scheduler.get_last_lr()[0]

                    accelerator.print(
                        f"epoch={epoch + 1}/{cfg.num_epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={train_loss:.4f} "
                        f"lr={lr:.2e}"
                    )

                    if cfg.use_tensorboard:
                        accelerator.log(
                            {
                                "train/loss": train_loss,
                                "train/lr": lr,
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                if global_step % cfg.validate_every_steps == 0:
                    val_loss = _validate(
                        accelerator,
                        model,
                        val_dataloader,
                    )

                    accelerator.print(
                        f"validation | "
                        f"epoch={epoch + 1}/{cfg.num_epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={val_loss:.4f}"
                    )

                    if cfg.use_tensorboard:
                        accelerator.log(
                            {
                                "val/loss": val_loss,
                                "val/epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                    log_validation_generations(
                        accelerator=accelerator,
                        model=model,
                        dataset=val_dataset,
                        collator=collator,
                        tokenizer=tokenizer,
                        epoch=epoch + 1,
                        global_step=global_step,
                    )

    _save(accelerator, model)
    accelerator.print(f"Saved to {output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()