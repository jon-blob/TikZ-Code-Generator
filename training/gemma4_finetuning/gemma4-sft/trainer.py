from __future__ import annotations

from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from tb_callback import LatexEvalCallback


def train_sft(cfg, model, processor, train_dataset, eval_dataset) -> None:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = SFTConfig(
        output_dir=str(cfg.output_dir),

        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        max_grad_norm=1.0,

        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_steps,

        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        prediction_loss_only=True,

        save_strategy="steps",
        save_steps=cfg.save_steps,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=cfg.logging_steps,
        report_to="tensorboard",
        logging_dir=str(cfg.output_dir / "logs"),

        optim="adamw_8bit",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,

        gradient_checkpointing=True,

        remove_unused_columns=False,
        dataset_text_field="",
        max_length=None,
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},

        seed=cfg.seed,
        data_seed=cfg.seed,

        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
    )

    FastVisionModel.for_training(model)
    collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=cfg.max_seq_length,
        resize=cfg.image_resize,
        completion_only_loss=True,
    )

    callback = LatexEvalCallback(
        processor=processor,
        image_paths=cfg.image_paths_for_callback,
        prompt=Path(cfg.instruction_path).read_text().strip(),
        output_dir=cfg.output_dir / "generations",
        max_new_tokens=8192,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(str(cfg.lora_output_dir))
    processor.save_pretrained(str(cfg.lora_output_dir))