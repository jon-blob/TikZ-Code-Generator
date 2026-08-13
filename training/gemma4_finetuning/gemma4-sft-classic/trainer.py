from __future__ import annotations

from pathlib import Path

import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from debug_tools import DebugVisionDataCollator, audit_labels
from tb_callback import LatexEvalCallback


def train_sft(cfg, model, processor, train_dataset, eval_dataset) -> None:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    debug_train_samples = getattr(cfg, "debug_train_sample_logging", True)
    num_workers = 0 if debug_train_samples else 4

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

        weight_decay=cfg.weight_decay,

        # For this diagnostic run, select checkpoints by free-generation behavior,
        # not automatically by the lowest teacher-forced eval loss.
        load_best_model_at_end=False,

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

        # Exact sample->optimizer-step mapping requires no worker prefetching.
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=False if num_workers == 0 else True,
        dataloader_prefetch_factor=None if num_workers == 0 else 2,
    )

    FastVisionModel.for_training(model)

    base_collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=cfg.max_seq_length,
        resize=cfg.image_resize,
        completion_only_loss=True,
    )

    audit_labels(
        base_collator=base_collator,
        dataset=train_dataset,
        model=model,
        processor=processor,
        output_path=cfg.output_dir / "generations" / "label_audit.json",
        n=getattr(cfg, "debug_label_audit_samples", 100),
    )

    # Fixed validation sample for teacher-forced stop probability/rank.
    probe_index = min(
        int(getattr(cfg, "debug_teacher_probe_eval_index", 0)),
        len(eval_dataset) - 1,
    )
    probe_feature = dict(eval_dataset[probe_index])
    probe_feature.pop("_debug_meta", None)
    teacher_probe_batch = base_collator([probe_feature])

    if debug_train_samples:
        collator = DebugVisionDataCollator(
            base_collator=base_collator,
            output_path=cfg.output_dir / "generations" / "train_samples_by_step.jsonl",
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            resume=cfg.resume_from_checkpoint,
        )
    else:
        collator = base_collator

    callback = LatexEvalCallback(
        processor=processor,
        image_paths=cfg.image_paths_for_callback,
        prompt=Path(cfg.instruction_path).read_text().strip(),
        output_dir=cfg.output_dir / "generations",
        max_new_tokens=getattr(cfg, "debug_max_new_tokens", 512),
        teacher_probe_batch=teacher_probe_batch,
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

    # Establish the pre-SFT baseline using the exact same callback and eval sample.
    if getattr(cfg, "debug_eval_at_step0", True) and not cfg.resume_from_checkpoint:
        print("\n[DEBUG] Step-0 evaluation before the first optimizer update", flush=True)
        trainer.evaluate()

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(str(cfg.lora_output_dir))
    processor.save_pretrained(str(cfg.lora_output_dir))
