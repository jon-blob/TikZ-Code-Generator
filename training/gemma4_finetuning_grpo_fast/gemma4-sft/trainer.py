from __future__ import annotations

from pathlib import Path

from tb_callback import LatexEvalCallback

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import torch
from trl import SFTConfig, SFTTrainer
import random

#just for debugging
def inspect_loss(
    trainer: SFTTrainer,
    model,
) -> None:
    was_training = model.training
    model.eval()

    def inspect(name: str, dataloader) -> None:
        batch = next(iter(dataloader))

        labels = batch["labels"]
        active_tokens = (labels != -100).sum(dim=1)

        print(f"\n{name}")
        print("Input shape:", batch["input_ids"].shape)
        print("Active loss tokens:", active_tokens.tolist())

        prepared = trainer._prepare_inputs(batch)

        with torch.no_grad():
            loss = trainer.compute_loss(
                model,
                prepared,
            )

        print("Direct microbatch loss:", float(loss))

    inspect(
        "TRAIN BATCH",
        trainer.get_train_dataloader(),
    )
    inspect(
        "EVAL BATCH",
        trainer.get_eval_dataloader(),
    )

    if was_training:
        model.train()




#just for debugging
def audit_end_tokens(
    dataset,
    collator,
    processor,
    max_seq_length,
    n=5000,
):
    tokenizer = processor.tokenizer

    eot_id = getattr(tokenizer, "eot_token_id", None)
    if eot_id is None:
        eot_token = getattr(tokenizer, "eot_token", "<turn|>")
        eot_id = tokenizer.convert_tokens_to_ids(eot_token)

    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        stop_ids = {eos_ids, eot_id}
    else:
        stop_ids = set(eos_ids or [])
        stop_ids.add(eot_id)

    missing_end = 0
    at_limit = 0
    empty = 0
    bad_examples = []

    for index in range(min(n, len(dataset))):
        batch = collator([dataset[index]])

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        active = labels[labels != -100]

        if active.numel() == 0:
            empty += 1
            continue

        if len(input_ids) >= max_seq_length:
            at_limit += 1

        # <turn|> kann vor einem abschließenden Newline stehen.
        tail_ids = active[-16:].tolist()
        has_stop = any(token_id in stop_ids for token_id in tail_ids)

        if not has_stop:
            missing_end += 1

            if len(bad_examples) < 10:
                bad_examples.append({
                    "index": index,
                    "length": len(input_ids),
                    "ending": tokenizer.decode(
                        active[-30:],
                        skip_special_tokens=False,
                    ),
                    "tail_ids": tail_ids,
                })

    print(f"Geprüft: {min(n, len(dataset))}")
    print(f"Am Sequenzlimit: {at_limit}")
    print(f"Ohne Stop-Token: {missing_end}")
    print(f"Leere Targets: {empty}")

    for example in bad_examples:
        print("\n", example)

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
        image_paths=[
            "/root/projects/training/gemma4_finetuning_grpo_fast/data/train/images/00000005.png",
            "/root/projects/training/gemma4_finetuning_grpo_fast/data/train/images/00323734.png",
            "/root/projects/training/gemma4_finetuning_grpo_fast/data/val/images/00000095.png",
            "/root/projects/training/gemma4_finetuning_grpo_fast/data/val/images/00001742.png",
        ],
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

    #just for debugging
    inspect_loss(trainer, model)
    #audit_end_tokens(train_dataset, collator, processor, cfg.max_seq_length, n=5000,)
    print("Configured LR:", trainer.args.learning_rate)
    print("Warmup steps:", trainer.args.warmup_steps)
    print("Resume:", cfg.resume_from_checkpoint)

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(str(cfg.lora_output_dir))
    processor.save_pretrained(str(cfg.lora_output_dir))
