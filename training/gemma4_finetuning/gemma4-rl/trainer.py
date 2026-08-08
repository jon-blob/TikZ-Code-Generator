from __future__ import annotations

from unsloth import FastVisionModel

import torch
from trl import GRPOConfig, GRPOTrainer

from rewards import TikZReward
from tb_callback import TensorBoardRewardCallback


def train_grpo(cfg, model, processor, dataset) -> None:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        learning_rate=cfg.learning_rate,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        beta=cfg.beta,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        report_to="tensorboard",
        logging_dir=str(cfg.output_dir / "logs"),
        remove_unused_columns=False,
        mask_truncated_completions=True,
        scale_rewards=False,
        loss_type="dr_grpo",
        seed=cfg.seed,
    )

    reward = TikZReward(cfg)
    callback = TensorBoardRewardCallback(str(cfg.output_dir / "logs"), reward)
    FastVisionModel.for_training(model)
    trainer = GRPOTrainer(
        model=model,
        args=args,
        processing_class=processor,
        reward_funcs=[reward],
        train_dataset=dataset,
        callbacks=[callback],
    )
    trainer.train()
    trainer.save_model(str(cfg.lora_output_dir))
    processor.save_pretrained(str(cfg.lora_output_dir))
