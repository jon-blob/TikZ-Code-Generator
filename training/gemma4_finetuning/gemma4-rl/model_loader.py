from unsloth import FastVisionModel


def load_model(cfg):
    print(f"Loading {cfg.model_name} in 4-bit={cfg.load_in_4bit}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        fast_inference=False,
    )

    if not getattr(model, "peft_config", None):
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            random_state=cfg.seed,
            use_gradient_checkpointing=(
                "unsloth" if cfg.gradient_checkpointing else False
            ),
        )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True

    print(
        "GRPO gradient checkpointing:", cfg.gradient_checkpointing,
        "| KV cache:", model.config.use_cache,
    )
    return model, processor
