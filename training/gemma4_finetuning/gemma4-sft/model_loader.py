from unsloth import FastVisionModel


def load_model(cfg):
    print(f"Loading {cfg.model_name} in 4-bit={cfg.load_in_4bit}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        fast_inference=False,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False, #normal run was without vision layers
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        random_state=cfg.seed,
    )
    return model, processor