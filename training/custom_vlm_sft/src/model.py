from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, Siglip2VisionModel

import config as cfg


def _model_dtype():
    if cfg.mixed_precision == "bf16":
        return torch.bfloat16
    if cfg.mixed_precision == "fp16":
        return torch.float16
    return torch.float32


class VisionProjector(nn.Module):
    def __init__(self, vision_dim: int, language_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.language_model_name,
        dtype=_model_dtype(),
        attn_implementation=cfg.attention_implementation,
    )
    llm.config.use_cache = False

    if cfg.gradient_checkpointing:
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if cfg.stage == "text_sft":
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        llm = get_peft_model(llm, lora)
    else:
        adapter_path = Path(cfg.text_adapter_checkpoint)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"text_adapter_checkpoint does not exist: {adapter_path}"
            )
        llm = PeftModel.from_pretrained(
            llm,
            adapter_path,
            is_trainable=(cfg.stage == "multimodal_sft"),
        )

        if cfg.stage == "projector":
            for parameter in llm.parameters():
                parameter.requires_grad = False

    return llm


class TikZVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage = cfg.stage
        self.llm = _load_llm()
        self.vision = None
        self.projector = None

        if self.stage != "text_sft":
            self.vision = Siglip2VisionModel.from_pretrained(
                cfg.vision_model_name,
                dtype=_model_dtype(),
                attn_implementation=cfg.attention_implementation,
            )
            self.vision.requires_grad_(False)
            self.vision.eval()

            language_dim = self.llm.get_input_embeddings().embedding_dim
            self.projector = VisionProjector(
                self.vision.config.hidden_size,
                language_dim,
            )

            if self.stage == "multimodal_sft":
                checkpoint = Path(cfg.projector_checkpoint)
                if not checkpoint.exists():
                    raise FileNotFoundError(
                        f"projector_checkpoint does not exist: {checkpoint}"
                    )
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.projector.load_state_dict(state)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision is not None:
            self.vision.eval()
        return self

    def _prepare_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.vision is None or self.projector is None:
            raise RuntimeError("Multimodal components are not initialized.")

        with torch.no_grad():
            vision_outputs = self.vision(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            vision_hidden = vision_outputs.last_hidden_state

        visual_embeds = self.projector(vision_hidden)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        visual_embeds = visual_embeds.to(dtype=text_embeds.dtype)

        sequences: list[torch.Tensor] = []
        sequence_labels: list[torch.Tensor] = []

        for i in range(input_ids.shape[0]):
            visual = visual_embeds[i][pixel_attention_mask[i].bool()]
            text_length = int(attention_mask[i].sum().item())
            text = text_embeds[i, :text_length]
            sequences.append(torch.cat([visual, text], dim=0))

            if labels is not None:
                visual_labels = torch.full(
                    (visual.shape[0],),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                sequence_labels.append(
                    torch.cat([visual_labels, labels[i, :text_length]], dim=0)
                )

        max_length = max(sequence.shape[0] for sequence in sequences)
        hidden_size = sequences[0].shape[-1]
        inputs_embeds = sequences[0].new_zeros(
            (len(sequences), max_length, hidden_size)
        )
        full_attention_mask = attention_mask.new_zeros(
            (len(sequences), max_length)
        )
        full_labels = (
            labels.new_full((len(sequences), max_length), -100)
            if labels is not None
            else None
        )

        for i, sequence in enumerate(sequences):
            length = sequence.shape[0]
            inputs_embeds[i, :length] = sequence
            full_attention_mask[i, :length] = 1
            if full_labels is not None:
                full_labels[i, :length] = sequence_labels[i]

        return inputs_embeds, full_attention_mask, full_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ):
        if self.stage == "text_sft":
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

        if pixel_values is None or pixel_attention_mask is None or spatial_shapes is None:
            raise ValueError("Missing SigLIP2 image inputs for multimodal training.")

        inputs_embeds, full_attention_mask, full_labels = (
            self._prepare_multimodal_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                labels=labels,
            )
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            use_cache=False,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        max_new_tokens: int = 4097,
        do_sample: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if self.stage == "text_sft":
            sequences = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
            prompt_length = input_ids.shape[1]
            return sequences[:, prompt_length:]

        if pixel_values is None or pixel_attention_mask is None or spatial_shapes is None:
            raise ValueError("Missing SigLIP2 image inputs for multimodal generation.")

        inputs_embeds, full_attention_mask, _ = self._prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            labels=None,
        )

        # With decoder-only generation and only inputs_embeds supplied, the
        # returned sequences contain the newly generated token IDs.
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )
