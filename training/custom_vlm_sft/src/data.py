from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import config as cfg


VALID_STAGES = {"text_sft", "projector", "multimodal_sft"}


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _read_text_file(value: Any, root: Path) -> str:
    path = _resolve_path(value, root)
    if not path.is_file():
        raise FileNotFoundError(f"Text file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def _read_image(value: Any, root: Path) -> Image.Image:
    path = _resolve_path(value, root)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def render_instruction(
    prompt: str,
    description: str | None,
    use_llm_description: bool,
) -> str:
    final_prompt = prompt.strip()
    if description and description.strip() and use_llm_description:
        final_prompt += (
            "\n\nAdditionally, here is a description of the image "
            "with some creation hints:\n"
            f"{description.strip()}"
        )
    return final_prompt


class TikZDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        if cfg.stage not in VALID_STAGES:
            raise ValueError(f"Unknown stage: {cfg.stage!r}")

        self.manifest_path = Path(manifest_path).expanduser()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest_path}")

        self.root = (
            Path(cfg.data_root).expanduser()
            if cfg.data_root is not None
            else self.manifest_path.parent
        )
        self.df = pd.read_csv(self.manifest_path)

        self.use_description = (
            cfg.text_use_description
            if cfg.stage == "text_sft"
            else cfg.multimodal_use_description
        )

        required = [cfg.reference_code_col]
        if cfg.stage != "text_sft":
            required.append(cfg.input_image_col)
        if self.use_description:
            required.append(cfg.description_col)

        missing = [name for name in required if name not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {self.manifest_path}: {missing}"
            )

        self.df = self.df.dropna(subset=required).reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(
                f"Dataset is empty after filtering required values: {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def _base_sample(self, index: int) -> tuple[pd.Series, dict[str, Any]]:
        row = self.df.iloc[index]
        item: dict[str, Any] = {
            "code": _read_text_file(row[cfg.reference_code_col], self.root),
            "description": "",
        }
        if self.use_description:
            item["description"] = _read_text_file(
                row[cfg.description_col], self.root
            )
        return row, item

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, item = self._base_sample(index)
        if cfg.stage != "text_sft":
            item["image"] = _read_image(row[cfg.input_image_col], self.root)
        return item

    def get_generation_sample(self, index: int) -> dict[str, Any]:
        """Load one fixed validation sample, including its image for logging."""
        row, item = self._base_sample(index)
        if cfg.input_image_col not in self.df.columns or pd.isna(row[cfg.input_image_col]):
            raise ValueError(
                f"Generation logging requires column {cfg.input_image_col!r} "
                f"with a valid path in {self.manifest_path}."
            )
        item["image"] = _read_image(row[cfg.input_image_col], self.root)
        return item


class TikZCollator:
    def __init__(self, tokenizer, image_processor=None) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad_token_id or eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _truncate_description(self, description: str) -> str:
        if not description or not description.strip():
            return ""

        description = description.strip()

        description_ids = self.tokenizer(
            description,
            add_special_tokens=False,
        )["input_ids"]

        if len(description_ids) <= cfg.max_description_tokens:
            return description

        return self.tokenizer.decode(
            description_ids[: cfg.max_description_tokens],
            skip_special_tokens=True,
        ).strip()

    def _instruction(self, description: str) -> str:
        use_description = (
            cfg.text_use_description
            if cfg.stage == "text_sft"
            else cfg.multimodal_use_description
        )

        if use_description:
            description = self._truncate_description(description)

        return render_instruction(
            prompt=cfg.prompt,
            description=description,
            use_llm_description=use_description,
        )

    def _prompt_ids(self, description: str) -> list[int]:
        messages = [{"role": "user", "content": self._instruction(description)}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        if len(prompt_ids) > cfg.max_prompt_tokens:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeding "
                f"max_prompt_tokens={cfg.max_prompt_tokens}. "
                "Increase max_prompt_tokens or reduce max_description_tokens."
            )
        return prompt_ids

    def _encode(self, description: str, code: str) -> tuple[list[int], list[int]]:
        prompt_ids = self._prompt_ids(description)

        target_ids = self.tokenizer(
            code,
            add_special_tokens=False,
        )["input_ids"]

        if len(target_ids) > cfg.max_target_tokens:
            raise ValueError(
                f"Target has {len(target_ids)} tokens, exceeding "
                f"max_target_tokens={cfg.max_target_tokens}."
            )

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids.append(eos_id)

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        return input_ids, labels

    def generation_batch(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Create a batch containing only the prompt and optional image."""
        prompt_ids = self._prompt_ids(sample["description"])
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        batch: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if cfg.stage != "text_sft":
            if self.image_processor is None:
                raise ValueError("image_processor is required for multimodal stages.")
            image_inputs = self.image_processor(
                images=[sample["image"]],
                return_tensors="pt",
                max_num_patches=cfg.max_num_patches,
            )
            batch.update(image_inputs)

        return batch

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(x["description"], x["code"]) for x in samples]
        max_len = max(len(ids) for ids, _ in encoded)
        batch_size = len(samples)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, (ids, sample_labels) in enumerate(encoded):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            labels[i, :length] = torch.tensor(sample_labels, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if cfg.stage != "text_sft":
            if self.image_processor is None:
                raise ValueError("image_processor is required for multimodal stages.")
            image_inputs = self.image_processor(
                images=[x["image"] for x in samples],
                return_tensors="pt",
                max_num_patches=cfg.max_num_patches,
            )
            batch.update(image_inputs)

        return batch
