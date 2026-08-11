from __future__ import annotations

from torch.utils.data import Dataset

from common.data_utils import (
    clean_code,
    load_image,
    load_instruction,
    load_rows,
    read_optional_text,
    read_text,
    render_instruction,
    resolve,
)


class DaTikZDataset(Dataset):
    def __init__(self, cfg, manifest_name: str, split: str) -> None:
        self.cfg = cfg
        self.root = cfg.dataset_path
        self.template = load_instruction(cfg.instruction_path)
        limit = cfg.num_examples_train if split == "train" else cfg.num_examples_val
        required = {cfg.image_column, cfg.code_column, cfg.vlm_description_column}
        self.rows = load_rows(self.root / manifest_name, required, limit)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = resolve(self.root, row[self.cfg.image_column])
        code_path = resolve(self.root, row[self.cfg.code_column])
        description_path = resolve(self.root, row[self.cfg.vlm_description_column])

        image = load_image(image_path)
        code = clean_code(read_text(code_path, "Reference code"))
        description = read_optional_text(description_path, "VLM description")
        instruction = render_instruction(self.template, description, self.cfg.use_llm_description)

        return {
            "prompt": [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }],
            "completion": [{
                "role": "assistant",
                "content": [{"type": "text", "text": code}],
            }],
        }