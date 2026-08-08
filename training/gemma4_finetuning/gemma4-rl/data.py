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
    def __init__(self, cfg, processor) -> None:
        self.cfg = cfg
        self.processor = processor
        self.root = cfg.dataset_path
        self.template = load_instruction(cfg.instruction_path)
        required = {cfg.image_column, cfg.code_column, cfg.vlm_description_column}
        self.rows = load_rows(self.root / cfg.manifest, required, cfg.num_examples)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = resolve(self.root, row[self.cfg.image_column])
        code_path = resolve(self.root, row[self.cfg.code_column])
        description_path = resolve(self.root, row[self.cfg.vlm_description_column])

        image = load_image(image_path)
        answer = clean_code(read_text(code_path, "Reference code"))
        description = read_optional_text(description_path, "VLM description")
        instruction = render_instruction(self.template, description, self.cfg.use_llm_description)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.enable_thinking,
        )
        return {"prompt": prompt, "image": image, "answer": answer}
