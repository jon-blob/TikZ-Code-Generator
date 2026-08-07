from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


def load_instruction(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Instruction not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Instruction is empty: {path}")
    return text


def render_instruction(prompt: str, description: str | None, use_llm_description: bool) -> str:
    final_prompt = prompt.strip()
    if description and description.strip() and use_llm_description:
        final_prompt += (
            "\n\nAdditionally, here is a description of the image "
            "with some creation hints:\n"
            f"{description.strip()}"
        )
    return final_prompt


def clean_code(text: str) -> str:
    code = str(text).strip()
    start, end = r"\documentclass", r"\end{document}"
    if start in code:
        code = code[code.index(start):]
    if end in code:
        code = code[:code.index(end) + len(end)]
    return code.strip()


def load_rows(manifest: Path, required: set[str], limit: int | None,) -> list[dict]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    rows = rows[:limit] if limit is not None else rows
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    return rows


def resolve(root: Path, value: str) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else root / path


def read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} is empty: {path}")
    return text


def read_optional_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB").copy()