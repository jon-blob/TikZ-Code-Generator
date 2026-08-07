from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# Konfiguration
DATA_DIR = Path("/home/jonas/Datasets/TikZ/normal")
INPUT_MANIFEST = DATA_DIR / "manifest_train.csv"
OUTPUT_MANIFEST = DATA_DIR / "manifest_train_max_2048.csv"
TOKENIZER_PATH = "/home/jonas/models/gemma-4-31B-it"

MAX_TOKENS = 2048
BATCH_SIZE = 128
TURN_SUFFIX = "<turn|>\n"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True,
)


def read_code(relative_path: str) -> str:
    path = Path(relative_path)
    if not path.is_absolute():
        path = DATA_DIR / path

    code = path.read_text(encoding="utf-8", errors="replace")

    if not code.endswith(TURN_SUFFIX):
        code += TURN_SUFFIX

    return code


df = pd.read_csv(INPUT_MANIFEST)
keep_indices = []

for start in tqdm(
    range(0, len(df), BATCH_SIZE),
    total=(len(df) + BATCH_SIZE - 1) // BATCH_SIZE,
    desc="Tokenisiere Manifest",
    unit="Batch",
):
    batch = df.iloc[start : start + BATCH_SIZE]
    texts = [read_code(path) for path in batch["code_path"]]

    lengths = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        return_length=True,
    )["length"]

    keep_indices.extend(
        index
        for index, length in zip(batch.index, lengths)
        if length <= MAX_TOKENS
    )

filtered = df.loc[keep_indices].reset_index(drop=True)
filtered.to_csv(OUTPUT_MANIFEST, index=False)

print(f"Original: {len(df):,}")
print(f"Behalten: {len(filtered):,}")
print(f"Entfernt: {len(df) - len(filtered):,}")
print(f"Gespeichert: {OUTPUT_MANIFEST}")


