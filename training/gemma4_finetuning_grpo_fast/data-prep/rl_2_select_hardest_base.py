from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel


# Konfiguration
DATA_DIR = Path("/home/jonas/Datasets/TikZ/normal")
INPUT_CSV = DATA_DIR / "manifest_train_max_2048.csv"
OUTPUT_CSV = DATA_DIR / "select_hardest_base.csv"

MODE = "hardest"  # "random" oder "hardest"
RANDOM_N = 500
CANDIDATE_N = 5000
TOP_K = 500
SEED = 3407

MODEL_PATH = "/home/jonas/models/gemma-4-31B-it-unsloth-bnb-4bit"
INSTRUCTION_PATH = "/home/jonas/PycharmProjects/tikzcodegenerator/evaluation/promptfoo/configs/prompt.txt"
MAX_LENGTH = 4096
LOAD_IN_4BIT = True


def path(value):
    value = Path(value)
    return value if value.is_absolute() else DATA_DIR / value


@torch.inference_mode()
def get_loss(row, model, processor, instruction):
    with Image.open(path(row.image_path)) as image_file:
        image = image_file.convert("RGB")
    code = path(row.code_path).read_text(encoding="utf-8").strip()

    user = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ],
    }
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": code}],
    }

    prompt = processor.apply_chat_template(
        [user],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = processor.apply_chat_template(
        [user, assistant],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    prompt_inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = processor(
        text=[full],
        images=[image],
        return_tensors="pt",
        add_special_tokens=False,
    )

    if inputs.input_ids.shape[1] > MAX_LENGTH:
        raise ValueError(f"Sequenz zu lang: {inputs.input_ids.shape[1]}")

    prompt_ids = prompt_inputs.input_ids[0]
    full_ids = inputs.input_ids[0]
    limit = min(len(prompt_ids), len(full_ids))
    different = (prompt_ids[:limit] != full_ids[:limit]).nonzero()
    prompt_length = different[0].item() if len(different) else limit

    labels = inputs.input_ids.clone()
    labels[:, :prompt_length] = -100

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inputs["labels"] = labels.to(device)

    return model(**inputs, use_cache=False).loss.item()


df = pd.read_csv(INPUT_CSV)

if MODE == "random":
    selected = df.sample(RANDOM_N, random_state=SEED)

elif MODE == "hardest":
    candidates = df.sample(CANDIDATE_N, random_state=SEED).copy()
    instruction = Path(INSTRUCTION_PATH).read_text(encoding="utf-8").strip()

    model, processor = FastVisionModel.from_pretrained(
        MODEL_PATH,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    model.eval()

    candidates["loss"] = [
        get_loss(row, model, processor, instruction)
        for row in tqdm(
            candidates.itertuples(index=False),
            total=len(candidates),
            desc="Berechne Loss",
        )
    ]
    selected = candidates.nlargest(TOP_K, "loss")

else:
    raise ValueError("MODE muss 'random' oder 'hardest' sein.")

output = pd.DataFrame({
    "input_image": selected.image_path.map(lambda x: str(path(x))),
    "reference_image": selected.image_path.map(lambda x: str(path(x))),
    "reference_code": selected.code_path.map(lambda x: str(path(x))),
    "llm_description": selected.vlm_description_path.map(lambda x: str(path(x))),
})

output.to_csv(OUTPUT_CSV, index=False)
print(f"{len(output)} Zeilen gespeichert: {OUTPUT_CSV}")