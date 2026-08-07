from unsloth import FastModel
from peft import PeftModel


BASE_MODEL = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
SFT_ADAPTER = "outputs/sft"
OUTPUT_PATH = "models/gemma4-31B-it-tikz-sft-normal"

model, processor = FastModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=9216,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(
    model,
    SFT_ADAPTER,
)

model.save_pretrained_gguf(
    OUTPUT_PATH,
    processor,
    quantization_method="q4_k_m",
)