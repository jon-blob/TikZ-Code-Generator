import os
import logging
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import BitsAndBytesConfig
from huggingface_hub import login

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("tikzilla-8b-rl-api")

BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "nllg/TikZilla-8B-RL")
QUANTIZATION = os.getenv("QUANTIZATION", "8bit").lower()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="TikZilla-8B-RL API")

model = None
tokenizer = None
gen_config = None


class TikZillaRequest(BaseModel):
    text: str = Field(..., description="Caption / description of the desired TikZ figure")

def build_quantization_config():
    if QUANTIZATION in ("none", "false", "0", "no"):
        return None

    if QUANTIZATION == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=[
                "vision_tower",
                "vision_model",
                "visual",
                "multi_modal_projector",
                "mm_projector",
                "projector",
                "image_projector",
                "connector",
                "encoder",
                "decoder",
                "embedder",
            ],
        )

    if QUANTIZATION == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    raise ValueError("Invalid QUANTIZATION. Use 'none', '8bit' or '4bit'.")


@app.on_event("startup")
def load_model():
    global model, tokenizer, gen_config

    try:

        quantization_config = build_quantization_config()

        model_id = BASE_MODEL_PATH

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token= HF_TOKEN,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token= HF_TOKEN,
            quantization_config=quantization_config
        )

        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        pad_token_id = (
            tokenizer.pad_token_id
            or tokenizer.eos_token_id
            or eos_token_id
        )

        gen_config = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            max_new_tokens=2048,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    except Exception:
        logger.exception("Failed to load model")
        raise


@app.post("/tikzilla_rl")
def tikzilla(request: TikZillaRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:

        messages = [
            {
                "role": "user",
                "content": (
                    "Generate a complete LaTeX document that contains a TikZ figure according to the following requirements:\n"
                    + request.text +
                    "\nWrap your code using \\documentclass[tikz]{standalone}, and include \\begin{document}...\\end{document}. "
                    "Only output valid LaTeX code with no extra text."
                ),
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **inputs,
            generation_config=gen_config
        )

        response_ids = output_ids[0][len(inputs["input_ids"][0]):]

        tikz_code = tokenizer.decode(response_ids, skip_special_tokens=True)

        return {
            "base_model_path": BASE_MODEL_PATH,
            "quantization": QUANTIZATION,
            "tikz": tikz_code,
        }

    except Exception as e:
        logger.exception("Request failed")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": BASE_MODEL_PATH,
        "quantization": QUANTIZATION,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
