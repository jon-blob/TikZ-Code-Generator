import os
import logging
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from automatikz.infer import TikzGenerator, load

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("automatikz-api")

BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "nllg/tikz-clima-13b")
HF_TOKEN = os.getenv("HF_TOKEN")
QUANTIZATION = os.getenv("QUANTIZATION", "8bit").lower()

app = FastAPI(title="AutomaTikZ API")

generator: Optional[TikzGenerator] = None


class AutomaTikZRequest(BaseModel):
    text: str = Field(..., description="Caption / description of the desired TikZ figure")


def build_quantization_config():
    if QUANTIZATION in ("none", "false", "0", "no"):
        return None
    if QUANTIZATION == "8bit":
        from transformers import BitsAndBytesConfig

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
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    raise ValueError("Invalid QUANTIZATION. Use 'none', '8bit' or '4bit'.")


@app.on_event("startup")
def load_model():
    global generator

    try:
        quantization_config = build_quantization_config()

        model, processor = load(
            BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            token=HF_TOKEN,
            quantization_config=quantization_config,
        )

        generator = TikzGenerator(model, processor, stream=False)

    except Exception:
        logger.exception("Failed to load AutomaTikZ model")
        raise


@app.post("/automatikz")
def automatikz(request: AutomaTikZRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        tikzdoc = generator(request.text)

        if not tikzdoc.has_content:
            raise HTTPException(status_code=500, detail="AutomaTikZ generated empty output")

        output_path = "/tmp/fig.tex"
        tikzdoc.save(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            tikz_code = f.read()

        return {
            "model": BASE_MODEL_PATH,
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
        "model": BASE_MODEL_PATH,
        "model_loaded": generator is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }