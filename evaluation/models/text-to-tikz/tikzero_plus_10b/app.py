import os
import logging
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import BitsAndBytesConfig
from huggingface_hub import login

from detikzify.model import load
from detikzify.infer import DetikzifyPipeline

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("tikzero-plus-api")

BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "nllg/tikzero-plus-10b")
QUANTIZATION = os.getenv("QUANTIZATION", "8bit").lower()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="TikZero API")

pipeline: Optional[DetikzifyPipeline] = None


class TikZeroRequest(BaseModel):
    text: str = Field(..., description="Caption / description of the desired scientific figure")

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
    global pipeline

    try:

        quantization_config = build_quantization_config()

        pipeline = DetikzifyPipeline(*load(
            model_name_or_path=BASE_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            token= HF_TOKEN,
            quantization_config=quantization_config
        ))

    except Exception:
        logger.exception("Failed to load model")
        raise


@app.post("/tikzero")
def tikzero(request: TikZeroRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        fig = pipeline.sample(text=request.text)

        output_path = "/tmp/fig.tex"
        fig.save(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            tikz_code = f.read()

        return {
            "base_model_path": BASE_MODEL_PATH,
            "quantization": QUANTIZATION,
            "tikz": tikz_code,
        }

    except Exception as e:
        logger.exception("Request failed")
        raise HTTPException(status_code=500, detail=str(e))
