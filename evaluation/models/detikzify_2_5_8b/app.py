from operator import itemgetter
from io import BytesIO
from typing import Optional
import os
import logging

import asyncio
import gc

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from transformers import BitsAndBytesConfig, set_seed
from detikzify.model import load
from detikzify.infer import DetikzifyPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("detikzify-api")


MODEL_PATH = os.getenv("MODEL_PATH", "/models/detikzify_2_5_8b")
QUANTIZATION = os.getenv("QUANTIZATION", "8bit").lower()

app = FastAPI(title="DeTikZify API")

pipeline: Optional[DetikzifyPipeline] = None


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
            ],
        )

    raise ValueError(
        f"Invalid QUANTIZATION={QUANTIZATION!r}. Use '8bit' or 'none'."
    )


@app.on_event("startup")
def load_model():
    global pipeline
    logger.info("Loading model")
    try:
        quantization_config = build_quantization_config()

        load_kwargs = {
            "model_name_or_path": MODEL_PATH,
            "device_map": "auto",
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model, processor = load(**load_kwargs)
        pipeline = DetikzifyPipeline(model, processor)

        logger.info("model loaded")

    except Exception:
        logger.exception("Failed to load model")
        raise



inference_lock = asyncio.Lock()


@app.post("/detikzify")
async def detikzify(
    image: UploadFile = File(...),
    seed: int = Form(42, ge=0, le=2**32 - 1),
    expansions: int = Form(20, ge=1),
):
    logger.info(
        "REQUEST RECEIVED: image=%r seed=%r expansions=%r",
        image.filename,
        seed,
        expansions,
    )

    if pipeline is None:
        logger.info("Model is not loaded yet")
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.exception("Invalid image")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {e}",
        )

    
    with open("/app/image.png", "wb") as handle:
        handle.write(image_bytes)
    
    best_fig = None

    try:
        async with inference_lock:
            # Setzt Python-, NumPy- und PyTorch-Zufallszustände.
            set_seed(seed)

            best_score = None

            with torch.inference_mode():
                for score, fig in pipeline.simulate(
                    image=pil_image,
                    expansions=expansions,
                    timeout=None,
                ):
                    if best_score is None or score > best_score:
                        best_score = score
                        best_fig = fig

            if best_fig is None or best_score is None:
                logger.error("No TikZ figure generated")
                raise HTTPException(
                    status_code=500,
                    detail="No TikZ figure generated",
                )

            tikz_code = best_fig.code

        logger.info(
            "Returning TikZ result: seed=%r expansions=%r score=%r",
            seed,
            expansions,
            float(best_score),
        )

        return JSONResponse({
            "model_path": MODEL_PATH,
            "quantization": QUANTIZATION,
            "seed": seed,
            "expansions": expansions,
            "score": float(best_score),
            "tikz": tikz_code,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("DeTikZify request failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        best_fig = None
        pil_image.close()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
