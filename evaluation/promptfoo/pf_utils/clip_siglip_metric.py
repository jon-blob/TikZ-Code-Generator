from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from config import MODELS, PATHS, RENDER
from pf_utils.device import get_device
from pf_utils.image_utils import load_rgb


@lru_cache(maxsize=2)
def load_model(model_key: str):
    model_name = {"clip": MODELS.clip, "siglip": MODELS.siglip}.get(model_key)
    if not model_name:
        raise ValueError(f"Unknown model: {model_key}")

    device = get_device()
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=PATHS.model_cache,
        local_files_only=MODELS.local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=PATHS.model_cache,
        local_files_only=MODELS.local_files_only,
    ).to(device).eval()
    return processor, model, device


def embedding(image_path: str | Path, model_key: str) -> torch.Tensor:
    processor, model, device = load_model(model_key)
    inputs = processor(images=load_rgb(image_path, RENDER.image_size), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            features = model.get_image_features(**inputs)
        else:
            output = model(**inputs)
            features = getattr(output, "image_embeds", None)
            if features is None:
                features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state.mean(dim=1)

    return F.normalize(features.float(), dim=-1).squeeze(0)

def embedding_training(image_path: str | Path, model_key: str) -> torch.Tensor: 
    processor, model, device = load_model(model_key) 
    inputs = processor( images=load_rgb(image_path, RENDER.image_size), return_tensors="pt", ) 
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    
    with torch.inference_mode(): 
        output = ( 
            model.get_image_features(**inputs) 
            if hasattr(model, "get_image_features") 
            else model(**inputs) 
        ) 
    features = ( 
        output 
        if isinstance(output, torch.Tensor) 
        else output.pooler_output 
    ) 
    
    return F.normalize(features.float(), dim=-1).squeeze(0)


def image_cosine_similarity(image_a: str | Path, image_b: str | Path, model_key: str, train_mode=False) -> float:
    if not train_mode:
        return float(torch.dot(embedding(image_a, model_key), embedding(image_b, model_key)).item())
    elif train_mode:
        return float(torch.dot(embedding_training(image_a, model_key), embedding_training(image_b, model_key)).item())
