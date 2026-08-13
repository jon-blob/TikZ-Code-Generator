"""Configurable CLIP/SigLIP2 image encoder and similarity analyzer."""

from __future__ import annotations

from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

import config


class ClipAnalyzer:
    """Encode images with either CLIP or SigLIP2 and compare cosine similarity."""

    ENCODERS = {
        "clip": lambda: config.CLIP_MODEL,
        "siglip2": lambda: config.SIGLIP2_MODEL,
    }

    def __init__(self) -> None:
        self.encoder_name = config.IMAGE_ENCODER.lower()
        if self.encoder_name not in self.ENCODERS:
            raise ValueError(f"IMAGE_ENCODER must be one of {sorted(self.ENCODERS)}")

        if config.IMAGE_DEVICE == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.IMAGE_DEVICE

        self.device = torch.device(device)
        self.model_name = self.ENCODERS[self.encoder_name]()
        cache_dir = str(config.CACHE_DIR / self.encoder_name)

        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )

    @property
    def threshold(self) -> float:
        try:
            return float(config.IMAGE_SIMILARITY_THRESHOLDS[self.encoder_name])
        except KeyError as error:
            raise KeyError(
                f"Missing IMAGE_SIMILARITY_THRESHOLDS[{self.encoder_name!r}]"
            ) from error

    def _encode(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        model_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            features = self.model.get_image_features(**model_inputs)

        if isinstance(features, torch.Tensor):
            embeddings = features
        elif hasattr(features, "pooler_output"):
            embeddings = features.pooler_output
        elif isinstance(features, (tuple, list)) and features:
            embeddings = features[0]
        else:
            raise TypeError(
                f"Unsupported image feature output: {type(features).__name__}"
            )

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Image encoder did not return tensor embeddings")

        return F.normalize(embeddings.to(torch.float32), dim=-1)

    def analyze(self, rows: list[dict]) -> list[tuple[float, list[float]]]:
        original = [
            Image.open(BytesIO(row["original_image"])).convert("RGB")
            for row in rows
        ]
        rendered = [
            Image.open(BytesIO(row["rendered_image"])).convert("RGB")
            for row in rows
        ]

        embeddings = self._encode(original + rendered)
        size = len(rows)
        original_embeddings = embeddings[:size]
        rendered_embeddings = embeddings[size:]
        similarities = (original_embeddings * rendered_embeddings).sum(dim=-1)

        return [
            (
                float(similarities[index].cpu()),
                rendered_embeddings[index].cpu().tolist(),
            )
            for index in range(size)
        ]
