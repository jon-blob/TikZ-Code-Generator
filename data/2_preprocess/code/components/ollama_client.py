"""Ollama client for code-only, image-only and image+code descriptions."""

from __future__ import annotations

import base64
import time

import requests

import config


class OllamaClient:
    VALID_TYPES = {"code", "image", "image_code"}

    def __init__(self) -> None:
        self.prompts = {
            name: path.read_text(encoding="utf-8").strip()
            for name, path in config.DESCRIPTION_PROMPTS.items()
        }

    def describe(
        self,
        description_type: str,
        *,
        code: str | None = None,
        image: bytes | None = None,
    ) -> str:
        if description_type not in self.VALID_TYPES:
            raise ValueError(f"Unknown description type: {description_type}")

        if description_type == "code":
            if code is None:
                raise ValueError("code description requires code")
            prompt = f"{self.prompts['code']}\n\n```latex\n{code}\n```"
            model = config.OLLAMA_TEXT_MODEL
            images = None
        elif description_type == "image":
            if image is None:
                raise ValueError("image description requires image")
            prompt = self.prompts["image"]
            model = config.OLLAMA_VISION_MODEL
            images = [base64.b64encode(image).decode("ascii")]
        else:
            if code is None or image is None:
                raise ValueError("image_code description requires code and image")
            prompt = f"{self.prompts['image_code']}\n\n```latex\n{code}\n```"
            model = config.OLLAMA_VISION_MODEL
            images = [base64.b64encode(image).decode("ascii")]

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_predict": config.OLLAMA_NUM_PREDICT,
                "num_ctx": config.OLLAMA_NUM_CTX,
            },
        }
        if images:
            payload["images"] = images

        error: Exception | None = None
        for attempt in range(config.OLLAMA_RETRIES + 1):
            try:
                response = requests.post(
                    f"{config.OLLAMA_URL.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=config.OLLAMA_TIMEOUT,
                )
                response.raise_for_status()
                text = str(response.json()["response"]).strip()
                if not text:
                    raise ValueError("Empty Ollama response")
                return text
            except Exception as current:
                error = current
                if attempt < config.OLLAMA_RETRIES:
                    time.sleep(2**attempt)
        raise RuntimeError(error)
