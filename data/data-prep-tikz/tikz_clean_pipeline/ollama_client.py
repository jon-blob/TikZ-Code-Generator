"""Small synchronous Ollama client used by the pipeline."""

from pathlib import Path
import time

import requests

import config
from cleaning import DeterministicCleaner


class OllamaClient:
    def __init__(self, cleaner: DeterministicCleaner):
        self.cleaner = cleaner
        self.session = requests.Session()
        self.cleaning_prompt = self._read(config.CLEANING_PROMPT_PATH)
        self.description_prompt = self._read(config.DESCRIPTION_PROMPT_PATH)

    def check_connection(self) -> None:
        response = self.session.get(
            f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags",
            timeout=30,
        )
        response.raise_for_status()

    def clean_latex(self, tex: str) -> str:
        response = self._generate(
            system=self.cleaning_prompt,
            prompt=(
                "Remove the remaining visible human-readable text from this "
                "deterministically cleaned LaTeX source. Do not intentionally "
                "remove LaTeX comments.\n\n" + tex
            ),
            num_predict=config.OLLAMA_CLEAN_NUM_PREDICT,
            temperature=0.0,
        )
        return self.cleaner.normalize_llm_latex(response, original=tex)

    def describe_latex(self, tex: str) -> str:
        response = self._generate(
            system=self.description_prompt,
            prompt=(
                "Describe the image rendered by this LaTeX/TikZ/PGFPlots "
                "source.\n\n" + tex
            ),
            num_predict=config.OLLAMA_DESCRIPTION_NUM_PREDICT,
            temperature=0.1,
        )
        description = response.strip()
        if not description:
            raise ValueError("The LLM returned an empty description.")
        return description

    def _generate(
        self,
        system: str,
        prompt: str,
        num_predict: int,
        temperature: float,
    ) -> str:
        url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": config.OLLAMA_MODEL,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_predict": num_predict,
            },
        }

        last_error: Exception | None = None
        for attempt in range(config.OLLAMA_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=config.OLLAMA_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response.json()["response"]
            except (requests.RequestException, KeyError, ValueError) as error:
                last_error = error
                if attempt < config.OLLAMA_RETRIES:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Ollama request failed: {last_error}")

    @staticmethod
    def _read(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()
