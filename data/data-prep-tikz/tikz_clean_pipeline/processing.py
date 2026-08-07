"""Mode-specific processing for one dataset sample."""

from enum import Enum

import config
from cleaning import DeterministicCleaner
from ollama_client import OllamaClient
from rendering import ValidatedRenderer


class ProcessingMode(str, Enum):
    SIMPLE_LLM_DESCRIPTION = "simple_llm_description"
    DETERMINISTIC_CLEANING = "deterministic_cleaning"
    FULL_CLEANING = "full_cleaning"


class SampleProcessingError(RuntimeError):
    """Error enriched with the stage and latest generated LaTeX code."""

    def __init__(self, stage: str, latest_code: str, cause: Exception):
        super().__init__(str(cause))
        self.stage = stage
        self.latest_code = latest_code
        self.cause = cause


class SampleProcessor:
    def __init__(
        self,
        cleaner: DeterministicCleaner,
        ollama: OllamaClient,
        renderer: ValidatedRenderer,
    ) -> None:
        self.cleaner = cleaner
        self.ollama = ollama
        self.renderer = renderer

    def base_only(self, output_mode: ProcessingMode, original_code: str) -> dict:
        """Create a row whose mode-specific outputs are all None."""

        return self._base_row(output_mode, original_code)

    def validated_base_only(
        self,
        output_mode: ProcessingMode,
        original_code: str,
    ) -> dict:
        """Render, validate, and store the original image for a train base row."""

        try:
            rendered = self.renderer.render(original_code)
        except Exception as error:
            raise SampleProcessingError(
                "render_train_base",
                original_code,
                error,
            ) from error

        row = self._base_row(output_mode, original_code)
        row[config.IMAGE_WITH_TEXT_COL] = {
            "bytes": rendered,
            "path": None,
        }
        return row

    def process(
        self,
        mode: ProcessingMode,
        original_code: str,
    ) -> dict:
        row = self._base_row(mode, original_code)

        if mode is ProcessingMode.SIMPLE_LLM_DESCRIPTION:
            try:
                rendered = self.renderer.render(original_code)
            except Exception as error:
                raise SampleProcessingError(
                    "render_original",
                    original_code,
                    error,
                ) from error

            row[config.IMAGE_WITH_TEXT_COL] = {
                "bytes": rendered,
                "path": None,
            }

            try:
                row[config.DESCRIPTION_WITH_TEXT_COL] = self.ollama.describe_latex(
                    original_code
                )
            except Exception as error:
                raise SampleProcessingError(
                    "llm_description_with_text",
                    original_code,
                    error,
                ) from error
            return row

        try:
            deterministic_code = self.cleaner.remove_standard_text(original_code)
        except Exception as error:
            raise SampleProcessingError(
                "deterministic_cleaning",
                original_code,
                error,
            ) from error

        if not deterministic_code:
            error = ValueError("Deterministic cleaning returned empty LaTeX.")
            raise SampleProcessingError(
                "deterministic_cleaning",
                deterministic_code,
                error,
            ) from error

        if mode is ProcessingMode.DETERMINISTIC_CLEANING:
            try:
                rendered = self.renderer.render(deterministic_code)
            except Exception as error:
                raise SampleProcessingError(
                    "render_deterministic",
                    deterministic_code,
                    error,
                ) from error

            row[config.IMAGE_WITHOUT_TEXT_DETERMINISTIC_COL] = {
                "bytes": rendered,
                "path": None,
            }
            row[config.CODE_WITHOUT_TEXT_DETERMINISTIC_COL] = deterministic_code

            try:
                row[config.DESCRIPTION_WITHOUT_TEXT_DETERMINISTIC_COL] = (
                    self.ollama.describe_latex(deterministic_code)
                )
            except Exception as error:
                raise SampleProcessingError(
                    "llm_description_deterministic",
                    deterministic_code,
                    error,
                ) from error
            return row

        if mode is ProcessingMode.FULL_CLEANING:
            try:
                full_code = self.ollama.clean_latex(deterministic_code)
            except Exception as error:
                raise SampleProcessingError(
                    "llm_cleaning",
                    deterministic_code,
                    error,
                ) from error

            try:
                rendered = self.renderer.render(full_code)
            except Exception as error:
                raise SampleProcessingError(
                    "render_full",
                    full_code,
                    error,
                ) from error

            row[config.IMAGE_WITHOUT_TEXT_FULL_COL] = {
                "bytes": rendered,
                "path": None,
            }
            row[config.CODE_WITHOUT_TEXT_FULL_COL] = full_code

            try:
                row[config.DESCRIPTION_WITHOUT_TEXT_FULL_COL] = (
                    self.ollama.describe_latex(full_code)
                )
            except Exception as error:
                raise SampleProcessingError(
                    "llm_description_full",
                    full_code,
                    error,
                ) from error
            return row

        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _base_row(mode: ProcessingMode, original_code: str) -> dict:
        row = {config.CODE_WITH_TEXT_COL: original_code}

        if mode is ProcessingMode.SIMPLE_LLM_DESCRIPTION:
            row.update(
                {
                    config.IMAGE_WITH_TEXT_COL: None,
                    config.DESCRIPTION_WITH_TEXT_COL: None,
                }
            )
        elif mode is ProcessingMode.DETERMINISTIC_CLEANING:
            row.update(
                {
                    config.IMAGE_WITHOUT_TEXT_DETERMINISTIC_COL: None,
                    config.CODE_WITHOUT_TEXT_DETERMINISTIC_COL: None,
                    config.DESCRIPTION_WITHOUT_TEXT_DETERMINISTIC_COL: None,
                }
            )
        elif mode is ProcessingMode.FULL_CLEANING:
            row.update(
                {
                    config.IMAGE_WITHOUT_TEXT_FULL_COL: None,
                    config.CODE_WITHOUT_TEXT_FULL_COL: None,
                    config.DESCRIPTION_WITHOUT_TEXT_FULL_COL: None,
                }
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return row
