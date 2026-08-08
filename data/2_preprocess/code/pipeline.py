"""Visible orchestration of the complete dataset pipeline."""

from __future__ import annotations

import json
import shutil

import config
from first_stage_cleaner import BENCHMARK, DATIKZ, DatasetCleaner
from enrichment import Enricher
from output import FinalExporter, HuggingFaceUploader
from second_stage_cleaner import DatasetProcessor, Renderer


class Pipeline:
    def __init__(self) -> None:
        path = config.REPORT_DIR / "statistics.json"
        self.statistics = json.loads(path.read_text()) if path.exists() else {}

    def prepare(self) -> None:
        if config.OVERWRITE:
            for path in (
                config.STAGING_DIR,
                config.METADATA_DIR,
                config.REPORT_DIR,
                config.FAILURE_DIR,
                config.EXPORT_DIR,
            ):
                shutil.rmtree(path, ignore_errors=True)

        for path in (
            config.CACHE_DIR,
            config.STAGING_DIR,
            config.METADATA_DIR,
            config.REPORT_DIR,
            config.FAILURE_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)

        if config.RUN_PREPROCESSING:
            if not config.TOKENIZER_PATH.exists():
                raise FileNotFoundError(config.TOKENIZER_PATH)
            if not config.OVERWRITE and any(config.STAGING_DIR.rglob("*.parquet")):
                raise FileExistsError("Staging output already exists. Set OVERWRITE=True or disable preprocessing.")
            Renderer().check_dependencies()

        if config.RUN_ENRICHMENT and config.DESCRIPTIONS_PER_CLASS > 0 and not config.PROMPT_PATH.exists():
            raise FileNotFoundError(config.PROMPT_PATH)

    def preprocess(self) -> None:
        datasets, cleaning_stats = DatasetCleaner().run() #first stage cleaner (Load, shuffle, filter and deduplicate the two input datasets.)
        processor = DatasetProcessor() # second stage cleaner (Parallel TikZ rendering, validation and CLIP comparison.)
        processing_stats = {
            "datikz": processor.run("datikz", datasets["datikz"], DATIKZ),
            "benchmark": processor.run("benchmark", datasets["benchmark"], BENCHMARK),
        }
        self.statistics["cleaning"] = cleaning_stats
        self.statistics["processing"] = processing_stats
        self._save_statistics()

    def enrich(self) -> None:
        self.statistics["enrichment"] = Enricher().run() # add new knowledge to the dataset (Cluster images, create splits and generate selected Ollama descriptions.)
        self._save_statistics()

    def export(self) -> None:
        self.statistics["export"] = FinalExporter().run()
        self._save_statistics()

    def upload(self) -> None:
        self.statistics["upload"] = HuggingFaceUploader().run()
        self._save_statistics()

    def run(self) -> None:
        self.prepare()
        if config.RUN_PREPROCESSING:
            self.preprocess()
        if config.RUN_ENRICHMENT:
            self.enrich()
        if config.RUN_EXPORT:
            self.export()
        if config.RUN_UPLOAD:
            self.upload()

    def _save_statistics(self) -> None:
        path = config.REPORT_DIR / "statistics.json"
        path.write_text(json.dumps(self.statistics, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    Pipeline().run()
