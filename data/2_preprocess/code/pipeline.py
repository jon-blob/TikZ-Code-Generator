"""Visible orchestration of the complete dataset pipeline."""

from __future__ import annotations

import json
import shutil

import config
from cleaning import BENCHMARK, DATIKZ, InputDatasetCleaner
from components import Renderer
from enrichment import DatasetEnricher
from output import FinalExporter, HuggingFaceUploader
from processing import RenderedDatasetProcessor


class Pipeline:
    def __init__(self) -> None:
        path = config.REPORT_DIR / "statistics.json"
        self.statistics = json.loads(path.read_text()) if path.exists() else {}

    def prepare(self) -> None:
        if config.OVERWRITE:
            if config.RUN_PREPROCESSING:
                shutil.rmtree(config.STAGING_DIR, ignore_errors=True)
                shutil.rmtree(config.FAILURE_DIR, ignore_errors=True)
            if config.RUN_ENRICHMENT:
                shutil.rmtree(config.METADATA_DIR, ignore_errors=True)
            if config.RUN_EXPORT:
                shutil.rmtree(config.EXPORT_DIR, ignore_errors=True)
            shutil.rmtree(config.REPORT_DIR, ignore_errors=True)

        for path in (
            config.CACHE_DIR,
            config.STAGING_DIR,
            config.METADATA_DIR,
            config.REPORT_DIR,
            config.FAILURE_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)

        if (config.RUN_PREPROCESSING or config.RUN_ENRICHMENT) and not config.TOKENIZER_PATH.exists():
            raise FileNotFoundError(config.TOKENIZER_PATH)

        if config.RUN_PREPROCESSING:
            if not config.OVERWRITE and any(config.STAGING_DIR.rglob("*.parquet")):
                raise FileExistsError(
                    "Staging output already exists. Set OVERWRITE=True or disable preprocessing."
                )
            Renderer().check_dependencies()

        if config.RUN_ENRICHMENT:
            invalid = set(config.DESCRIPTION_TYPES).difference(config.DESCRIPTION_PROMPTS)
            if invalid:
                raise ValueError(f"Unknown DESCRIPTION_TYPES: {sorted(invalid)}")
            for description_type in config.DESCRIPTION_TYPES:
                path = config.DESCRIPTION_PROMPTS[description_type]
                if not path.exists():
                    raise FileNotFoundError(path)

    def preprocess(self) -> None:
        datasets, cleaning_stats = InputDatasetCleaner().run()
        processor = RenderedDatasetProcessor()
        processing_stats = {
            "datikz": processor.run("datikz", datasets["datikz"], DATIKZ),
            "benchmark": processor.run("benchmark", datasets["benchmark"], BENCHMARK),
        }
        self.statistics["cleaning"] = cleaning_stats
        self.statistics["processing"] = processing_stats
        self._save_statistics()

    def enrich(self) -> None:
        self.statistics["enrichment"] = DatasetEnricher().run()
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
        path.write_text(
            json.dumps(self.statistics, indent=2, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    Pipeline().run()
