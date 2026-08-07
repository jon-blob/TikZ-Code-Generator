"""Hugging Face split loading and mode-specific Parquet output."""

import gc
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image as HFImage, Value, load_dataset

import config


BASE_FEATURES = {
    config.CODE_WITH_TEXT_COL: Value("string"),
}

OUTPUT_FEATURES = {
    "simple_llm_description": Features(
        {
            config.IMAGE_WITH_TEXT_COL: HFImage(),
            **BASE_FEATURES,
            config.DESCRIPTION_WITH_TEXT_COL: Value("string"),
        }
    ),
    "full_cleaning": Features(
        {
            **BASE_FEATURES,
            config.IMAGE_WITHOUT_TEXT_FULL_COL: HFImage(),
            config.CODE_WITHOUT_TEXT_FULL_COL: Value("string"),
            config.DESCRIPTION_WITHOUT_TEXT_FULL_COL: Value("string"),
        }
    ),
    "deterministic_cleaning": Features(
        {
            **BASE_FEATURES,
            config.IMAGE_WITHOUT_TEXT_DETERMINISTIC_COL: HFImage(),
            config.CODE_WITHOUT_TEXT_DETERMINISTIC_COL: Value("string"),
            config.DESCRIPTION_WITHOUT_TEXT_DETERMINISTIC_COL: Value("string"),
        }
    ),
}

# Train outputs additionally keep the freshly rendered original image for every
# validated base row. The active cleaning rows may leave this column empty,
# because their corresponding cleaned image is stored in the mode-specific
# image column.
TRAIN_OUTPUT_FEATURES = {
    "simple_llm_description": OUTPUT_FEATURES["simple_llm_description"],
    "full_cleaning": Features(
        {
            config.IMAGE_WITH_TEXT_COL: HFImage(),
            **BASE_FEATURES,
            config.IMAGE_WITHOUT_TEXT_FULL_COL: HFImage(),
            config.CODE_WITHOUT_TEXT_FULL_COL: Value("string"),
            config.DESCRIPTION_WITHOUT_TEXT_FULL_COL: Value("string"),
        }
    ),
    "deterministic_cleaning": Features(
        {
            config.IMAGE_WITH_TEXT_COL: HFImage(),
            **BASE_FEATURES,
            config.IMAGE_WITHOUT_TEXT_DETERMINISTIC_COL: HFImage(),
            config.CODE_WITHOUT_TEXT_DETERMINISTIC_COL: Value("string"),
            config.DESCRIPTION_WITHOUT_TEXT_DETERMINISTIC_COL: Value("string"),
        }
    ),
}



class SplitLoader:
    def load(self, split_name: str):
        pattern = (
            f"hf://datasets/{config.HF_REPO_ID}/"
            f"{split_name}_part-*.parquet"
        )
        dataset = load_dataset(
            "parquet",
            data_files={split_name: pattern},
            split=split_name,
            streaming=False,
            cache_dir=str(config.HF_CACHE_DIR / "datasets"),
        )

        required = {config.CODE_WITH_TEXT_COL}
        missing = required.difference(dataset.column_names)
        if missing:
            raise KeyError(f"Missing columns in {split_name}: {sorted(missing)}")

        return dataset.select_columns([config.CODE_WITH_TEXT_COL])

    @staticmethod
    def clear_cache() -> None:
        gc.collect()
        shutil.rmtree(config.HF_CACHE_DIR, ignore_errors=True)


class ParquetShardWriter:
    """Write one mode to independent Parquet shards."""

    def __init__(
        self,
        split_name: str,
        mode_name: str,
        include_original_image: bool = False,
    ):
        feature_map = (
            TRAIN_OUTPUT_FEATURES
            if include_original_image
            else OUTPUT_FEATURES
        )
        if mode_name not in feature_map:
            raise ValueError(f"Unknown output mode: {mode_name}")

        self.split_name = split_name
        self.mode_name = mode_name
        self.output_dir = config.OUTPUT_DIR / split_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pattern = f"{split_name}-{mode_name}_part-*.parquet"
        existing = list(self.output_dir.glob(pattern))
        if existing and config.OVERWRITE_MODE_OUTPUT:
            for path in existing:
                path.unlink()
        elif existing:
            raise FileExistsError(
                f"Output already exists for {split_name}/{mode_name}."
            )

        self.schema = feature_map[mode_name].arrow_schema
        self.writer: pq.ParquetWriter | None = None
        self.buffer: list[dict] = []
        self.rows_in_file = 0
        self.part = 0

    def add(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= config.ARROW_WRITE_BATCH_SIZE:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._close_file()

    def _flush(self) -> None:
        while self.buffer:
            remaining = config.ROWS_PER_PARQUET - self.rows_in_file
            chunk = self.buffer[:remaining]

            if self.writer is None:
                path = self.output_dir / (
                    f"{self.split_name}-{self.mode_name}_part-{self.part:05d}.parquet"
                )
                self.writer = pq.ParquetWriter(
                    path,
                    self.schema,
                    compression="zstd",
                )

            table = pa.Table.from_pylist(chunk, schema=self.schema)
            self.writer.write_table(table)
            del self.buffer[: len(chunk)]
            self.rows_in_file += len(chunk)

            if self.rows_in_file >= config.ROWS_PER_PARQUET:
                self._close_file()
                self.part += 1

    def _close_file(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.rows_in_file = 0
