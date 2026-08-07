# TikZ independent-mode pipeline

The pipeline downloads one Hugging Face split and runs every configured mode independently. Each mode writes its own Parquet shards. The split cache is deleted after all modes for that split are finished.

## Configuration

All settings are in `config.py`.

```python
SPLITS: dict[str, list[dict]] = {
    "our_dataset_benchmark": [
        {
            "type": "simple_llm_description",
            "num": 700,
            "absolute_num": 30_000,
            "train": True,
        },
        {
            "type": "full_cleaning",
            "num": 700,
            "absolute_num": 30_000,
            "train": True,
        },
        {
            "type": "deterministic_cleaning",
            "num": 700,
            "absolute_num": 30_000,
            "train": True,
        },
    ],
}
```

Every mode has one of two submodes:

### `train=False` — benchmark

This preserves the previous behavior:

- `num` rows are selected for active mode processing.
- `absolute_num` rows are selected in total.
- The remaining `absolute_num - num` rows contain only `code_with_text`; mode-specific columns are `None`.
- Rejected active rows are not replaced, so the written row count can be below `absolute_num`.

### `train=True` — train

- `num` is the number of **successfully saved** rows receiving active mode processing.
- `absolute_num` is the exact final number of saved rows.
- After the active rows have been completed, every remaining candidate is re-rendered from its original `code_with_text`.
- These base-row validation renders run concurrently with a bounded worker pool; they never call Ollama.
- A validation render with a LaTeX error, render failure, invalid page count, or blank output is discarded.
- Rejected rows are replaced with later candidates from the deterministic random order until exactly `absolute_num` rows have been written.
- Every successful base-row render is stored in `image_with_text`; mode-specific generated columns remain `None`.

For the example above, each mode output therefore contains exactly 30,000 rows:

- 700 successfully processed mode rows,
- 29,300 successfully rendered and validated base rows.

The worker count is configured in `config.py`:

```python
TRAIN_RENDER_WORKERS = max(1, min(8, os.cpu_count() or 1))
TRAIN_RENDER_MAX_IN_FLIGHT = TRAIN_RENDER_WORKERS * 2
```

Only the train base rows use this pool. Active rows remain sequential because
they may use Ollama. Parquet writing and failure logging also stay in the main
thread. The render workers only launch independent LaTeX and image-conversion
subprocesses.

All modes start from the same seeded random order. Train outputs may contain different final samples because failures are mode-dependent and are replaced independently.

## Output files

```text
/workspace/data/clean_parquets/our_dataset_benchmark/
├── our_dataset_benchmark-full_cleaning_part-00000.parquet
├── our_dataset_benchmark-simple_llm_description_part-00000.parquet
└── our_dataset_benchmark-deterministic_cleaning_part-00000.parquet
```

Each shard contains at most 50,000 rows.

### `simple_llm_description`

- `image_with_text`
- `code_with_text`
- `llm_description_with_text`

### `full_cleaning`

For `train=True`, this schema additionally contains `image_with_text` for the
validated base rows.

- `code_with_text`
- `image_without_text_full`
- `code_without_text_full`
- `llm_description_without_text_full`

### `deterministic_cleaning`

For `train=True`, this schema additionally contains `image_with_text` for the
validated base rows.

- `code_with_text`
- `image_without_text_deterministic`
- `code_without_text_deterministic`
- `llm_description_without_text_deterministic`

## Validation

Every actively processed row is rendered from its corresponding LaTeX source. For `train=True`, every base row is also rendered from the original code before it is saved.

A row is kept only when:

- LaTeX renders successfully,
- the LaTeX log contains no errors,
- the generated PDF has exactly one page,
- the rendered image is not effectively white.

LaTeX warnings and bad boxes are counted but do not reject a row.

`simple_llm_description` stores the newly rendered and validated image for both active and train base rows. For the cleaning modes, train base-row renders are stored in `image_with_text`, while cleaned active-row images remain in their mode-specific image column.

## Token limit

The tokenizer is loaded from:

```text
unsloth/gemma-4-31B-it-unsloth-bnb-4bit
```

The original LaTeX code may contain at most 8,000 tokens.

## Failure diagnostics

Failures are written per split and mode:

```text
/workspace/data/clean_parquets/failures/<split>-<mode>.jsonl
/workspace/data/clean_parquets/failures/<split>/<mode>/sample-000000123.txt
```

The readable text file contains the original code and the latest generated code available when the error occurred. Validation failures of train base rows use the stage `render_train_base`.

## Run

```bash
cd /workspace/tikz_clean_pipeline
pip install -r requirements.txt
python run_pipe.py
```
