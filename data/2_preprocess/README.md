# TikZ preprocessing pipeline

The pipeline keeps the data sources separate:

- DaTikZ becomes `train` and `val`.
- The local dataset becomes `benchmark`.
- Benchmark duplicates are removed from DaTikZ.

## Files

- `code/config.py`: all settings
- `code/pipeline.py`: pipeline order
- `code/first_stage_cleaner.py`: loading, date filtering, deduplication and token filtering
- `code/second_stage_cleaner.py`: rendering, validation and CLIP embeddings
- `code/enrichment.py`: joint clustering, splits and Ollama descriptions
- `code/output.py`: Parquet export and Hugging Face upload
- `code/tikz_rendering.py`: TikZ rendering implementation

## Joint clustering

CLIP embeddings from DaTikZ and benchmark are used together to fit one shared
`IncrementalPCA` and one shared `MiniBatchKMeans` model. The embedding batches
are read in round-robin order and processed incrementally, so the complete
embedding matrices do not have to fit into memory.

Only the embeddings are combined. The samples themselves remain in their
original datasets, and the resulting cluster labels are written back to each
source separately.

The datasets used for fitting are configured in `config.py`:

```python
CLUSTER_DATASETS = ("datikz", "benchmark")
```

## Run

Edit the paths and settings in `code/config.py`, then run from the code folder:

```bash
cd code
python pipeline.py
```

The final columns are:

- `input_image`
- `reference_image`
- `reference_code`
- `llm_description`
- `type`
- `source`
- `class`
