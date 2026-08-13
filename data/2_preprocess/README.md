# TikZ preprocessing pipeline

This package prepares DaTikZ-V4 and the benchmark dataset, renders valid TikZ samples, computes configurable image embeddings, adds repetition/token metadata, clusters the images, creates selected descriptions, and exports final Parquet shards.

## Pipeline

```text
Input Parquets
    ↓
InputDatasetCleaner
    ├── shuffle
    ├── benchmark date/origin filter
    ├── within/cross-dataset deduplication
    └── token-length filter
    ↓
RenderedDatasetProcessor
    ├── render + validate
    ├── CLIP or SigLIP2 embeddings
    └── original/rendered cosine similarity
    ↓
DatasetEnricher
    ├── local repetition class + token_len
    ├── PCA or UMAP
    ├── KMeans or HDBSCAN
    └── selected Ollama descriptions
    ↓
FinalExporter
    ↓
train + benchmark Parquet shards
```

## Code layout

```text
2_preprocess/
├── README.md
├── prompts/
│   ├── description_code_prompt.txt
│   ├── description_image_prompt.txt
│   └── description_image_code_prompt.txt
└── code/
    ├── config.py
    ├── pipeline.py
    ├── cleaning.py
    ├── processing.py
    ├── enrichment.py
    ├── output.py
    ├── tikz_rendering.py
    └── components/
        ├── __init__.py
        ├── clip_analyzer.py
        ├── clusterer.py
        ├── ollama_client.py
        ├── renderer.py
        ├── repetition_classifier.py
        └── reporter.py
```

The three high-level processing classes are:

- `InputDatasetCleaner`: source loading, filtering, deduplication and token filtering.
- `RenderedDatasetProcessor`: rendering, validation, image similarity and image embeddings.
- `DatasetEnricher`: repetition metadata, clustering and descriptions.

Reusable implementation details live in `code/components/`.

## Image encoder, reducer and clusterer

All three choices are independent:

```python
IMAGE_ENCODER = "clip"          # "clip" or "siglip2"
CLUSTER_REDUCER = "umap"       # "pca" or "umap"
CLUSTER_ALGORITHM = "hdbscan"  # "kmeans" or "hdbscan"
```

Models:

```python
CLIP_MODEL = "openai/clip-vit-base-patch32"
SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
```

The rendered and original image are encoded by the selected model. Cosine similarity is used only for reporting low-similarity samples; these rows are not automatically removed.

The staging schema stores generic `image_embedding`, `image_similarity`, and `image_encoder` fields. If enrichment is run against staging data created with a different encoder than the current `IMAGE_ENCODER`, the pipeline raises an error instead of silently mixing embeddings.

## Repetition classification

Repetition is measured independently inside each sample. Comments are removed, numbers are normalized to `NUM`, and local 1- to 4-gram repetition coverage is computed.

```python
REPETITION_NGRAM_ORDERS = (1, 2, 3, 4)
REPETITION_NGRAM_WEIGHTS = {
    1: 0.40,
    2: 0.30,
    3: 0.20,
    4: 0.10,
}
REPETITION_MIN_REPEAT_COUNT = 2
```

Five classes are used:

```python
REPETITION_MEDIUM_THRESHOLD = 0.40
REPETITION_HIGH_THRESHOLD = 0.60
REPETITION_VERY_HIGH_THRESHOLD = 0.70
REPETITION_CRITICAL_THRESHOLD = 0.80
```

```text
score < 0.40  -> low
score < 0.60  -> medium
score < 0.70  -> high
score < 0.80  -> very_high
otherwise     -> critical
```

`token_len` is computed separately with `TOKENIZER_PATH` and `add_special_tokens=True`, matching the first-stage token filter.

## Description types

Three description types are supported:

```python
DESCRIPTION_TYPES = [
    "code",
    "image",
    "image_code",
]
```

Remove an entry to disable that description type. Examples:

```python
DESCRIPTION_TYPES = ["code"]
```

or:

```python
DESCRIPTION_TYPES = ["image", "image_code"]
```

The types map to these final columns:

| Type | Input | Column |
|---|---|---|
| `code` | reference code | `llm_description` |
| `image` | rendered image | `llm_description_image` |
| `image_code` | rendered image + reference code | `llm_description_image_code` |

`llm_description` keeps the old column name for backward compatibility.

Code-only descriptions use the text model. Image-based descriptions require a vision-capable Ollama model:

```python
OLLAMA_TEXT_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_VISION_MODEL = "qwen3-vl:30b"
```

The REST API sends rendered PNG bytes as base64 image input for `image` and `image_code`.

### Parallel Ollama requests

The preprocessing client can send several independent description requests concurrently:

```python
OLLAMA_PARALLEL_REQUESTS = 2
```

This is a global limit across all enabled description types and all sampling groups.
If a request fails, the next candidate from the same dataset/image-class/repetition-class/description-type group is used as fallback.

The Ollama server must also allow parallel inference. For two parallel requests, start/configure it with for example:

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

If the server only permits one parallel request, the Python workers can still submit concurrently but Ollama will queue them. Start with `2` and increase only if GPU/RAM usage and throughput improve.

### Description sampling

Sampling remains separate for every:

```text
dataset × image class × repetition class
```

The target count is configured per repetition class:

```python
DESCRIPTIONS_PER_REPETITION_CLASS = {
    "low": 10,
    "medium": 10,
    "high": 10,
    "very_high": 10,
    "critical": 10,
}
```

Each enabled description type attempts to reach that target independently within the same group. A failure in one description type does not discard successfully generated descriptions of another type.

## Prompt files

```text
prompts/description_code_prompt.txt
prompts/description_image_prompt.txt
prompts/description_image_code_prompt.txt
```

The image-only prompt asks the model to describe only visible content and not infer hidden source-code details. The combined prompt uses the rendered image as the visual source and the reference code for exact labels, values, coordinates and techniques.

## Final columns

```text
input_image
reference_image
reference_code
llm_description
llm_description_image
llm_description_image_code
type
source
class
repetition_class
token_len
```

## Re-running stages

Full pipeline:

```python
RUN_PREPROCESSING = True
RUN_ENRICHMENT = True
RUN_EXPORT = True
RUN_UPLOAD = False
```

Re-run enrichment/export using compatible existing staging data:

```python
OVERWRITE = True
RUN_PREPROCESSING = False
RUN_ENRICHMENT = True
RUN_EXPORT = True
RUN_UPLOAD = False
```

If `IMAGE_ENCODER` is changed, preprocessing must be run again because the staging embeddings depend on the selected encoder.

## Run

```bash
cd 2_preprocess/code
python pipeline.py
```

Authenticate to Hugging Face separately with `hf auth login` when upload is enabled. Do not commit access tokens in `config.py`.
