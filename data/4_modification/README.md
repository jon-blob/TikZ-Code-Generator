# Dataset modification

Downloads the already published Hugging Face dataset and adds more LLM descriptions without rerunning preprocessing or rendering.

The script reuses:

```text
2_preprocess/code/components/ollama_client.py
2_preprocess/prompts/description_code_prompt.txt
2_preprocess/prompts/description_image_prompt.txt
2_preprocess/prompts/description_image_code_prompt.txt
```

## Configure

Edit `config.py`:

```python
DESCRIPTION_TYPES = ["code", "image"]

ADDITIONAL_DESCRIPTIONS_PER_REPETITION_CLASS = {
    "low": 100,
    "medium": 200,
    "high": 600,
    "very_high": 100,
    "critical": 100,
}

OLLAMA_PARALLEL_REQUESTS = 5
```

The requested amount is **additional** and is applied separately for every:

```text
split × image class × repetition class × description type
```

Only rows where the corresponding description column is empty are candidates. Existing descriptions are never overwritten.

## Run

Authenticate once if needed:

```bash
hf auth login
```

Make sure Ollama is running, then:

```bash
cd 4_modification
python add_descriptions.py
```

The original dataset is downloaded to `4_modification/dataset/`. The modified copy is written to `4_modification/dataset_modified/`.

Set `UPLOAD = True` in `config.py` only when the modified dataset should be uploaded.
