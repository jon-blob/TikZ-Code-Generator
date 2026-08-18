# TikZ data pipeline

Execution order:

```text
0_setup
   ↓
1_analyze                 optional
   ↓
2_preprocess
   ↓
3_inspect_clean_dataset   optional
   ↓
4_modification            optional post-processing of the published dataset
   ↓
5_export
```

## Structure

```text
data/
├── 0_setup/
├── 1_analyze/
├── 2_preprocess/
├── 3_inspect_clean_dataset/
├── 4_modification/
│   ├── README.md
│   ├── config.py
│   └── add_descriptions.py
└── 5_export/
```

`4_modification/add_descriptions.py` downloads the published Hugging Face dataset and can add further code-only, image-only, and image+code descriptions. It imports the existing Ollama client from `2_preprocess` and uses the existing prompt files, so the description logic is not duplicated.

`5_export` contains the former `4_export` scripts unchanged.
