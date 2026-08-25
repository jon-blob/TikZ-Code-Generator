# TikZ Data Pipeline

This pipeline prepares raw TikZ data for later **Supervised Fine-Tuning (SFT)**, **Reinforcement Learning (RL)**, and evaluation.

The central idea is to turn heterogeneous raw data into a controlled dataset in which every sample

- contains valid and renderable TikZ code,
- includes a reproducible reference image,
- is assigned to a visual class,
- is characterized by its local code repetition,
- optionally contains a textual description,
- and can subsequently be selected specifically for SFT, RL, or benchmarking.

The directory structure largely corresponds to the conceptual steps of the data preparation process.

---

## Overall Structure

```text
data/
├── 0_setup/
├── 1_analyze/
│   ├── classification/
│   └── repetition_detection/
├── 2_preprocess/
│   ├── code/
│   │   └── components/
│   └── prompts/
├── 3_inspect_clean_dataset/
├── 4_modification/
└── 5_export/
```

The data flow can be simplified as follows:

```text
Raw TikZ datasets
        │
        ▼
1_analyze
Investigate methods for classes and repetition
        │
        ▼
2_preprocess
clean → render → analyze → cluster → describe
        │
        ▼
3_inspect_clean_dataset
Visually inspect the result
        │
        ▼
4_modification
Optionally extend the existing dataset
        │
        ▼
5_export
Create targeted datasets for SFT, RL, and evaluation
```

---

# 0_setup

`0_setup` describes the technical foundation of the pipeline.

```text
0_setup/
├── environment.yaml
└── install_texlive_full.sh
```

The data pipeline essentially requires three types of infrastructure:

1. **Python/ML environment** for Parquet processing, embeddings, clustering, and export.
2. **LaTeX/TikZ environment** so that reference code can actually be compiled and rendered.
3. **Local models**, including models for tokenization, image embeddings, and optional description generation.

The setup directory contains only this technical foundation. The actual data logic is implemented entirely in the following steps.

---

# 1_analyze

`1_analyze` contains exploratory notebooks used to investigate the structure of the raw dataset.

```text
1_analyze/
├── classification/
│   ├── ...image.ipynb
│   └── ...code.ipynb
└── repetition_detection/
    ├── find_common_tikz_ngrams.ipynb
    └── test_local_tikz_ngram_repetition_4_classes.ipynb
```

This directory does not directly modify the final dataset. Its purpose is to identify suitable methods for the later automated processing steps.

## 1.1 Classification

The classification notebooks investigate whether TikZ examples can automatically be divided into meaningful **visual classes**.

The basic idea is:

```text
Image or TikZ code
        │
        ▼
Embedding model
        │
        ▼
High-dimensional feature vector
        │
        ▼
Dimensionality reduction
        │
        ▼
Clustering
        │
        ▼
Visual classes
```

### Image-Based Classification

The rendered images are converted into embeddings using a vision encoder such as CLIP or SigLIP.

This causes similar diagrams to lie closer together in embedding space. For example, groups may form for

- graphs,
- geometric drawings,
- diagrams with boxes,
- arrow diagrams,
- plots,
- or other recurring visual structures.

Because these embeddings are very high-dimensional, methods such as **PCA** or **UMAP** are investigated before the data is clustered.

Clustering methods including **KMeans** and **HDBSCAN** are compared.

KMeans enforces a fixed number of groups. HDBSCAN, by contrast, can discover clusters automatically and mark samples that do not clearly belong to any group as `noise`.

### Code-Based Classification

In addition, the pipeline investigates whether similar classes can be derived directly from TikZ code embeddings.

The production pipeline then uses one shared, consistent classification so that SFT, RL, and benchmark samples can later be selected specifically according to visual structure.

---

## 1.2 Repetition Detection

TikZ code can be structured in very different ways. Some examples consist of many individually written commands, while others contain very strong local repetition.

This is particularly relevant for later training: with repetitive TikZ code, a model may be more likely to fall into loops or generate long sequences of similar tokens.

For this reason, the pipeline analyzes local **n-gram repetition**.

Before the analysis, the code is simplified:

- comments are removed,
- numbers are normalized,
- TikZ/LaTeX commands and additional symbols are tokenized.

Then, for several n-gram lengths, the pipeline measures what fraction of local sequences occurs more than once.

For each n-gram length, a coverage value is computed:

```text
repeated n-gram positions
──────────────────────────
   all n-gram positions
```

The coverage values from different n-gram lengths are then combined into a shared repetition score using weights.

```text
Repetition Score
    = weighted average
      of multiple local n-gram coverages
```

Discrete classes are created from this continuous score:

```text
low
medium
high
very_high
critical
```

This makes repetition an explicit property of every sample, allowing it to be used later for targeted dataset selection and RL.

---

# 2_preprocess

`2_preprocess` is the **core of the data pipeline**.

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
        ├── renderer.py
        ├── clip_analyzer.py
        ├── clusterer.py
        ├── repetition_classifier.py
        ├── ollama_client.py
        └── reporter.py
```

Conceptually, the preprocessing pipeline consists of four phases:

```text
Cleaning
   ↓
Rendering + Validation
   ↓
Enrichment
   ↓
Final Export
```

---

## 2.1 Cleaning

`cleaning.py` standardizes and cleans the different source datasets.

### Handle Data Sources Separately

Training data and benchmark data are initially loaded independently.

This is important because benchmark samples must not later be duplicated accidentally by samples from the training dataset.

### Normalize Code

TikZ code is converted into a more canonical form, for example by removing outer Markdown code fences and standardizing line breaks.

### Remove Duplicates

A hash is computed for normalized code.

This is used to detect two types of duplicates:

1. Duplicates **within the same dataset**.
2. Duplicates **between training data and benchmark data**.

If there is an overlap, the benchmark takes priority. The corresponding sample is removed from the training pool.

The idea is:

```text
Benchmark ∩ Training = ∅
```

with respect to identical normalized TikZ code.

This reduces data leakage between training and evaluation.

### Control Token Length

The code is additionally tokenized using the model tokenizer that will be relevant later.

This allows extremely long examples to be excluded before the expensive rendering step.

---

## 2.2 Rendering and Validation

After cleaning, every remaining TikZ code sample is actually rendered.

```text
TikZ code
   │
   ▼
LaTeX Engine
   │
   ▼
PDF
   │
   ▼
PNG
```

`renderer.py` and `tikz_rendering.py` handle this task.

A sample is not checked only for successful compilation. The pipeline also verifies whether the result is suitable as a training sample.

This includes in particular:

- a successful LaTeX run,
- no critical LaTeX errors,
- exactly one relevant page, i.e. no multi-page PDFs,
- actually visible image content rather than blank white pages,
- normalized image size and canvas, with larger images than in DaTikZ-V4 so that text in the image is easier to read.

Samples that cannot be rendered or are obviously empty are discarded and documented separately.

### Comparison with the Original Image

Many source datasets already contain an original image in addition to the TikZ code.

The pipeline renders the code again and compares the two images in embedding space:

```text
Original image ──► Image Encoder ──► embedding_original

TikZ render    ──► Image Encoder ──► embedding_rendered

                        │
                        ▼
                 cosine similarity
```

The similarity primarily serves as a **quality signal and diagnostic**.

A low score may indicate that the code and the original image do not match well. Such cases are reported, but they are not automatically removed based on this signal alone.

At the same time, the embedding of the newly rendered image is stored. This embedding is later used for clustering.

---

## 2.3 Staging

After successful rendering, a sample first enters an internal staging dataset.

This dataset already contains the key stable information:

```text
sample_id
reference_code
rendered_image
source
image_embedding
image_similarity
image_encoder
```

The separation between **staging** and the **final dataset** is important: expensive steps such as rendering and embedding computation do not necessarily have to be repeated when only later metadata logic is changed.

---

## 2.4 Enrichment

`enrichment.py` adds additional properties to the rendered samples.

```text
Staging Sample
   ├──► Repetition Class
   ├──► Image Class
   ├──► Token Length
   └──► LLM Description
```

### Repetition Class

`repetition_classifier.py` implements the local n-gram idea investigated in `1_analyze` for production use.

Each sample is assigned a repetition class from `low` to `critical`.

The token length according to the actual model tokenizer is also stored.

### Image Class

`clusterer.py` clusters the image embeddings.

An important point is that the relevant data splits are processed in the **same embedding and clustering space**.

This means, for example, that `class_7` in the training dataset represents the same visual region as `class_7` in the benchmark.

Conceptually:

```text
Train embeddings ─────┐
                      ├──► shared reducer ──► shared clustering
Benchmark embeddings ─┘
```

The classes are not manually defined semantic labels. They represent automatically discovered visual groups.

With a density-based clusterer, samples without clear group membership can be treated as `noise`.

### LLM Descriptions

Additional natural-language descriptions can be generated for selected samples.

There are three possible perspectives:

```text
code       → description based only on TikZ code
image      → description based only on the rendered image
image_code → description based jointly on image and code - was not used
```

The descriptions are intended to provide the later VLM with additional instruction or context data.

Importantly, they are not simply generated for random samples. Selection can be stratified by

```text
Data split
× Image Class
× Repetition Class
```

This makes it possible to provide descriptions specifically for rarer visual structures or more difficult repetition groups as well.

---

## 2.5 Final Preprocessing Dataset

`output.py` then combines rendered data and metadata.

A final sample conceptually contains:

```text
input_image
reference_image
reference_code
llm_description*
class
repetition_class
token_len
source
```

In this pipeline, `input_image` and `reference_image` both originate from the validated TikZ render.

The resulting dataset is therefore **not yet a concrete training split**. Instead, it is the clean master dataset from which different training and evaluation datasets are assembled in `5_export`.

---

# 3_inspect_clean_dataset

After automated preprocessing, a visual quality-control step follows.

```text
3_inspect_clean_dataset/
└── inspect_tikz_classes.ipynb
```

The notebook focuses in particular on:

- the distribution of the automatically discovered classes,
- example images within each class,
- differences between train and benchmark,
- and unusual or difficult-to-interpret clusters.

A typical view is a matrix of random samples from one class.

This makes it possible to verify whether a numerical cluster actually corresponds to a recognizable visual structure.

This step is important because clustering initially finds only geometric structures in embedding space. Only visual inspection reveals whether these groups are meaningful for later dataset selection.

---

# 4_modification

`4_modification` allows an already generated dataset to be **extended afterward** without rerunning the expensive previous steps.

```text
4_modification/
├── config.py
├── add_descriptions.py
└── README.md
```

The main use case is adding more LLM descriptions.

The basic idea is:

```text
already cleaned dataset
        │
        ▼
Find samples without the desired description
        │
        ▼
Group by class and repetition
        │
        ▼
Generate additional descriptions
        │
        ▼
Extended version of the dataset
```

Existing descriptions are not regenerated from the raw data. Instead, the process preferentially fills samples that do not yet have the desired descriptions.

This makes it possible to improve coverage of specific classes or repetition groups without having to

- deduplicate,
- render,
- compute embeddings,
- or cluster

again.

---

# 5_export

`5_export` converts the cleaned master dataset into the **concrete file formats and subsets** used by training and evaluation.

```text
5_export/
├── config.py
├── helpers.py
├── export_train.py
├── export_benchmark.py
├── main_sft.py
└── main_rl.py
```

No fundamental data cleaning takes place here anymore. Instead, the following decision is made:

> Which of the already validated samples should be used for which purpose?

---

## 5.1 Shared Export Concept

A training sample is split into a file-based structure:

```text
sample/
├── input_image.png
├── reference_image.png
├── llm_description.txt
└── reference_code.txt
```

In addition, a manifest stores metadata such as

- visual class,
- repetition class,
- token length,
- data source,
- and the description source used.

This makes the image, prompt context, and target code directly addressable by the training pipeline.

---

## 5.2 SFT Export

`main_sft.py` generates the data for Supervised Fine-Tuning as well as the benchmark.

### Deliberately Weight Visual Classes

Not every automatically discovered image class is equally important for training.

For this reason, each class can be assigned a qualitative importance value.

This importance controls the desired composition of the training dataset.

```text
Image Classes
     │
     ▼
Importance Mapping
     │
     ▼
Weighted sample selection
```

This prevents a very large but less relevant class from automatically dominating the training dataset.

If an important class does not contain enough samples, the missing amount is redistributed as sensibly as possible across other suitable classes. `noise` is used only as a fallback and not as a preferred source.

### Validation

The validation set is selected before the actual training split.

It attempts to cover the active visual classes in a targeted way and is then kept completely separate from the training pool.

### Benchmark

The benchmark is constructed independently of the SFT class distribution.

This makes it possible to test different or more broadly distributed visual classes than those emphasized during training.

If a desired benchmark class contains too few real samples, `noise` can be used to fill the missing slots.

### Description Mix

If a sample has multiple descriptions, one of them can be selected during export.

This allows training, for example, to see a mixture of

- code-based descriptions,
- image-based descriptions,
- and multimodal descriptions

without requiring separate master datasets.

### CrystalBLEU Reference Data

A portion of the remaining TikZ code is exported separately as a reference corpus for **CrystalBLEU**.

This corpus is not used as ordinary training data but supports later code-based evaluation.

It is important that the corpus is built from samples that have not already been consumed for the corresponding training or validation purpose.

---

## 5.3 RL Export

`main_rl.py` creates a separate dataset for the Reinforcement Learning stage.

The selection continues to use the visual-class logic from the SFT export, but additionally takes **repetition difficulty** into account in a targeted way.

The basic idea is:

```text
cleaned training pool
        │
        ├──► low repetition
        │
        └──► high / critical repetition
                 │
                 ▼
        controlled mixed RL dataset
```

Why is this useful?

With low-repetition TikZ code, the model learns or stabilizes normal structured generation. Highly repetitive examples, by contrast, deliberately represent more difficult cases in which a generative model is more likely to produce redundant or loop-like outputs.

The RL dataset therefore intentionally combines easier and more problematic repetition ranges instead of using only a random subset of the SFT data.

Within these repetition groups, the visual-class weighting is preserved. This allows the export to control two axes simultaneously:

```text
What is shown in the image?
        ×
How repetitive is the target code?
```

The manifest can then be ordered by repetition difficulty, making a curriculum-like ordering possible as well.

---

# Interaction of the Metadata

The pipeline generates three particularly important properties for each sample:

```text
class
repetition_class
llm_description
```

They answer different questions:

| Metadata | Meaning |
|---|---|
| `class` | What visual structure does the diagram have? |
| `repetition_class` | How strongly does the TikZ code repeat locally? |
| `llm_description` | How can the desired image content be described in natural language? |

Together, these properties enable much more controlled dataset selection than pure random sampling.

For example, the export can specifically

- give greater weight to certain visual diagram types,
- exclude highly repetitive samples from SFT,
- use the same difficult samples later in a targeted way for RL,
- and mix descriptions depending on the training objective.

---

# Overall Idea of the Pipeline

The entire data pipeline follows a simple principle:

> **First establish data quality, then understand the data structure, and only afterward select samples for the actual training process.**

```text
Raw Data
   │
   ▼
Deduplication + Filtering
   │
   ▼
TikZ Rendering + Validation
   │
   ▼
Image Embeddings
   │
   ├──► Visual Clustering
   │
   └──► Render Consistency Check
   │
   ▼
Repetition Analysis
   │
   ▼
Optional LLM Descriptions
   │
   ▼
Clean Master Dataset
   │
   ├──► SFT Dataset
   ├──► RL Dataset
   ├──► Validation
   ├──► Benchmark
   └──► CrystalBLEU Corpus
```

This cleanly separates data cleaning from training-sample selection.

`2_preprocess` mainly decides **whether a sample is technically and semantically usable and which properties it has**.

`5_export` then decides **for which training or evaluation purpose that sample should be used**.

This separation makes the pipeline flexible: changes to the SFT/RL composition do not automatically require rerendering or reclustering the entire dataset.
