# Evaluation Pipeline

This pipeline evaluates **Image-to-TikZ models** on several levels. A model receives an input image and generates TikZ/LaTeX code from it. The evaluation then checks not only whether the code compiles, but also how similar the rendered result is to the reference image and how strongly the generated code differs structurally from the reference code.

The evaluation deliberately considers several perspectives at the same time:

- **functional correctness:** Can the generated TikZ code be rendered at all?
- **visual similarity:** Does the result look like the reference?
- **perceptual similarity:** Are shape, structure, and semantic image content perceived as similar?
- **code similarity:** Is the generated TikZ code structurally similar to the reference?
- **diagnostic quantities:** For example, how long is a model's reasoning/thinking output?

No single metric fully describes the quality of a TikZ result. The strength of the pipeline therefore lies in considering several complementary metrics together.

---

## Structure

```text
evaluation/
├── models/
│   ├── ollama/
│   ├── geotikzbridge-8b/
│   └── detikzify_2_5_8b/
│
├── promptfoo/
│   ├── configs/
│   │   ├── image_to_tikz.yaml
│   │   ├── providers.yaml
│   │   └── prompt*.txt
│   │
│   ├── assertions/
│   │   ├── tikz_is_renderable.py
│   │   ├── image_structural_similarity.py
│   │   ├── image_ms_structural_similarity.py
│   │   ├── image_clip_similarity.py
│   │   ├── image_siglip_similarity.py
│   │   ├── image_lpips.py
│   │   ├── image_dreamsim.py
│   │   ├── image_dists.py
│   │   ├── tikz_crystalbleu.py
│   │   ├── tikz_ted.py
│   │   ├── tikz_relative_length_output_reference.py
│   │   └── tikz_thinking_length.py
│   │
│   ├── pf_utils/
│   │   ├── tikz_rendering.py
│   │   ├── image_utils.py
│   │   ├── clip_siglip_metric.py
│   │   ├── lpips_metric.py
│   │   ├── dreamsim_metric.py
│   │   ├── dists_metric.py
│   │   ├── ssim_metric.py
│   │   ├── ms_ssim_metric.py
│   │   ├── crystalbleu_metric.py
│   │   ├── ted_metric.py
│   │   └── clean_text_in_tex.py
│   │
│   ├── config.py
│   ├── tests.py
│   └── run.py
│
└── analyze/
    └── analyze-v2.ipynb
```

The basic data flow is:

```text
Benchmark dataset
        │
        ▼
    Input image
        │
        ▼
       Model
        │
        ▼
 generated TikZ code
        │
        ├──────────────► Code metrics
        │
        ▼
   LaTeX rendering
        │
        ▼
   generated image
        │
        └──────────────► Image metrics

Reference code ──────────► Code metrics
Reference image ─────────► Image metrics

All scores
        │
        ▼
 Promptfoo results
        │
        ▼
      analyze/
```

---

# 0. `models/` – standardized model interfaces

The `models/` directory encapsulates different Image-to-TikZ models behind HTTP interfaces. This allows the actual evaluation to work independently of how a model is loaded or executed internally.

From Promptfoo's perspective, the process is essentially always the same:

```text
Image + prompt
    │
    ▼
Model API
    │
    ▼
TikZ code
```

The differences between the models therefore remain contained within `models/`.

## `models/ollama/`

This adapter connects the evaluation to multimodal models served through Ollama.

In particular, it handles:

- passing the image to the VLM,
- constructing the text prompt,
- optionally including an additional image description,
- extracting the actual LaTeX/TikZ code from the model response,
- forwarding metadata such as the length of a thinking/reasoning output.

This makes it possible to compare different general-purpose vision-language models using the same evaluation pipeline.

### Thinking Length

If a model returns a separate thinking text, its character length is stored. This quantity is **not a quality metric**, but a diagnostic measure.

For example, it can help investigate whether a model produces more internal text for harder examples or whether longer reasoning correlates with better results.

---

## `models/geotikzbridge-8b/`

This directory encapsulates GeoTikZBridge as its own inference service.

The central idea is the same:

1. The input image is prepared for the vision model.
2. Image and prompt are passed to the model together.
3. The model response is reduced to the actual TikZ code.
4. This code is returned to the evaluation.

The adapter therefore ensures that a specialized TikZ model can be evaluated through the same interface as the general-purpose VLMs.

---

## `models/detikzify_2_5_8b/`

DeTikZify is also integrated through a small API wrapper.

One special feature is that DeTikZify can internally generate multiple candidates. Its internal pipeline evaluates these candidates and returns the best TikZ draft it finds.

For the external evaluation, this internal search is not particularly relevant: Promptfoo ultimately receives exactly one TikZ code output and evaluates it with the same metrics as all other models.

This allows models with different inference strategies to be compared under a common evaluation scheme.

---

# 1. `promptfoo/` – core of the evaluation

`promptfoo/` contains the actual benchmark. Test cases are generated here, models are called, TikZ code is rendered, and the various metrics are computed.

Conceptually, the process for each example is:

```text
Manifest row
    │
    ├── input_image
    ├── reference_image
    ├── reference_code
    └── optional additional information
            │
            ▼
         Provider
            │
            ▼
       generated code
            │
     ┌──────┴──────┐
     ▼             ▼
 Code metrics    Rendering
                   │
                   ▼
             generated image
                   │
                   ▼
               Image metrics
```

---

## `promptfoo/configs/`

The files in `configs/` define the logical structure of the benchmark.

### `image_to_tikz.yaml`

This file connects:

- prompt,
- provider,
- test cases,
- assertions or metrics,
- aggregated average values.

Each model is evaluated on the same test cases. The individual metric values are then aggregated across the benchmark, producing comparable average values for each model.

### `providers.yaml`

This file describes how the different model servers are called and how their responses are converted into a common TikZ output format.

The key idea is **standardization**: different models may expose different APIs, but the evaluation always sees the same output type afterward.

### `prompt*.txt`

These files contain the task description for the model. They define what information the model should transfer from the image into TikZ code.

---

## `promptfoo/tests.py` – benchmark examples

`tests.py` reads the benchmark dataset manifest and converts it into individual Promptfoo test cases.

A test case typically connects:

- the input image for the model,
- the reference image for image metrics,
- the reference code for code metrics,
- optional additional information such as an image description.

This ensures that all models use exactly the same references.

---

# 1.1 Rendering and image normalization

Before an image metric can be computed, the TikZ code generated by the model must first be converted into an image.

This task is handled by `pf_utils/tikz_rendering.py`.

Conceptually, the following happens:

```text
Model output
    │
    ▼
Extract TikZ/LaTeX
    │
    ▼
Compile LaTeX
    │
    ▼
PDF
    │
    ▼
PNG
    │
    ▼
Crop + canvas normalization
    │
    ▼
Image for the metrics
```

Normalization is important because an image metric could otherwise interpret different white margins or different canvas sizes as image differences.

The images are therefore placed on a comparable white background and normalized to a comparable crop and image space.

Rendering is also cached. If the same TikZ code is needed multiple times for different image metrics, it does not have to be compiled again for every metric.

---

# 1.2 Renderability

File:

```text
assertions/tikz_is_renderable.py
```

## Idea

The simplest and at the same time most fundamental metric checks whether the generated code can be successfully converted into an image at all.

Formally, it is binary:

```text
Renderability = 1 if rendering succeeds
Renderability = 0 if rendering fails
```

A visually perfect TikZ draft is useless if the code does not compile. This metric therefore represents the minimum functional requirement.

Additional LaTeX diagnostics are also recorded:

- `latex_errors`: actual LaTeX errors,
- `latex_warnings`: compiler warnings,
- `latex_badboxes`: overfull/underfull box issues.

These values help with error analysis, but they do not provide a complete assessment of visual quality.

---

# 1.3 Pixel- and structure-oriented image metrics

## SSIM – Structural Similarity Index

Files:

```text
assertions/image_structural_similarity.py
pf_utils/ssim_metric.py
```

### What does SSIM measure?

SSIM compares two images based on their **local image structure** instead of simply subtracting individual pixels from each other.

The classic SSIM idea decomposes similarity into three components:

- luminance,
- contrast,
- structure.

In simplified form, the following is computed for local image regions:

```text
SSIM(x, y)
≈
(luminance similarity)
× (contrast similarity)
× (structure similarity)
```

A commonly used form is:

```text
               (2 μx μy + C1)(2 σxy + C2)
SSIM = -----------------------------------------
        (μx² + μy² + C1)(σx² + σy² + C2)
```

Here, `μ` denotes local means, `σ` local contrasts, and `σxy` the shared structure of the images.

### Interpretation

A high SSIM value means that edges, local shapes, contrasts, and spatial structures are similar.

However, SSIM is relatively sensitive to geometric shifts. Two semantically identical diagrams can therefore receive a worse score if elements are slightly displaced.

---

## MS-SSIM – Multi-Scale Structural Similarity

Files:

```text
assertions/image_ms_structural_similarity.py
pf_utils/ms_ssim_metric.py
```

### Idea

MS-SSIM extends SSIM across **multiple resolution levels**.

Instead of comparing only the original images, the images are repeatedly downscaled. Structure and contrast information are compared at each scale.

```text
Original resolution ──► fine details
       ↓
smaller scale ─────────► medium-scale structures
       ↓
even smaller scale ────► global shapes
```

The results from these levels are combined into a single score.

### Why is this useful?

For TikZ figures, exact pixel position is not the only thing that matters. Often, both global structure and local details need to match. MS-SSIM is therefore more robust than plain SSIM to small local deviations.

---

# 1.4 Semantic embedding metrics

## CLIP Similarity

Files:

```text
assertions/image_clip_similarity.py
pf_utils/clip_siglip_metric.py
```

### Idea

CLIP maps images into a high-dimensional **embedding space**. Images with similar visual or semantic content should be close to each other in that space.

Two normalized vectors are produced for the reference and generated image:

```text
reference image ─► CLIP encoder ─► e_ref
generated image ─► CLIP encoder ─► e_gen
```

Cosine similarity is then computed:

```text
                    e_ref · e_gen
cosine similarity = ----------------
                    ||e_ref|| ||e_gen||
```

Because the embeddings are normalized beforehand, this is effectively equivalent to their dot product.

### What does the score mean?

A high value means that CLIP represents both images as semantically similar.

Unlike SSIM, every pixel does not have to be in exactly the same position. CLIP is therefore more focused on **global and semantic similarity**.

On the other hand, it may overlook small geometric errors that can still matter in a technical diagram.

---

## SigLIP Similarity

Files:

```text
assertions/image_siglip_similarity.py
pf_utils/clip_siglip_metric.py
```

### Idea

SigLIP follows a similar principle to CLIP: the image is projected into a semantic embedding space and then compared to the other image using cosine similarity.

The main theoretical difference lies in how the underlying model is trained. While CLIP is typically trained with a contrastive softmax objective, SigLIP uses a **sigmoid-based pairwise learning objective**.

For evaluation, however, the process is the same:

```text
Image A ─► SigLIP ─► normalized embedding A
Image B ─► SigLIP ─► normalized embedding B
                          │
                          ▼
                   Cosine Similarity
```

### Why use CLIP and SigLIP together?

Both metrics capture semantic image similarity, but they rely on different representations. If both scores agree, this is a stronger signal than relying on only one embedding model.

---
# 1.5 Learned perceptual metrics

## LPIPS – Learned Perceptual Image Patch Similarity

Files:

```text
assertions/image_lpips.py
pf_utils/lpips_metric.py
```

### Idea

LPIPS does not compare images directly in pixel space, but instead uses the activations of a pretrained neural network.

In simplified form:

```text
Image A ─► CNN ─► Feature maps from multiple layers
Image B ─► CNN ─► Feature maps from multiple layers
                    │
                    ▼
             Feature distances
                    │
                    ▼
              LPIPS Distance
```

Early network layers respond more strongly to edges and textures, while later layers respond to more complex shapes. LPIPS combines these differences into a perceptual distance.

The raw value is a **distance**:

```text
small distance = similar images
large distance = different images
```

To ensure that all central quality metrics use the same direction, the pipeline converts the distance into a similarity:

```text
similarity = 1 / (1 + distance)
```

This means:

```text
small distance  → Similarity close to 1
large distance  → lower Similarity
```

### Strength

LPIPS often correlates better with human perception than pure pixel metrics. Small pixel deviations are penalized less strongly when the perceived structure remains similar.

---

## DreamSim

Files:

```text
assertions/image_dreamsim.py
pf_utils/dreamsim_metric.py
```

### Idea

DreamSim is a learned perceptual similarity metric that uses representations from modern vision models and is more strongly aligned with **human similarity judgments**.

Two images are processed by the DreamSim model. The model returns a perceptual distance.

The pipeline converts it to:

```text
similarity = 1 - distance
```

The score is then clipped to the valid evaluation range.

### Interpretation

DreamSim is intended to answer more directly:

> Would a human perceive these two images as visually similar?

This makes the metric complementary to both pixel-based measures such as SSIM and semantic embeddings such as CLIP.

---

## DISTS – Deep Image Structure and Texture Similarity

Files:

```text
assertions/image_dists.py
pf_utils/dists_metric.py
```

### Idea

DISTS considers deep image features and distinguishes in particular between two aspects:

- **structure**,
- **texture**.

Instead of comparing pixels directly, statistical properties of neural-network feature maps are compared. This allows DISTS to detect structural and texture-related changes without relying exclusively on pixel-perfect alignment.

The underlying metric returns a distance. In this pipeline, it is also converted into a similarity:

```text
similarity = 1 / (1 + distance)
```

### Relevance for TikZ

TikZ images often contain clear geometric structures, lines, and recurring graphical patterns. DISTS therefore complements LPIPS with a metric that explicitly considers structure and texture together.

---

# 1.6 Code metrics

Image metrics answer the question:

> Does the rendered result look correct?

Code metrics instead answer:

> How similar is the generated TikZ representation to the reference?

These are not the same thing. Two very different TikZ programs can generate the same image. Code scores should therefore never be interpreted in isolation as visual quality.

---

## CrystalBLEU

Files:

```text
assertions/tikz_crystalbleu.py
pf_utils/crystalbleu_metric.py
```

### Starting point: BLEU

BLEU originally comes from machine translation. It counts how many short token sequences—so-called **n-grams**—from a reference also occur in the generated text.

For example, the following:

```text
\draw (A) -- (B);
```

produces token sequences of different lengths. High overlap indicates similar local code structures.

BLEU combines the precision of several n-gram orders and also uses a length penalty so that extremely short outputs do not receive artificially high scores.

### Why CrystalBLEU?

Program code contains many trivial and very frequent patterns. In TikZ, these can be commonly used commands or syntax fragments.

If these common patterns were counted normally, two programs could receive a high BLEU score even though their actual logic is different.

CrystalBLEU therefore identifies the most frequent n-grams in a larger code corpus and **ignores them during comparison**.

```text
Code corpus
    │
    ▼
most frequent n-grams
    │
    ▼
ignored during BLEU comparison
```

The score is therefore more focused on characteristic code structures.

### Interpretation

A high CrystalBLEU value means that the generated TikZ code shares many non-trivial local token patterns with the reference.

However, it does not prove that the two programs are semantically identical.

---

## TED – Token Edit Distance

Files:

```text
assertions/tikz_ted.py
pf_utils/ted_metric.py
```

### Important note

In this implementation, `TED` is **not a Tree Edit Distance over a syntax tree**.

Instead, the code tokenizes both TeX programs and computes a **Levenshtein Edit Distance at token level**.

### Theory

Levenshtein distance asks:

> What is the minimum number of elementary changes required to transform sequence A into sequence B?

The allowed operations are:

- insert a token,
- delete a token,
- replace a token.

Example:

```text
Reference:  A B C D
Generated:  A B X D
```

Here, one replacement is required.

The pipeline normalizes the edit distance by the length of the reference:

```text
normalized_distance = edits / reference_length
```

It is then converted into a similarity:

```text
similarity = 1 / (1 + normalized_distance)
```

### Interpretation

A high value means that only a small number of token operations are required to transform the generated code into the reference code.

However, the metric does not account for semantic equivalence: two syntactically different TikZ solutions can produce the same image and still have a large edit distance.

---

## Relative Length Similarity

File:

```text
assertions/tikz_relative_length_output_reference.py
```

### Idea

This metric compares only the character length of the generated code with the length of the reference code.

First, the relative length error is computed:

```text
                 |L_output - L_reference|
relative_error = ------------------------
                       L_reference
```

This is then converted into a similarity:

```text
similarity = 1 / (1 + relative_error)
```

If both code outputs have the same length, the similarity is maximal. As the relative length difference increases, the similarity decreases.

### Purpose

The metric is primarily diagnostic. It can reveal whether a model systematically generates extremely short or extremely long solutions.

It does **not** indicate whether the code is correct. An incorrect program can have exactly the same length as the reference.

---

# 1.7 Text-independent evaluation

TikZ figures can contain text labels. This can strongly affect image metrics.

For example, a model may reconstruct the geometry correctly but spell a node label incorrectly. A pixel-based metric would then evaluate both the text error and the geometry together.

The pipeline therefore provides an optional mode in which the contents of TikZ nodes are replaced with neutral placeholders.

Conceptually, for example:

```text
node {...}  → node {aaaa}
node {...}  → node {aaab}
node {...}  → node {aaac}
```

The replacement is performed separately, but in the same order, for the reference and model output.

For code metrics, the modified TeX strings are then compared.

For image metrics, both the reference code and the generated code are re-rendered after text replacement.

This makes it possible to investigate more directly:

> Did the model reconstruct the graphical and geometric structure correctly, independent of the specific text content?

---
# 1.8 How the metrics work together

The metrics are intentionally redundant and complementary.

| Group | Metric | Main question |
|---|---|---|
| Function | Renderability | Is the code executable? |
| local image structure | SSIM | Do local structure, luminance, and contrast match? |
| multi-scale structure | MS-SSIM | Do details and global structure match across multiple scales? |
| semantic image embedding | CLIP | Are the images globally/semantically similar? |
| semantic image embedding | SigLIP | Are the images similar in an alternative semantic embedding space? |
| learned perception | LPIPS | Are deep features perceptually similar? |
| learned perception | DreamSim | Does the similarity better match human perception? |
| structure + texture | DISTS | Do deep structure and texture features match? |
| code n-grams | CrystalBLEU | Do both programs share characteristic local code patterns? |
| code edit distance | TED | How many token changes separate the programs? |
| code length | Relative Length | Is the code of a similar length? |
| diagnostics | Thinking Length | How long is the separate reasoning output? |

A typical example shows why multiple metrics are necessary:

```text
Case A:
SSIM low
CLIP/SigLIP high
```

This may mean that the diagram is semantically correct, but individual elements are slightly shifted.

```text
Case B:
CrystalBLEU low
Image metrics high
```

This may mean that the model chose a different TikZ implementation that is nevertheless visually very similar.

```text
Case C:
Code metrics high
Image metrics low
```

This may indicate small but visually decisive changes in coordinates, styles, or parameters.

The evaluation should therefore primarily be understood as a **multidimensional quality profile**, not as a search for a single perfect metric.

---

# 1.9 Score normalization and aggregation

The regular quality metrics are limited by the shared assertion logic to a range from `0` to `1`.

Conceptually, this means for the quality metrics:

```text
1 → high similarity / good result
0 → low similarity / poor result
```

Distance metrics such as LPIPS, DISTS, and Token Edit Distance are therefore converted into similarities before being returned.

Each metric also has a threshold that Promptfoo can use to mark a test as passed or failed. However, the actual analysis should not rely only on the binary pass/fail value, but should compare the continuous scores.

Average values are computed per model across all test cases. This gives each model a profile such as:

```text
Model
├── Renderability
├── SSIM
├── MS-SSIM
├── CLIP
├── SigLIP
├── LPIPS
├── DreamSim
├── DISTS
├── CrystalBLEU
├── TED Similarity
├── Relative Length Similarity
└── Thinking Length
```

`thinking_length` is an exception: it is a raw character count and not a quality metric normalized to `[0, 1]`.

---

# 1.10 `assertions/` and `pf_utils/`

The metrics are deliberately separated into two layers.

## `assertions/`

The assertion files form the interface to Promptfoo.

They handle tasks such as:

- loading reference data,
- calling the appropriate metric code,
- converting a distance into a similarity,
- making the pass/fail decision,
- returning additional diagnostic values.

## `pf_utils/`

This is where the actual technical implementation of the metrics and helper functions lives.

This separation keeps the Promptfoo integration thin and makes it possible to use or test the metric implementations independently as well.

---

# 2. `analyze/` – comparison and interpretation of results

After the benchmark has run, the results are available in Promptfoo databases and the corresponding result directories.

`analyze/analyze-v2.ipynb` loads these data and performs the cross-model analysis.

The analysis essentially consists of three ideas.

## 2.1 Average values per model

For each metric, a model's scores are averaged across all examples.

This makes it easy to quickly investigate which model, for example, achieves the best average visual similarity or the highest renderability.

However, an average should never be considered in isolation because it can hide outliers and differences in example difficulty.

---

## 2.2 Score distributions

The complete distributions of the metrics are also examined.

Box plots and individual points can show, for example:

- median,
- spread,
- outliers,
- very easy and very difficult examples,
- a model's stability across the dataset.

Two models can have the same average even though one behaves very consistently while the other alternates between perfect and very poor results.

---

## 2.3 Combined score for case analysis

The notebook can combine several selected metrics into a shared analysis score.

Because different metrics may have different value ranges or distributions, they are first normalized within the considered results using min-max scaling:

```text
             x - min(x)
normalized = ----------
             max(x)-min(x)
```

For quantities where a smaller value is better, the direction is then inverted.

The normalized metrics are subsequently averaged:

```text
combined_score = mean(normalized metric scores)
```

This combined score is **not a new universal quality metric**. It is mainly used to sort examples for qualitative analysis.

This allows particularly good and particularly poor cases for a model to be selected and then visually inspected.

---

# Overall idea

The evaluation does not attempt to reduce TikZ quality to a single number.

Instead, a generated result is examined successively from several perspectives:

```text
1. Can the code be rendered?
           │
           ▼
2. Do pixels and local image structure match?
           │
           ▼
3. Do global and perceptual image features match?
           │
           ▼
4. Is the generated code similar to the reference?
           │
           ▼
5. How stable are these values across the entire benchmark?
```

This makes it possible to distinguish between different types of errors:

- syntactically invalid TikZ code,
- geometrically incorrect diagrams,
- small local rendering deviations,
- semantically incorrect reconstructions,
- visually correct but differently implemented TikZ solutions,
- unusually short or long code generations.

For Image-to-TikZ in particular, this multidimensional view is important because **code similarity and image similarity are not the same thing**. The actual goal is a renderable program whose output reproduces the visual structure of the reference as closely as possible—regardless of whether exactly the same TikZ code was used.
