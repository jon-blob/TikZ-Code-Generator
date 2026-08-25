# TikZ Training Pipeline

This pipeline trains vision-language models to generate a complete, compilable LaTeX/TikZ document from an input image.

There are two model paths:

1. `custom_vlm_sft/` – custom VLM based on SigLIP2 + Qwen2.5-Coder.
2. `gemma4_finetuning/` – Gemma vision model with **Supervised Fine-Tuning (SFT)** followed by **GRPO Reinforcement Learning**.

The Gemma path is the actual main pipeline and is therefore described in more detail.

---

## 1. Overall Concept

```text
Training data
    │
    ├── input_image
    ├── reference_code
    ├── reference_image
    └── llm_description
    │
    ▼
Gemma SFT
Image + Prompt -> Ground-Truth TikZ
    │
    ▼
SFT Checkpoint
    │
    ▼
GRPO Reinforcement Learning
    │
    ├── generate multiple TikZ completions
    ├── compile/render code
    ├── evaluate LaTeX quality
    ├── compare code with reference
    └── compare render with target image
    │
    ▼
Reward
    │
    ▼
GRPO Policy Update
```

The fundamental difference:

- **SFT** directly learns the known reference code using cross-entropy loss.
- **GRPO** generates multiple solutions of its own and learns which ones are rated better within a group.

---

# 2. Directory Structure

```text
training/
├── README.md
│
├── custom_vlm_sft/
│   ├── config.py
│   ├── train.py
│   └── src/
│       ├── data.py
│       ├── model.py
│       ├── render.py
│       └── validation.py
│
└── gemma4_finetuning/
    ├── training_config.py
    │
    ├── common/
    │   └── data_utils.py
    │
    ├── prompts/
    │   └── instruction.txt
    │
    ├── gemma4-sft/
    │   ├── data.py
    │   ├── model_loader.py
    │   ├── trainer.py
    │   ├── train.py
    │   ├── tb_callback.py
    │   └── debug_tools.py
    │
    ├── gemma4-rl/
    │   ├── data.py
    │   ├── model_loader.py
    │   ├── trainer.py
    │   ├── train.py
    │   ├── rewards.py
    │   └── reward_functions/
    │       ├── render_reward.py
    │       ├── diagnostic_reward.py
    │       ├── code_reward.py
    │       └── visual_reward.py
    │
    ├── scripts/
    │   ├── install_latex.sh
    │   └── export_gguf.py
    │
    └── runs/
        └── TensorBoard Logs
```

### Most Important Files

| File | Purpose |
|---|---|
| `training_config.py` | Central SFT and GRPO configuration |
| `common/data_utils.py` | Loading images, code, prompt, and manifests |
| `gemma4-sft/data.py` | Creates multimodal SFT examples |
| `gemma4-sft/model_loader.py` | Loads Gemma and configures LoRA |
| `gemma4-sft/trainer.py` | Creates the SFT trainer |
| `gemma4-rl/data.py` | Creates GRPO prompts and keeps reward references available |
| `gemma4-rl/trainer.py` | Creates the GRPO trainer |
| `gemma4-rl/rewards.py` | Combines all reward components |
| `render_reward.py` | Checks compilability and renders TikZ |
| `diagnostic_reward.py` | Evaluates LaTeX errors, warnings, and badboxes |
| `code_reward.py` | CrystalBLEU + Tree-Edit Similarity |
| `visual_reward.py` | SigLIP + LPIPS + DreamSim |

---

# 3. Data

The central manifest columns are:

```text
input_image
reference_image
reference_code
llm_description
```

| Field | Meaning |
|---|---|
| `input_image` | Image that the model sees as input |
| `reference_code` | Ground-truth TikZ for SFT and code reward |
| `reference_image` | Target image for the visual RL reward |
| `llm_description` | Optional additional image description/hints |

The shared prompt is located at:

```text
gemma4_finetuning/prompts/instruction.txt
```

It instructs the model to return only a complete, compilable LaTeX document.

If `USE_LLM_DESCRIPTION=True`, the additional description is appended to the prompt.

---

# 4. Custom VLM – Brief Overview

The alternative custom VLM path combines:

- SigLIP2 as the vision encoder,
- a small trainable vision projector,
- Qwen2.5-Coder as the language model,
- LoRA on the language model.

The visual features are projected to the embedding dimension of the LLM and placed before the text tokens.

The training roughly proceeds in three stages:

```text
1. Text SFT of the Qwen model
2. Training the vision projector
3. Joint multimodal SFT
```

The loss is calculated only on the target code, not on image or prompt tokens.

---

# 5. Gemma Training

The Gemma path consists of two phases:

```text
SFT -> SFT Checkpoint -> GRPO
```

SFT first ensures that the model can reliably generate complete and syntactically plausible TikZ documents. GRPO then specifically optimizes for renderability, code quality, and visual similarity.

---

# 6. Gemma SFT

Start:

```bash
python gemma4-sft/train.py
```

## 6.1 Model

An instruction-tuned Gemma model is used, loaded with Unsloth and then fine-tuned via PEFT/LoRA.

The specific model and LoRA hyperparameters are defined in the training configuration. For understanding the pipeline, the main point is that only a small portion of the model parameters is adapted through LoRA, while the base model remains largely unchanged. Both language-related and vision-related model components can be included in the fine-tuning.

## 6.2 Structure of an SFT Example

An example consists of:

```text
USER:
    [input_image]
    [instruction + optional llm_description]

ASSISTANT:
    [reference_code]
```

The dataset is constructed in `gemma4-sft/data.py`.

The reference code is first reduced to the actual document section:

```latex
\documentclass
...
\end{document}
```

## 6.3 Completion-only Loss

Important:

```python
completion_only_loss = True
```

The cross-entropy loss is therefore calculated only for the assistant completion:

```text
Image + Prompt      -> no loss tokens
Ground-Truth Code   -> loss tokens
```

The model therefore directly learns:

```text
Image + Instruction -> TikZ Code
```

## 6.4 Training and Configuration

The usual training parameters such as learning rate, batch size, gradient accumulation, number of epochs, warmup, optimizer, and scheduler are controlled centrally through the training configuration.

During training, checkpoints, evaluation loss, and free example generations are logged regularly.

Free generation is important because a good teacher-forced loss does not automatically mean that the model can independently generate correctly terminating TikZ documents.

---

# 7. Transition from SFT to GRPO

The RL stage starts from an already trained SFT checkpoint. This means RL begins with a model that has already learned the desired output structure and basic TikZ syntax.

This is important because a model without prior SFT would often produce non-renderable LaTeX code, causing almost all RL samples to receive the same poor reward.

---

# 8. Gemma GRPO

Start:

```bash
python gemma4-rl/train.py
```

For each example, the RL dataset provides:

```python
{
    "prompt": prompt,
    "image": input_image,
    "reference_image": reference_image,
    "answer": reference_code,
}
```

The model sees only `input_image` and the prompt. `reference_image` and `reference_code` are used exclusively for reward calculation.

## 8.1 GRPO Principle

Multiple candidates are generated for each prompt:

```text
Prompt
  │
  ├── Completion A -> Reward rA
  ├── Completion B -> Reward rB
  ├── Completion C -> Reward rC
  └── ...
                │
                ▼
         relative comparison
                │
                ▼
           GRPO Update
```

The candidates are generated using sampling so that different solutions arise within a group. The number of generated candidates and the degree of sampling variation are controlled through the RL configuration. The optimization and GRPO-specific settings are also defined there.

---

# 9. RL Reward

The reward logic is located in:

```text
gemma4-rl/rewards.py
```

and consists of four parts:

```text
1. Renderability
2. LaTeX diagnostics
3. Code similarity
4. Visual similarity
```

The calculation is hierarchical:

```text
Generated TikZ
    │
    ▼
Render LaTeX
    │
    ├── error -> Reward = -2
    │
    └── successful
            │
            ├── Diagnostic Reward
            ├── Code Reward
            └── Visual Reward
                    │
                    ▼
               Total Reward
```

## 9.1 Non-renderable Code

If the document does not compile or no valid rendering is produced:

```text
R = -2.0
```

The remaining metrics are then no longer calculated.

This makes renderability the most important initial requirement.

---

## 9.2 Diagnostic Reward

For renderable documents, LaTeX errors, warnings, and badboxes are taken into account.

With

```text
E = Errors
W = Warnings
B = Badboxes
```

the following is calculated:

```text
R_diag = clip(
    1 - 0.10E - 0.05W - 0.01B,
    -2,
    1
)
```

A clean document therefore receives:

```text
R_diag = 1
```

The more LaTeX issues occur, the smaller this component becomes.

---

## 9.3 Code Reward

The code is compared with the ground-truth TikZ.

The following are used:

- **CrystalBLEU** for token-based code similarity,
- **Tree Edit Distance (TED)** for structural similarity.

Let:

```text
C = CrystalBLEU       in [0,1]
T = TED Similarity    in [0,1]
```

Internally, the code score is:

```text
R_code = (C + 0.5T) / 1.5
```

This is multiplied by `3.0` in the total reward.

This simplifies the effective code contribution to:

```text
R_code,total = 2C + T
```

Maximum contribution:

```text
3.0
```

CrystalBLEU is therefore weighted twice as strongly as TED similarity.

---

## 9.4 Visual Reward

The rendered model image is compared with `reference_image`.

The following are used:

- SigLIP Similarity,
- LPIPS Similarity,
- DreamSim Similarity.

With

```text
S = SigLIP
L = LPIPS Similarity
D = DreamSim
```

the internal visual score is:

```text
R_visual = 0.15S + 0.50L + 0.35D
```

In the total reward, this is additionally multiplied by `0.5`:

```text
R_visual,total
= 0.075S + 0.25L + 0.175D
```

Maximum visual contribution:

```text
0.5
```

This means the visual reward is currently weighted significantly less than the code reward.

---

# 10. Total Reward Formula

For **non-renderable** candidates:

```text
R_total = -2.0
```

For **renderable** candidates:

```text
R_total
= 1
+ clip(1 - 0.10E - 0.05W - 0.01B, -2, 1)
+ 2C
+ T
+ 0.075S
+ 0.25L
+ 0.175D
```

with:

```text
E = LaTeX Errors
W = LaTeX Warnings
B = Badboxes
C = CrystalBLEU
T = TED Similarity
S = SigLIP Similarity
L = LPIPS Similarity
D = DreamSim Similarity
```

The first constant term

```text
+1
```

is the reward for a document being renderable at all.

### Reward Range

```text
non-renderable:  -2.0
renderable:      -1.0 to 5.5
```

Perfect theoretical candidate:

```text
Renderable       = 1.0
Diagnostic       = 1.0
Code             = 3.0
Visual           = 0.5
----------------------
Total            = 5.5
```

A central property of the design is therefore:

> Every renderable candidate can theoretically score better than a non-renderable candidate.

The reward priorities are approximately:

```text
1. generate valid, renderable LaTeX/TikZ code
2. match the reference code structurally and at the token level
3. generate LaTeX that is as clean as possible
4. make the rendered image visually resemble the target image
```

---

# 11. From Reward to GRPO Update

Four rewards are produced for each prompt:

```text
r1, r2, r3, r4
```

GRPO considers not only their absolute values but also compares the candidates within the group.

A candidate with a higher reward receives a more positive learning signal than a worse candidate from the same group.

As a result, the model gradually learns to generate outputs more frequently that:

- compile,
- are cleaner,
- are more similar to the reference code,
- and visually match the target better.

If all four candidates receive the same reward, however, there is hardly any relative learning signal. Therefore, both sufficient sampling variation and a differentiating reward are important.

---

# 12. Logging and Outputs

SFT and GRPO use TensorBoard.

Particularly important for RL are:

```text
reward/render_ok_rate
reward/avg_total_score
reward/avg_diagnostic_score
reward/avg_code_score
reward/avg_visual_score
reward/avg_crystalbleu
reward/avg_ted
reward/avg_siglip
reward/avg_lpips
reward/avg_dreamsim
```

This makes it possible to see **why** the reward changes.

For example, the total reward can increase even though image quality barely improves, because the code component, with a maximum of `3.0`, is weighted much more strongly than the visual component, with a maximum of `0.5`.

Outputs are currently located under:

```text
/home/jonas/models/sft/checkpoints/
/home/jonas/models/sft/lora/

/home/jonas/models/grpo/checkpoints/
/home/jonas/models/grpo/lora/
```

---

# 13. Starting Training

Environment:

```bash
cd training/gemma4_finetuning
conda env create -f environment.yaml
conda activate gemma4-finetuning
```

Before training, the paths in `training_config.py` in particular must be correct:

```text
DATA_DIR_SFT
DATA_DIR_RL
MODELS_DIR
```

SFT:

```bash
python gemma4-sft/train.py
```

Then GRPO:

```bash
python gemma4-rl/train.py
```

---

# 14. Important Notes

Some details in the repository should be considered during further development:

- Older documentation or exports may differ from the model configuration currently in use.
- `ted_scale` exists in the config but is not used in the current reward implementation.
- The LaTeX installation script and `configure_runtime()` currently do not expect exactly the same installation path.
- The visual reward compares the model render with `reference_image`, not directly with `input_image`.

---

# 15. Pipeline in Brief

```text
SFT
===
input_image
+ instruction
+ optional llm_description
        │
        ▼
Gemma + LoRA
        │
        ▼
reference_code
        │
        ▼
Completion-only Cross-Entropy Loss


GRPO
====
input_image + prompt
        │
        ▼
multiple TikZ completions
        │
        ├── render
        ├── diagnose LaTeX
        ├── compare code
        └── compare images
        │
        ▼
Reward per candidate
        │
        ▼
relative group comparison
        │
        ▼
GRPO Policy Update
```

The core of the pipeline is therefore:

> **SFT teaches the model to generate valid TikZ code. GRPO then optimizes the self-generated solutions based on renderability, code quality, and visual agreement.**
