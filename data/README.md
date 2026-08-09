# Setup

### Install dependencies

```bash
conda env create -f environment.yaml
conda activate data-tools
```

### Install latex
```bash
sh install_texlive_full.sh
```

### Install ollama and download the qwen coder
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama run qwen3-coder:30b-a3b-q4_K_M
```

### Download DaTikZ-V4

```bash
hf download nllg/DaTikZ-V4 \
  --repo-type dataset \
  --local-dir ./dataset
```

### Download our collected dataset

```bash
hf download loss-boss/tikz-benchmark \
  --repo-type dataset \
  --local-dir ./dataset
```

### Download final dataset
```bash
hf download loss-boss/tikz-dataset-clean \
  --repo-type dataset \
  --local-dir ./dataset
```