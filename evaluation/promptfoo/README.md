# Image-to-TikZ Benchmark

## Setup

First, install TeX Live Full:

```bash
sudo sh scripts/install_texlive_full.sh
```

Then create the Conda environment and install Promptfoo:

```bash
conda env create -f environment.yml
conda activate promptfoo-tikz
npm install
```

After that, only adjust the paths and benchmark settings in `config.py`. By default, the following files and directories are expected:

```text
data/image_manifest.csv
data/images/*
data/references/*
data/benchmark_corpus/*.txt
```

The CSV file must contain at least the columns `reference_image` and `reference_code`. Absolute paths are kept unchanged. Relative paths are resolved relative to the manifest file. Legacy container paths under `/images` and `/references` are automatically mapped to the central data directories.



## Setup docker
docker build -t promptfoo-tikz:latest .

docker run --rm -it \
  --gpus all \
  --network host \
  -v /home/jonas/Datasets:/home/jonas/Datasets:ro \
  promptfoo-tikz:latest \
  python run.py eval

enroot import -o promptfoo-tikz.sqsh dockerd://promptfoo-tikz:latest

## Running the Benchmark

The model server must be running at the URL configured in `config.py`.

```bash
python run.py eval
python run.py view
```

Additional Promptfoo arguments are forwarded:

```bash
python run.py eval --no-cache
python run.py eval --watch
```

## Configuration

- `PATHS`: Data, results, and cache directories
- `RENDER`: TeX Live path, engines, timeout, DPI, and image size
- `PROMPTFOO`: Providers, concurrency, and UI port
- `BENCHMARK.text_replace_metrik`: Replaces TikZ node text with `aaaa`, `aaab`, ... before all metrics are calculated
- `MODELS`: Model names and compute device
- `METRICS`: Central thresholds and metric parameters
- `DEBUG`: Debug images and tracebacks

The default TeX Live installation is:

```python
texlive_root = Path("/usr/local/texlive/2026")
texlive_platform = "x86_64-linux"
```

The resulting path `/usr/local/texlive/2026/bin/x86_64-linux` is prepended to `PATH` before the benchmark starts. Change `texlive_platform` when using a different architecture. `texlive-core` is intentionally not included in the Conda environment so that only the complete TeX Live installation is used.

Secrets must not be stored in `config.py`. When a provider requires an API key, set it only in the active shell or with `conda env config vars set ...`.

## GPU

`requirements.txt` installs the standard PyTorch distribution. If your cluster requires a specific CUDA wheel, reinstall PyTorch after creating the Conda environment according to the official PyTorch installation instructions. All metrics automatically use CUDA when it is available.

## Security

Model-generated TeX code is untrusted. As in the original project, rendering uses a tolerant LaTeX fallback and a timeout, but it does not provide a complete operating-system-level sandbox. Only render unknown TeX code in an appropriately isolated environment.

## Text-Independent Metrics

Enable the mode in `config.py`:

```python
@dataclass(frozen=True)
class Benchmark:
    text_replace_metrik: bool = True
```

When enabled, non-empty TikZ node contents in the reference and model output are replaced independently according to their order with `aaaa`, `aaab`, `aaac`, and so on.

Code metrics use the modified TeX strings. Image metrics re-render both the modified reference code and the modified model output before calculating the score.