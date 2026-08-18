from dataclasses import dataclass
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Paths:
    data: Path = Path("/home/jonas/Datasets/TikZ/our_dataset_benchmark_merged/simple_llm_description")
    manifest: Path = data / "manifest.csv"
    images: Path = data / "images"
    input_images: Path = data / "images"
    references: Path = data / "references"
    crystalbleu_corpus: Path = Path("/home/jonas/Datasets/TikZ/our_dataset_benchmark_merged/crystalbleu_corpus")
    results: Path = ROOT / "result"
    promptfoo_db: Path = results / "promptfoo-db"
    generated_images: Path = results / "generated_images"
    cache: Path = ROOT / ".cache"
    render_cache: Path = cache / "renders"
    model_cache: Path = cache / "models"
    metric_cache: Path = cache / "metrics"


@dataclass(frozen=True)
class Render:
    texlive_root: Path = Path("/usr/local/texlive/2026")
    texlive_platform: str = "x86_64-linux"
    engines: tuple[str, ...] = ("pdflatex", "lualatex", "xelatex")
    timeout_seconds: int = 60
    dpi: int = 400
    image_size: int = 512
    crop_pdf: bool = True
    crop_png: bool = True
    normalize_canvas: bool = True
    upscale_canvas: bool = True
    halt_on_error: bool = True
    tolerant_fallback: bool = True
    disable_page_numbers: bool = True
    latex_runs: int = 2

    @property
    def texlive_bin(self) -> Path:
        return self.texlive_root / "bin" / self.texlive_platform


@dataclass(frozen=True)
class ProviderRun:
    label: str
    result_name: str
    ollama_model: str | None = None


@dataclass(frozen=True)
class Promptfoo:
    config_file: Path = ROOT / "configs" / "image_to_tikz.yaml"
    max_concurrency: int = 1
    assertions_max_concurrency: int = 4
    view_port: int = 15500
    sequential_providers: bool = True
    unload_ollama: bool = True
    ollama_url: str = "http://127.0.0.1:11434"

    provider_order: tuple[ProviderRun, ...] = (
        ProviderRun(
            label="qwen3-vl:8b-instruct-bf16",
            result_name="qwen3-vl-8b-instruct-bf16",
            ollama_model="qwen3-vl:8b-instruct-bf16",
        ),
        ProviderRun(
            label="gemma4:31b-it-q4_K_M",
            result_name="gemma4-31b-it-q4_K_M",
            ollama_model="gemma4:31b-it-q4_K_M",
        ),
        ProviderRun(
            label="gemma4:12b-it-bf16",
            result_name="gemma4-12b-it-bf16",
            ollama_model="gemma4:12b-it-bf16",
        ),
        ProviderRun(
            label="qwen3.5:9b-bf16",
            result_name="qwen3.5-9b-bf16",
            ollama_model="qwen3.5:9b-bf16",
        ),
        ProviderRun(
            label="qwen3.6:35b-a3b",
            result_name="qwen3.6-35b-a3b",
            ollama_model="qwen3.6:35b-a3b",
        ),
    )


@dataclass(frozen=True)
class Models:
    device: str = "auto"
    local_files_only: bool = False
    clip: str = "openai/clip-vit-base-patch32"
    siglip: str = "google/siglip-base-patch16-224"
    dreamsim: str = "ensemble"


@dataclass(frozen=True)
class Benchmark:
    text_replace_metrik: bool = False


@dataclass(frozen=True)
class Debug:
    save_images: bool = True
    tracebacks: bool = True


PATHS = Paths()
RENDER = Render()
PROMPTFOO = Promptfoo()
MODELS = Models()
BENCHMARK = Benchmark()
DEBUG = Debug()

# Central metric settings. Assertion-level `config` values may override these.
METRICS = {
    "renderable": {"threshold": 1.0},
    "ssim": {"threshold": 0.75},
    "ms_ssim": {"threshold": 0.75},
    "clip": {"threshold": 0.60},
    "siglip": {"threshold": 0.60},
    "lpips": {"threshold": 0.80, "net_type": "alex"},
    "dreamsim": {"threshold": 0.75},
    "dists": {"threshold": 0.80},
    "crystalbleu": {"threshold": 0.10, "k": 500, "n": 4, "smoothing": True, "use_cache": True},
    "ted": {"threshold": 0.50},
    "relative_length": {"threshold": 0.80},
}


def ensure_directories() -> None:
    for path in (
        PATHS.results,
        PATHS.promptfoo_db,
        PATHS.generated_images,
        PATHS.render_cache,
        PATHS.model_cache,
        PATHS.metric_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)


def apply_runtime_environment() -> dict[str, str]:
    """Set non-secret runtime values for Promptfoo and model libraries."""
    ensure_directories()

    pythonpath = os.environ.get("PYTHONPATH", "")
    path = os.environ.get("PATH", "")
    texlive_bin = str(RENDER.texlive_bin) if RENDER.texlive_bin.is_dir() else ""

    values = {
        "PATH": os.pathsep.join(filter(None, (texlive_bin, path))),
        "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), pythonpath))),
        "PROMPTFOO_PYTHON": sys.executable,
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
        "PROMPTFOO_FAILED_TEST_EXIT_CODE": "0",
        "PROMPTFOO_CONFIG_DIR": str(PATHS.promptfoo_db),
        "PROMPTFOO_ASSERTIONS_MAX_CONCURRENCY": str(
            PROMPTFOO.assertions_max_concurrency
        ),
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HOME": str(PATHS.model_cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(PATHS.model_cache / "huggingface" / "hub"),
        "HF_HUB_CACHE": str(PATHS.model_cache / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(PATHS.model_cache / "huggingface" / "transformers"),
        "HF_ASSETS_CACHE": str(PATHS.model_cache / "huggingface" / "assets"),
        "TORCH_HOME": str(PATHS.model_cache / "torch"),
        "DREAMSIM_CACHE_DIR": str(PATHS.model_cache / "dreamsim"),
    }
    os.environ.update(values)
    return values