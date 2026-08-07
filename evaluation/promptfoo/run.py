import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

from config import PATHS, PROMPTFOO, ROOT, ProviderRun, apply_runtime_environment, ensure_directories

def rename_result_directory(provider: ProviderRun) -> None:
    source = PATHS.results
    target = source.parent / provider.result_name

    if not source.exists():
        print(
            f"Warning: result directory does not exist: {source}",
            file=sys.stderr,
            flush=True,
        )
        return

    if target.exists():
        raise SystemExit(
            f"Result directory already exists: {target}"
        )

    source.rename(target)

    print(
        f"Result directory renamed:\n"
        f"  {source}\n"
        f"  -> {target}",
        flush=True,
    )

def promptfoo_command() -> list[str]:
    name = "promptfoo.cmd" if sys.platform == "win32" else "promptfoo"
    local = ROOT / "node_modules" / ".bin" / name
    if local.exists():
        return [str(local)]

    executable = shutil.which("promptfoo")
    if executable:
        return [executable]

    npx = shutil.which("npx")
    if npx:
        return [npx, "--no-install", "promptfoo"]

    raise SystemExit("Promptfoo is missing. Run `npm install` first.")


def validate_inputs() -> None:
    required = (
        PROMPTFOO.config_file,
        PATHS.manifest,
        PATHS.crystalbleu_corpus,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        paths = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing configured inputs:\n{paths}")


def exact_provider_pattern(label: str) -> str:
    return f"^{re.escape(label).replace(r'\-', '-')}$"


def has_provider_filter(arguments: list[str]) -> bool:
    names = ("--filter-providers", "--filter-targets")
    return any(
        argument in names
        or argument.startswith("--filter-providers=")
        or argument.startswith("--filter-targets=")
        for argument in arguments
    )


def unload_ollama(provider: ProviderRun) -> None:
    if not PROMPTFOO.unload_ollama or not provider.ollama_model:
        return

    url = f"{PROMPTFOO.ollama_url.rstrip('/')}/api/generate"
    body = json.dumps(
        {
            "model": provider.ollama_model,
            "keep_alive": 0,
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60):
            pass
    except (OSError, URLError) as error:
        print(
            f"Warning: could not unload {provider.ollama_model}: {error}",
            file=sys.stderr,
            flush=True,
        )


def eval_command(promptfoo: list[str], extra: list[str]) -> list[str]:
    return [
        *promptfoo,
        "eval",
        "-c",
        str(PROMPTFOO.config_file),
        "-j",
        str(PROMPTFOO.max_concurrency),
        *extra,
    ]


def run_eval(promptfoo: list[str], extra: list[str]) -> int:
    validate_inputs()

    if not PROMPTFOO.sequential_providers or has_provider_filter(extra):
        return subprocess.run(
            eval_command(promptfoo, extra),
            cwd=ROOT,
        ).returncode

    for provider in PROMPTFOO.provider_order:
        ensure_directories()

        print(
            f"\n===== Provider: {provider.label} =====\n",
            flush=True,
        )

        try:
            result = subprocess.run(
                eval_command(
                    promptfoo,
                    [
                        "--filter-providers",
                        exact_provider_pattern(provider.label),
                        "--tag",
                        f"provider={provider.label}",
                        *extra,
                    ],
                ),
                cwd=ROOT,
            )
        finally:
            unload_ollama(provider)

        rename_result_directory(provider)

        if result.returncode != 0:
            return result.returncode

    return 0


def run(command: str, extra: list[str]) -> int:
    apply_runtime_environment()
    promptfoo = promptfoo_command()

    if command == "eval":
        return run_eval(promptfoo, extra)

    cmd = [
        *promptfoo,
        "view",
        str(PATHS.promptfoo_db),
        "--port",
        str(PROMPTFOO.view_port),
        "--no",
        *extra,
    ]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("eval", "view"))
    args, extra = parser.parse_known_args()
    return run(args.command, extra)


if __name__ == "__main__":
    raise SystemExit(main())