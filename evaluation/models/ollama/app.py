import asyncio
import base64
import os
import re
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat",
)
OLLAMA_CONCURRENCY = int(os.getenv("OLLAMA_CONCURRENCY", "1"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=900)
    app.state.semaphore = asyncio.Semaphore(OLLAMA_CONCURRENCY)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def clean_tex(code: str) -> str:
    if not code:
        return ""

    code = code.strip().lstrip("\ufeff")

    match = re.search(
        r"```[a-zA-Z0-9_-]*[ \t]*\r?\n?(.*?)```",
        code,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        code = match.group(1).strip()

    start = code.find(r"\documentclass")
    end = code.rfind(r"\end{document}")
    if start >= 0 and end >= start:
        return code[start:end + len(r"\end{document}")].strip()

    start = code.find(r"\begin{tikzpicture}")
    end = code.rfind(r"\end{tikzpicture}")
    if start >= 0 and end >= start:
        return code[start:end + len(r"\end{tikzpicture}")].strip()

    return code.strip()


@app.post("/generate")
async def generate(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    llm_description: UploadFile = File(...),
    use_llm_description: bool = Form(True),
    think: bool = Form(False),
    num_predict: int = Form(8192, ge=1),
    num_ctx: int = Form(16384, ge=1),
    thinking_token_multiplier: int = Form(2, ge=1),
    debug: bool = Form(False),
    model: str = Form(...),
    seed: int = Form(42),
) -> dict:
    image_data = await image.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Image is empty")

    if debug:
        with open("image.png", "wb") as handle:
            handle.write(image_data)

    try:
        description = (await llm_description.read()).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="LLM description must be UTF-8",
        ) from error

    final_prompt = prompt.strip()
    if use_llm_description and description:
        final_prompt += (
            "\n\nAdditionally, here is a description of the image "
            "with some creation hints:\n"
            f"{description}"
        )

    multiplier = thinking_token_multiplier if think else 1
    effective_num_predict = num_predict * multiplier
    effective_num_ctx = num_ctx * multiplier

    """payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": final_prompt,
                "images": [base64.b64encode(image_data).decode("ascii")],
            }
        ],
        "stream": False,
        "think": think,
        "options": {
            "temperature": 0.0,
            "seed": seed,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "num_predict": effective_num_predict,
            "num_ctx": effective_num_ctx,
        },
    }"""

    payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt,
                    "images": [base64.b64encode(image_data).decode("ascii")],
                }
            ],
            "stream": False,
            "think": think,
            "options": {
                "temperature": 0.2,
                "top_k": 20,
                "top_p": 0.9,
                "min_p": 0.0,
                "num_predict": effective_num_predict,
                "num_ctx": effective_num_ctx,
            },
        }

    if debug:
        print(
            "\n===== REQUEST =====\n"
            f"model: {model}\n"
            f"think: {think}\n"
            f"num_predict: {effective_num_predict}\n"
            f"num_ctx: {effective_num_ctx}\n"
            f"\n{final_prompt}\n"
            "===================\n",
            flush=True,
        )

    try:
        async with request.app.state.semaphore:
            response = await request.app.state.client.post(
                OLLAMA_URL,
                json=payload,
            )
        response.raise_for_status()
        response_data = response.json()
    except httpx.TimeoutException as error:
        raise HTTPException(status_code=504, detail="Ollama timed out") from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama returned HTTP {error.response.status_code}: "
                f"{error.response.text[:2000]}"
            ),
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned invalid JSON",
        ) from error

    message = response_data.get("message") or {}
    raw_output = str(message.get("content") or "").strip()
    thinking_output = str(message.get("thinking") or "").strip()
    thinking_length = len(thinking_output)

    if debug:
        print(
            "\n===== OLLAMA RESPONSE =====\n"
            f"done_reason: {response_data.get('done_reason')}\n"
            f"eval_count: {response_data.get('eval_count')}\n"
            f"thinking_length: {thinking_length}\n"
            f"content_length: {len(raw_output)}\n"
            "\n===== RAW OUTPUT =====\n"
            f"{raw_output}\n"
            "===========================\n",
            flush=True,
        )

    output = clean_tex(raw_output)
    if not output:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Ollama returned no usable LaTeX output",
                "thinking_length": thinking_length,
            },
        )

    return {
        "output": output,
        "thinking_length": thinking_length,
    }