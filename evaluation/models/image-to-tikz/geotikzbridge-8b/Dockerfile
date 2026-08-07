FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV HF_HUB_ENABLE_HF_TRANSFER=1


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-pictures \
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    dvisvgm \
    ghostscript \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN pip install --upgrade pip setuptools wheel

# CUDA-enabled PyTorch wheels.
RUN pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Minimal GeoTikzBridge / InternVL runtime requirements.
# Excluded:
# - decord: video decoding, problematic on Python 3.12
# - deepspeed: heavy training/inference framework, not needed for this FastAPI app
# - torch/torchvision: installed explicitly above as CUDA wheels
RUN pip install \
    "accelerate<1" \
    bitsandbytes \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    huggingface_hub \
    hf_transfer \
    imageio \
    numpy==1.26.4 \
    opencv-python-headless \
    orjson \
    peft==0.10.0 \
    pyyaml \
    scipy \
    sentencepiece==0.2.1 \
    shortuuid \
    termcolor \
    timm==0.9.12 \
    tokenizers==0.15.1 \
    tqdm \
    transformers==4.37.2 \
    yacs \
    safetensors \
    fastapi \
    uvicorn \
    python-multipart \
    pillow

COPY app.py /app/app.py

EXPOSE 8007

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007"]