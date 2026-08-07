#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1
export HF_HOME=/app/.cache/huggingface
export TRANSFORMERS_CACHE=/app/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p /app
cd /app

apt-get update

apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-pictures \
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    dvisvgm \
    ghostscript \
    poppler-utils

rm -rf /var/lib/apt/lists/*

python3 -m venv /opt/venv

export PATH=/opt/venv/bin:$PATH

pip install --upgrade pip setuptools wheel

pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

pip install \
    accelerate==1.8.1 \
    bitsandbytes \
    safetensors \
    transformers==4.53.2 \
    huggingface_hub \
    hf_transfer

pip install \
    fastapi \
    uvicorn \
    pydantic

mkdir -p /app

# cp app.py /app/app.py

ls -l /app