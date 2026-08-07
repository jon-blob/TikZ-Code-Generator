docker build -t tikzilla-8b-sft-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -p 8005:8005 \
  -e BASE_MODEL_PATH=nllg/TikZilla-8B \
  -e QUANTIZATION=8bit \
  -e HF_TOKEN="" \
  tikzilla-8b-sft-cuda128