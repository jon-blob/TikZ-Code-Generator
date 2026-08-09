# Ollama TikZ API

## Docker

```bash
docker build -t ollama-tikz-api .

docker run --rm -it \
  --network host \
  ollama-tikz-api
```

## Conda

```bash
conda env create -f environment.yml
conda activate ollama-client
uvicorn app:app --host 0.0.0.0 --port 8444
```

## Enroot: TikZ API

```bash
docker build -t ollama-tikz-api:latest .

enroot import \
  --output ollama-tikz-api.sqsh \
  dockerd://ollama-tikz-api:latest
```

Copy `ollama-tikz-api.sqsh` to the server, then start it:

```bash
enroot start \
  --rw \
  --env OLLAMA_URL=http://127.0.0.1:11434/api/chat \
  ollama-tikz-api.sqsh
```

## Enroot: Ollama Server

```bash
enroot import \
  --output ollama-server.sqsh \
  docker://ollama/ollama:latest

mkdir -p "$HOME/ollama-models"

NVIDIA_VISIBLE_DEVICES=all \
NVIDIA_DRIVER_CAPABILITIES=compute,utility \
enroot start \
  --rw \
  --mount "$HOME/ollama-models:/models" \
  --env OLLAMA_MODELS=/models \
  --env OLLAMA_HOST=0.0.0.0:11434 \
  ollama-server.sqsh \
  ollama serve
```

## Download a Model

```bash
curl http://127.0.0.1:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6:35b-a3b","stream":false}'
```


## add new models
ollama create gemma4-31B-it-tikz-sft-normal-1100-q4_k_m -f Modelfile
ollama show gemma4-31B-it-tikz-sft-normal-1100-q4_k_m