docker build -t automatikz-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/path/to/models:/models/" \
  -p 8003:8003 \
  --env-file .env \
  automatikz-cuda128
