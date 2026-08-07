docker build -t tikzero-plus-10b-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4:/DaTikZ-V4" \
  -v "/home/jonas/models:/models/" \
  tikzero-plus-10b-cuda128 \
  bash

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/models:/models/" \
  -p 8002:8002 \
  --env-file .env \
  tikzero-plus-10b-cuda128