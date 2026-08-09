docker build -t geotikzbridge-8b-cuda128 .

docker run --rm -it \
--gpus all \
--env-file .env \
-v "$PWD:/app" \
-v "/home/jonas/models:/models" \
-p 8007:8007 \
geotikzbridge-8b-cuda128


