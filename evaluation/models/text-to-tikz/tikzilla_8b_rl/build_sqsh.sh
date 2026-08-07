#!/bin/bash
set -e

PROJECT="$PRAKT_DIR/projects/tikzcodegenerator"
export ENROOT_CACHE_PATH=/usr/prakt/s0042/projects/enroot-cache
export ENROOT_DATA_PATH=/usr/prakt/s0042/projects/enroot-data

mkdir -p $ENROOT_CACHE_PATH
mkdir -p $ENROOT_DATA_PATH

IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
NAME=tikzilla-8b-rl

echo "Importing image..."
enroot import docker://$IMAGE

SQSH=$(ls *12.8.1-cudnn-devel-ubuntu24.04*.sqsh)

echo "Creating container..."
enroot create --name $NAME "$SQSH"

echo "Running setup..."
enroot start \
    --root \
    --rw \
    --mount $PWD:/app \
    --mount "/usr/prakt/s0042/projects/models:/models" \
    --env QUANTIZATION=none \
    $NAME \
    bash /app/setup_container.sh

echo "Exporting..."
enroot export --output ${NAME}.sqsh $NAME

echo "Done."