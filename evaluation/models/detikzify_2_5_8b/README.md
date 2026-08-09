docker build -t detikzify-2-5-8b-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/Datasets/TikZ/DaTikZ-V4:/DaTikZ-V4" \
  -v "/home/jonas/models:/models/" \
  detikzify-2-5-8b-cuda128 \
  bash


docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/models:/models/" \
  -p 8000:8000 \
  -e QUANTIZATION=8bit \
  detikzify-2-5-8b-cuda128


docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -v "/home/jonas/models:/models/" \
  -p 8000:8000 \
  -e QUANTIZATION=none \
  detikzify-2-5-8b-cuda128



on the server:

apptainer build detikzify-2-5-8b.sif docker-archive:///usr/prakt/s0030/projects/tikzcodegenerator/evaluation/models/detikzify_2_5_8b/detikzify_2_5_8b.tar


apptainer run --nv \
  --bind "$PWD:/app" \
  --bind "/usr/prakt/s0030/projects/models:/models" \
  --env QUANTIZATION=none \
  --pwd /app \
  detikzify-2-5-8b.sif



# DeTikZify with Enroot

## 0. Convert Docker `.tar` to Enroot `.sqsh`

Run this on a machine that has **Docker + Enroot** installed.

```bash
docker build -t detikzify-2-5-8b-cuda128 .
```

Then import the Docker image into Enroot.

Adjust the image name/tag according to the output of `docker images`:

```bash
enroot import --output detikzify-2-5-8b.sqsh dockerd://detikzify-2-5-8b-cuda128:latest
```

Copy the `.sqsh` file to the cluster.

---

## 1. Create the Enroot container on the cluster

```bash
enroot create --name detikzify detikzify-2-5-8b.sqsh
```

---

## 2. Check required host paths

```bash
ls -lah /usr/prakt/s0030/projects/models
ls -lah /usr/prakt/s0030/projects/tikzcodegenerator/evaluation/models/detikzify_2_5_8b
```

The host-side paths used in `--mount` must already exist.

---

## 3. TMUX commands
Start a new tmux:
```bash
tmux new -s detikzfy2-5
```

To exit: Ctrl-b, then d

to join a running tmux:

```bash
tmux attach -t promptfoo_splits
```

List running tmux:
```bash
tmux ls
```

Stop a tmux:
```bash
exit
```
or 
```bash
tmux kill-session -t promptfoo_splits
```

## 4. Start DeTikZify

Run this on a GPU node or inside a Slurm job.
```bash
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

```bash
cd /usr/prakt/s0030/projects/tikzcodegenerator/evaluation/models/detikzify_2_5_8b

enroot start --root --rw \
  --mount "$PWD:/app" \
  --mount "/usr/prakt/s0030/projects/models:/models" \
  --env QUANTIZATION=none \
  detikzify \
  sh -lc 'cd /app && python3 -m uvicorn app:app --host 0.0.0.0 --port 8000'
```

If the image already has a correct default command, you can also try:

```bash
cd /usr/prakt/s0030/projects/tikzcodegenerator/evaluation/models/detikzify_2_5_8b

enroot start --root --rw \
  --mount "$PWD:/app" \
  --mount "/usr/prakt/s0030/projects/models:/models" \
  --env QUANTIZATION=none \
  detikzify
```

---

