docker build -t tikzilla-8b-rl-cuda128 .

docker run --rm -it \
  --gpus all \
  -v "$PWD:/app" \
  -p 8006:8006 \
  -e BASE_MODEL_PATH=nllg/TikZilla-8B-RL \
  -e QUANTIZATION=8bit \
  -e HF_TOKEN="" \
  tikzilla-8b-rl-cuda128

Run the build_sqsh.sh to build a modified image from the official cuda image,
the environment modification is inside the setup_container.sh.

<!-- # Tkzilla with Enroot

## 0. Directly export Enroot `.sqsh`

Create a `build.conf` file out of the `Dockerfile`

Then create a temporal container, adjust the name as wish:

```bash
enroot batch build.conf tikzilla-8b-rl-temp
```

Check the temporal container list with:

```bash
enroot list
```

Copy the app.py file into the temporal container:

```bash
# enroot cp app.py tikzilla-8b-rl-temp:/app/app.py
```

Export the temporal container into a `sqsh` file:

```bash
enroot export -o tikzilla-8b-rl.sqsh tikzilla-8b-rl-temp
```

Delete the temporal container:

```bash
enroot remove tikzilla-8b-rl-temp
```

Copy the `.sqsh` file to the cluster.

---

## 1. Create the Enroot container on the cluster

```bash
enroot create --name tikzilla-8b-rl tikzilla-8b-rl.sqsh
```

Copy the app.py file into the container for testing:

```bash
enroot cp app.py tikzilla-8b-rl:/app/app.py
```

```bash
# enroot start --rw --env NVIDIA_VISIBLE_DEVICES=all my-app-container uvicorn app:app --host 0.0.0.0 --port 8006
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
 -->
