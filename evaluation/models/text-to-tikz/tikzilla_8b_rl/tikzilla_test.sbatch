#!/bin/bash
#SBATCH --job-name=tikzilla-standalone-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --time=0-02:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course
#SBATCH --mail-type=ALL
#SBATCH --mail-user=slurmbot-!rdElDorfwPMmeWVLPg:matrix.org@vision.in.tum.de,s0042
#SBATCH --output=/usr/prakt/s0042/projects/tikzcodegenerator/evaluation/logs/tikzilla_standalone-%j.out
#SBATCH --error=/usr/prakt/s0042/projects/tikzcodegenerator/evaluation/logs/tikzilla_standalone-%j.err

set -euo pipefail

echo "Job started on $(hostname)"
date
nvidia-smi

PROJECT=/usr/prakt/s0042/projects/tikzcodegenerator
LOG_DIR=$PROJECT/evaluation/logs

TIKZILLA_SQSH=$PROJECT/evaluation/models/tikzilla_8b_rl/tikzilla-8b-rl.sqsh  
CLIENT_DIR=$PROJECT/evaluation/models/tikzilla_8b_rl
TIKZILLA_CONTAINER="tikzilla-client-${SLURM_JOB_ID}" 

TEST_PROMPT="a histogram with 10 random bins "

mkdir -p "$LOG_DIR"

cleanup() {
  echo "Stopping services and cleaning up container..."
  kill "${CLIENT_PID:-}" 2>/dev/null || true
  enroot remove -f "$TIKZILLA_CONTAINER" 2>/dev/null || true
}
trap cleanup EXIT

########test code##############
echo "===== HOST ====="
hostname

echo "===== Host environment ====="
id
echo "HOME=$HOME"

echo "===== XDG ====="
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"

echo "===== ENROOT ====="
echo "ENROOT_RUNTIME_PATH=$ENROOT_RUNTIME_PATH"
echo "ENROOT_CACHE_PATH=$ENROOT_CACHE_PATH"
echo "ENROOT_DATA_PATH=$ENROOT_DATA_PATH"

echo "===== /run/user ====="
ls -ld /run
ls -ld /run/user || true
ls -ld /run/user/$(id -u) || true

env | sort


echo "before"

id

echo "$XDG_RUNTIME_DIR"

enroot start alpine true

echo "after"
#########################################

echo "Preparing Tikzilla Enroot container..."
enroot create --name "$TIKZILLA_CONTAINER" "$TIKZILLA_SQSH"

echo "Starting Tikzilla service inside Enroot..."
enroot start \
  --rw \
  --env XDG_RUNTIME_DIR="" \
  --mount $CLIENT_DIR:/app \
  --mount "/usr/prakt/s0042/projects/models:/models" \
  --env NVIDIA_VISIBLE_DEVICES=all \
  "$TIKZILLA_CONTAINER" \
  sh -lc '
  set -x
  cd /app
  python3 app.py
  '
  # sh -lc 'cd /app && python3 -m uvicorn app:app --host 0.0.0.0 --port 8006' \
  > "$LOG_DIR/tikzilla-client-${SLURM_JOB_ID}.log" 2>&1 &

CLIENT_PID=$!


echo "Waiting for Tikzilla service to come online..."
READY=0
for i in {1..60}; do
  if curl -s http://127.0.0.1:8006/docs >/dev/null; then
    echo "Tikzilla is ready."
    READY=1
    break
  fi
  sleep 2
done

if [ "$READY" -eq 0 ]; then
    echo "Tikzilla failed to start."
    cat "$LOG_DIR/tikzilla-client-${SLURM_JOB_ID}.log"
    exit 1
fi


echo "Sending test request to Tikzilla..."
curl http://127.0.0.1:8006/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$TEST_PROMPT\"}"
  > "$LOG_DIR/response_${SLURM_JOB_ID}.json"

echo
echo "Job finished."
date