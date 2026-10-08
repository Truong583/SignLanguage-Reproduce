#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data assets runs
IMAGE="${IMAGE:-mixsigngraph-repro:local}"
MEMORY="${MEMORY:-16g}"
CPUS="${CPUS:-4}"
GPU_IDS="${GPU_IDS:-0}"
SHM_SIZE="${SHM_SIZE:-2g}"
NETWORK="${NETWORK:-none}"
DATA_MODE="${DATA_MODE:-ro}"
ASSET_MODE="${ASSET_MODE:-ro}"
case "$DATA_MODE:$ASSET_MODE" in ro:ro|rw:ro|ro:rw|rw:rw) ;; *) echo 'Invalid mount mode' >&2; exit 2;; esac
gpu_args=()
if [[ "$GPU_IDS" != "none" ]]; then
  if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then echo 'GPU_IDS must be an explicit list, e.g. 0,1,2' >&2; exit 2; fi
  gpu_args=(--gpus "\"device=$GPU_IDS\"")
fi
exec docker run --rm --init \
  --user "$(id -u):$(id -g)" \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 2048 --memory "$MEMORY" --memory-swap "$MEMORY" --cpus "$CPUS" \
  --shm-size "$SHM_SIZE" --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --network "$NETWORK" "${gpu_args[@]}" \
  --mount "type=bind,src=$ROOT,dst=/workspace,readonly" \
  -v "$ROOT/data:/workspace/data:$DATA_MODE" \
  -v "$ROOT/assets:/workspace/assets:$ASSET_MODE" \
  -v "$ROOT/runs:/workspace/runs:rw" \
  --workdir /workspace "$IMAGE" "$@"
