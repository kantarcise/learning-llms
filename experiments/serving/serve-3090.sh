#!/usr/bin/env bash
# Launch Qwen/Qwen2.5-7B-Instruct on the Linux RTX 3090 via the isolated vLLM venv.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VLLM_BIN="${VLLM_BIN:-$PWD/artifacts/serving/vllm-venv/bin/vllm}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

mkdir -p "$PWD/artifacts/serving"
echo "Starting vLLM ($("$VLLM_BIN" --version 2>/dev/null || echo unknown)) for $MODEL on $HOST:$PORT"
exec "$VLLM_BIN" serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "$@"
