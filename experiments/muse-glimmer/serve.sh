#!/usr/bin/env bash
set -euo pipefail

muse_dir="${MUSE_DIR:-$HOME/.local/share/muse-glimmer}"

exec env LD_LIBRARY_PATH="$muse_dir/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$muse_dir/runtime/llama-server" \
  -m "$muse_dir/models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf" \
  --alias muse-glimmer-30B --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 -np 1 --jinja \
  --chat-template-kwargs '{"reasoning_strength":"low"}' \
  --temp 1.0 --top-p 0.95 --top-k 64 "$@"
