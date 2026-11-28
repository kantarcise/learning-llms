# Month 3 — Remote vLLM Serving on the RTX 3090

This lab serves `Qwen/Qwen2.5-7B-Instruct` (BF16) on the Linux RTX 3090 using **vLLM** and measures end-to-end streaming latency from the Mac client against a pre-test **p95 TTFT ≤ 500 ms** target. Tracked in [Issue #65](https://github.com/kantarcise/learning-llms/issues/65).

## 1. Launch on the Linux RTX 3090

Install vLLM once into an isolated CUDA 12.6 virtual environment (matched to NVIDIA driver `560.35.03`):

```bash
uv venv --python 3.12 artifacts/serving/vllm-venv
uv pip install --python artifacts/serving/vllm-venv/bin/python --torch-backend=cu126 \
  "https://github.com/vllm-project/vllm/releases/download/v0.9.2/vllm-0.9.2%2Bcu126-cp38-abi3-manylinux1_x86_64.whl"
uv pip install --python artifacts/serving/vllm-venv/bin/python "transformers==4.53.2" "huggingface-hub<1.0"
```

Start the persistent server:

```bash
bash experiments/serving/serve-3090.sh
```

Defaults: `--host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85`.

## 2. Test from the Mac (zero dependencies)

```bash
# Verify endpoint reachability
curl http://192.168.1.15:8000/v1/models

# Stream one real response over SSE
curl -N http://192.168.1.15:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "In one sentence, name the capital of Japan and its main island."}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'

# Timed single streamed request (Python stdlib only — works directly on macOS)
python3 experiments/serving/client.py \
  --base-url http://192.168.1.15:8000 \
  --mode single \
  --prompt "In two concise sentences, explain why KV caching speeds up LLM decoding." \
  --max-tokens 96
```

## 3. Benchmark sweeps from the Mac (target: p95 TTFT ≤ 500 ms)

Keep output budget fixed at `--max-tokens 128`, and vary concurrency and prompt length separately:

```bash
# Sweep A — Concurrency at short prompts (~20 words, fixed 128 output tokens)
for c in 1 2 4 8 16; do
  echo "=== Concurrency $c ==="
  python3 experiments/serving/client.py \
    --base-url http://192.168.1.15:8000 \
    --mode bench \
    --concurrency "$c" \
    --requests $((c * 4)) \
    --approx-words 20 \
    --max-tokens 128 \
    --target-p95-ttft-ms 500
done

# Sweep B — Prompt length at concurrency 1 (fixed 128 output tokens)
for w in 20 250 1000 2500; do
  echo "=== Approx words $w ==="
  python3 experiments/serving/client.py \
    --base-url http://192.168.1.15:8000 \
    --mode bench \
    --concurrency 1 \
    --requests 6 \
    --approx-words "$w" \
    --max-tokens 128 \
    --target-p95-ttft-ms 500
done
```
