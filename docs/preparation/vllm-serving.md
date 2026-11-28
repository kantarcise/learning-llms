# Month 3 — Remote vLLM Serving on the RTX 3090

In [Month 2](cosmopedia.md#what-happens-as-requests-arrive-faster), we measured KV-cached generation inside a single Python process. Month 3 moves to a real client–server setup: a persistent **vLLM** engine running on the Linux RTX 3090 workstation, called over the home LAN from the Mac client (`sezaiMac`). Tracked in [Issue #65](https://github.com/kantarcise/learning-llms/issues/65).

## Host inventory and server setup

Before benchmarking, we audited the Linux host ([Issue #56](https://github.com/kantarcise/learning-llms/issues/56)) and reclaimed 85 GB of regenerable Docker and package caches (`392 GB → 306 GB` used, **111 GB free** on `/dev/sdb1`), preserving both operating systems and the shared EFI boot partition.

| Component | Verified specification |
| --- | --- |
| CPU & board | AMD Ryzen 5 3600 (6C/12T) on ASUSTeK ROG STRIX B450-F GAMING (AM4) |
| RAM & PSU | 31 GiB system RAM (0 swap); Corsair RM750 (750 W, 80 PLUS Gold) |
| GPU & driver | NVIDIA GeForce RTX 3090 (24 GiB VRAM), driver `560.35.03` (max CUDA 12.6) |
| Engine | `vllm 0.9.2+cu126` (`torch 2.7.0+cu126`, V1 engine, Flash Attention backend, `transformers 4.53.2`) |
| Model | `Qwen/Qwen2.5-7B-Instruct` (`bfloat16`, **14.25 GiB** weights) |
| KV cache pool | **4.48 GiB → 83,856 tokens** at default `max_num_batched_tokens=2048` (`4.27 GiB → 79,984 tokens` at `8192`) |
| Endpoint | `http://192.168.1.15:8000` (`/v1/chat/completions`, `/v1/models`, `/metrics`) |

`Qwen2.5-7B-Instruct` uses Grouped Query Attention (4 KV heads instead of 28 query heads), so each cached token occupies only **~56 KiB** in BF16—allowing 83,856 tokens (`10.24×` concurrency at the 8,192-token context cap) inside 4.48 GiB of remaining VRAM.

**Launch command** ([serve-3090.sh](https://github.com/kantarcise/learning-llms/blob/main/experiments/serving/serve-3090.sh)):

```bash
bash experiments/serving/serve-3090.sh
# Defaults: --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85
```

**Copyable Mac client request** ([client.py](https://github.com/kantarcise/learning-llms/blob/main/experiments/serving/client.py), Python stdlib only):

```bash
python3 experiments/serving/client.py \
  --base-url http://192.168.1.15:8000 \
  --mode single \
  --prompt "In two concise sentences, explain why KV caching speeds up LLM decoding." \
  --max-tokens 96
```

From `sezaiMac`, that single streamed request returned 46 tokens in **1,013.4 ms** with **65.6 ms** time-to-first-token (TTFT) and **21.1 ms/token** time-per-output-token (TPOT, ~47 tok/s).

## Benchmark design and latency target

Before running sweeps, we locked a client-side latency target of **p95 TTFT ≤ 500 ms** with `0` errors. The Mac client sent unique prompts (avoiding accidental prefix reuse except in Test C), kept `--max-tokens 128` fixed, and sampled vLLM's Prometheus `/metrics` endpoint before and after each run to separate queue wait, prefill, and decode time.

### Sweep A — Concurrency at short prompts (~54 prompt tokens)

| Concurrency | Reqs | Wall (s) | Req/s | Out tok/s | TTFT p50 / p95 (ms) | Total p95 (ms) | TPOT p50 (ms) | Server queue / prefill / decode (ms) | Target ≤ 500 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 4 | 3.71 | 1.08 | 46.4 | 37.4 / 38.5 | 941.0 | 21.4 | 0.1 / 20.8 / 871.8 | Pass |
| 2 | 8 | 3.65 | 2.19 | 93.7 | 40.0 / 98.7 | 931.4 | 20.7 | 0.0 / 21.1 / 859.1 | Pass |
| 4 | 16 | 3.78 | 4.23 | 180.3 | 55.9 / 122.4 | 960.8 | 20.9 | 0.0 / 25.1 / 867.9 | Pass |
| 8 | 32 | 3.94 | 8.12 | 345.2 | 58.2 / 118.0 | 1046.5 | 21.7 | 0.0 / 36.9 / 881.7 | Pass |
| 16 | 64 | 4.17 | 15.36 | **630.6** | 70.1 / **106.5** | 1071.8 | 22.9 | 0.0 / 35.1 / 916.7 | Pass |

Every level passed with `0` errors and `0` preemptions. Output throughput scaled **13.6×** from `c=1` to `c=16` while per-user TPOT rose only **7%** (`21.4 → 22.9 ms`). Decode at batch size 1 is memory-bandwidth-bound (reading 14.25 GiB of BF16 weights per token), so continuous batching amortizes that weight read across all 16 concurrent requests.

### Sweep B — Prompt length at `c=1`, before and after tuning `max_num_batched_tokens`

With vLLM V1's default `--max-num-batched-tokens 2048`, a 4,203-token prompt missed the 500 ms target (**673.2 ms** p95 TTFT, **588.7 ms** server prefill, **0.1 ms** queue wait). Because vLLM split 4,203 tokens into three 2,048-token prefill chunks, we tested the smallest single-host change: raising `--max-num-batched-tokens 8192` so long prompts run in one unchunked pass.

| Mean prompt tokens | Baseline (`2048`) prefill / p95 TTFT | Tuned (`8192`) prefill / p95 TTFT | TPOT p50 (ms) | Target ≤ 500 ms |
| ---: | ---: | ---: | ---: | :---: |
| 54 | 23.2 ms / 45.8 ms | 23.3 ms / 79.7 ms | 20.9 | Pass |
| 431 | 119.2 ms / 153.5 ms | 122.8 ms / 142.9 ms | 21.1 | Pass |
| 1,686 | 272.9 ms / 330.6 ms | 278.3 ms / 340.0 ms | 21.1 | Pass |
| **4,203** | **588.7 ms / 673.2 ms** | **543.7 ms / 662.5 ms** | 21.1 | **Fail** |

Removing the three-chunk split shaved **45.0 ms (-7.6%)** off server prefill (`588.7 → 543.7 ms`) at the cost of ~210 MiB of KV cache (`83,856 → 79,984` tokens). Even in one unchunked pass, `543.7 ms` (~7,730 prefill tok/s, ~108 TFLOPS effective BF16) reflects the RTX 3090's dense compute roofline for a 7B model rather than queuing or memory pressure.

### Bonus tests — Automatic Prefix Caching (Test C) and concurrent long prompts (Test D)

| Test | Workload (`max_num_batched_tokens=8192`) | Mean prompt tok | Out tok/s | TTFT p50 / p95 (ms) | TPOT p50 (ms) | Prefix cache hit | Target ≤ 500 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **Test C** | `c=1`, 6 reqs, shared ~4.2k document + unique tail question | 4,227 | 38.5 | **51.0** / 792.5 | 21.5 | **82.7%** | Fail (cold req 1) |
| **Test D** | `c=4`, 12 reqs, ~1.7k prompts | 1,695 | **146.9** | 478.4 / 1021.8 | 22.5 | 50.0%* | Fail |

- **Test C per-request TTFT (ms):** `[1023.5, 44.5, 49.2, 99.6, 45.5, 52.8]`. Once Request 1 populated the shared document's KV blocks, Requests 2–6 dropped by more than **10×** (`44.5–99.6 ms`), turning a 660 ms 4k-token prefill into a ~50 ms cache lookup.
- **Test D:** Four simultaneous ~1.7k-token prompts (~6,780 prefill tokens batched inside `max_num_batched_tokens=8192`) stacked prefill compute to `1,021.8 ms` p95 TTFT while boosting output throughput 3.4× to `146.9 tok/s`. *Note:* `prefix_cache_hit_pct = 50.0%` occurred because requests 0–5 reused prompt strings still cached from Sweep B ten minutes earlier, while requests 6–11 were cold prefills.

## Keep-or-upgrade decision (Issue #65)

**Keep the single RTX 3090.** It meets `p95 TTFT ≤ 500 ms` up to **16 concurrent short requests (15.36 req/s, 630.6 tok/s, 106.5 ms p95 TTFT)** and single-caller prompts up to **1,686 tokens (331 ms p95 TTFT)** with zero errors and zero preemptions. The only missed target—cold 4,203-token 7B BF16 prefill (`544 ms` server prefill)—is single-request compute-bound: a second replica (`DP=2`) would not speed it up (`queue_mean` was `0.1 ms`), and adding a second 3090 for `TP=2` exceeds the 750 W PSU and B450 PCIe slot budget. Keeping per-page chunks ≤ ~2k tokens and using prefix caching on shared context fits Month 4 and Month 5 cleanly on existing hardware.

[Recorded JSON results](https://github.com/kantarcise/learning-llms/blob/main/experiments/serving/results/rtx3090-mac-baseline.json) · [Issue #65](https://github.com/kantarcise/learning-llms/issues/65)
