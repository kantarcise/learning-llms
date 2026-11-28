"""Zero-dependency streaming client and latency benchmark for vLLM (Month 3 lab).

Uses only the Python standard library so it runs identically on macOS and Linux.
"""

import argparse
import http.client
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.parse import urlparse


def percentile(values: list[float], pct: float) -> float:
    """Return the linear-interpolated percentile (0-100) of a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    frac = rank - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def make_prompt(request_id: int, approx_words: int, shared_prefix: bool = False) -> str:
    """Build a prompt of controlled length, optionally sharing a long prefix across requests."""
    if approx_words <= 25:
        return (
            f"Request #{request_id}: In one sentence, what is the capital of France "
            "and why is the Seine river associated with it?"
        )
    if shared_prefix:
        # Shared document first so vLLM Automatic Prefix Caching reuses its KV blocks;
        # unique question at the tail so every request still computes a fresh answer.
        header = (
            "Shared engineering log for prefix-caching test. Read the log below and "
            "answer the specific question at the end concisely.\n\n"
        )
        sentences = [
            f"Batch 0-{i}: worker {i % 7} processed {(i * 13) % 97 + 10} records "
            f"with queue wait {(i * 17) % 85 + 5} ms and cache hit ratio {60 + (i % 35)}%."
            for i in range(max(1, approx_words // 18))
        ]
        tail = (
            f"\n\nQuestion #{request_id}: Focusing on worker {request_id % 7}, "
            "summarize the main bottleneck and one concrete fix in two sentences."
        )
        return header + " ".join(sentences) + tail

    header = (
        f"Request #{request_id}. Read the short engineering log below and reply in "
        "plain sentences explaining the main bottleneck and one concrete fix.\n\n"
    )
    sentences = [
        f"Batch {request_id}-{i}: worker {i % 7} processed {(i * 13) % 97 + 10} records "
        f"with queue wait {(i * 17) % 85 + 5} ms and cache hit ratio {60 + (i % 35)}%."
        for i in range(max(1, approx_words // 18))
    ]
    return header + " ".join(sentences)


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Extract key vLLM Prometheus counters and gauges from /metrics text."""
    wanted_prefixes = (
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:inter_token_latency_seconds_sum",
        "vllm:inter_token_latency_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:num_preemptions_total",
        "vllm:kv_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, _, raw_value = line.rpartition(" ")
        metric_name = name_and_labels.split("{", 1)[0]
        if metric_name in wanted_prefixes:
            try:
                out[metric_name] = float(raw_value)
            except ValueError:
                continue
    return out


def fetch_metrics(base_url: str) -> dict[str, float]:
    """Fetch and parse vLLM /metrics; returns empty dict if unreachable."""
    parsed = urlparse(base_url)
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=10)
    try:
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        if resp.status != 200:
            return {}
        return parse_prometheus_metrics(body)
    finally:
        conn.close()


def stream_chat_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    echo_stream: bool = False,
) -> dict[str, object]:
    """Send one streaming /v1/chat/completions request and time TTFT and completion."""
    parsed = urlparse(base_url)
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")

    conn = conn_cls(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=300)
    t_start = perf_counter()
    t_first_text: float | None = None
    chunks: list[str] = []
    usage: dict[str, int] = {}
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "status": resp.status,
                "error": err_body[:400],
                "total_ms": (perf_counter() - t_start) * 1000.0,
            }
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            event = json.loads(data_str)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content") or ""
            if text:
                if t_first_text is None:
                    t_first_text = perf_counter()
                chunks.append(text)
                if echo_stream:
                    sys.stdout.write(text)
                    sys.stdout.flush()
        t_end = perf_counter()
        if echo_stream:
            sys.stdout.write("\n")
            sys.stdout.flush()
        ttft_ms = (t_first_text - t_start) * 1000.0 if t_first_text else None
        total_ms = (t_end - t_start) * 1000.0
        comp_tokens = int(usage.get("completion_tokens", len(chunks)))
        tpot_ms = (
            ((t_end - t_first_text) * 1000.0 / max(1, comp_tokens - 1))
            if t_first_text and comp_tokens > 1
            else None
        )
        return {
            "ok": True,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "tpot_ms": tpot_ms,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": comp_tokens,
            "text": "".join(chunks),
        }
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "total_ms": (perf_counter() - t_start) * 1000.0,
        }
    finally:
        conn.close()


def run_benchmark(
    base_url: str,
    model: str,
    concurrency: int,
    total_requests: int,
    approx_words: int,
    max_tokens: int,
    target_p95_ttft_ms: float,
    shared_prefix: bool = False,
) -> dict[str, object]:
    """Run a concurrent workload and summarize latency, throughput, and server deltas."""
    before_metrics = fetch_metrics(base_url)
    wall_start = perf_counter()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                stream_chat_request,
                base_url,
                model,
                make_prompt(i, approx_words, shared_prefix=shared_prefix),
                max_tokens,
            )
            for i in range(total_requests)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall_seconds = perf_counter() - wall_start
    after_metrics = fetch_metrics(base_url)

    ok_runs = [r for r in results if r.get("ok")]
    errors = [r for r in results if not r.get("ok")]
    ttfts = [float(r["ttft_ms"]) for r in ok_runs if r.get("ttft_ms") is not None]
    totals = [float(r["total_ms"]) for r in ok_runs]
    tpots = [float(r["tpot_ms"]) for r in ok_runs if r.get("tpot_ms") is not None]
    prompt_toks = [int(r["prompt_tokens"]) for r in ok_runs]
    comp_toks = [int(r["completion_tokens"]) for r in ok_runs]

    def delta(key: str) -> float | None:
        if key in before_metrics and key in after_metrics:
            return after_metrics[key] - before_metrics[key]
        return None

    def server_mean_ms(prefix: str) -> float | None:
        d_sum = delta(f"{prefix}_sum")
        d_cnt = delta(f"{prefix}_count")
        if d_sum is not None and d_cnt and d_cnt > 0:
            return (d_sum / d_cnt) * 1000.0
        return None

    cache_hits = delta("vllm:prefix_cache_hits_total")
    cache_queries = delta("vllm:prefix_cache_queries_total")
    cache_hit_pct = (
        round((cache_hits / cache_queries) * 100.0, 1)
        if cache_hits is not None and cache_queries and cache_queries > 0
        else None
    )

    p95_ttft = percentile(ttfts, 95) if ttfts else None
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "approx_words": approx_words,
        "shared_prefix": shared_prefix,
        "max_tokens": max_tokens,
        "completed": len(ok_runs),
        "errors": len(errors),
        "wall_seconds": round(wall_seconds, 2),
        "requests_per_second": round(len(ok_runs) / wall_seconds, 2),
        "output_tokens_per_second": round(sum(comp_toks) / wall_seconds, 1),
        "mean_prompt_tokens": round(mean(prompt_toks), 1) if prompt_toks else 0.0,
        "mean_completion_tokens": round(mean(comp_toks), 1) if comp_toks else 0.0,
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 1),
            "p95": round(p95_ttft, 1),
            "max": round(max(ttfts), 1),
        }
        if ttfts and p95_ttft is not None
        else None,
        "per_request_ttft_ms": [round(t, 1) for t in ttfts] if shared_prefix else None,
        "total_ms": {
            "p50": round(percentile(totals, 50), 1),
            "p95": round(percentile(totals, 95), 1),
            "max": round(max(totals), 1),
        }
        if totals
        else None,
        "tpot_ms_p50": round(percentile(tpots, 50), 1) if tpots else None,
        "target_p95_ttft_ms": target_p95_ttft_ms,
        "meets_target": (
            p95_ttft is not None and p95_ttft <= target_p95_ttft_ms and len(errors) == 0
        ),
        "server_timings_ms": {
            "queue_mean": round(v, 1)
            if (v := server_mean_ms("vllm:request_queue_time_seconds")) is not None
            else None,
            "prefill_mean": round(v, 1)
            if (v := server_mean_ms("vllm:request_prefill_time_seconds")) is not None
            else None,
            "decode_mean": round(v, 1)
            if (v := server_mean_ms("vllm:request_decode_time_seconds")) is not None
            else None,
            "preemptions": delta("vllm:num_preemptions_total"),
            "prefix_cache_hit_pct": cache_hit_pct,
        },
        "error_samples": errors[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mode", choices=["single", "bench"], default="single")
    parser.add_argument(
        "--prompt",
        default="In two sentences, explain why KV caching speeds up autoregressive decoding.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--approx-words", type=int, default=20)
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--target-p95-ttft-ms", type=float, default=500.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "single":
        res = stream_chat_request(
            args.base_url,
            args.model,
            args.prompt,
            args.max_tokens,
            echo_stream=True,
        )
        summary = {k: v for k, v in res.items() if k != "text"}
        print("\n--- Request timings ---")
        print(json.dumps(summary, indent=2))
        return

    report = run_benchmark(
        args.base_url,
        args.model,
        args.concurrency,
        args.requests,
        args.approx_words,
        args.max_tokens,
        args.target_p95_ttft_ms,
        shared_prefix=args.shared_prefix,
    )
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
