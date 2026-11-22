"""Replay open-loop arrivals against a single, length-bucketed CUDA worker."""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from model import Config, LanguageModel


def p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


@torch.inference_mode()
def replay(
    model: LanguageModel,
    prompts: torch.Tensor,
    lengths: list[int],
    arrivals: list[float],
    cached: bool,
    output_tokens: int,
    max_batch: int,
) -> tuple[dict, list[list[int]]]:
    """Drain fixed-size continuations; prompts are int64 [requests, max length]."""
    waiting: list[int] = []
    queue_ms: list[float] = []
    first_ms: list[float] = []
    gaps_ms: list[float] = []
    batches: Counter[int] = Counter()
    outputs: list[list[int]] = [[] for _ in arrivals]
    peak_cache = 0
    next_request = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    while next_request < len(arrivals) or waiting:
        if not waiting:
            delay = start + arrivals[next_request] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        now = time.perf_counter() - start
        while next_request < len(arrivals) and arrivals[next_request] <= now:
            waiting.append(next_request)
            next_request += 1
        if not waiting:
            continue
        length = lengths[waiting[0]]
        batch = [i for i in waiting if lengths[i] == length][:max_batch]
        selected = set(batch)
        waiting = [i for i in waiting if i not in selected]
        batches[len(batch)] += 1
        queue_ms.extend(1000 * (now - arrivals[i]) for i in batch)
        prefix = prompts[batch, :length]
        cache = None
        emitted = []
        previous = 0.0
        for step in range(output_tokens):
            if cached:
                scores, cache = model.cached_logits(prefix, cache)
            else:
                scores = model(prefix, last_only=True)
            token = scores.argmax(-1, keepdim=True)
            emitted.append(token)
            torch.cuda.synchronize()
            ready = time.perf_counter() - start
            if step:
                gaps_ms.extend([1000 * (ready - previous)] * len(batch))
            else:
                first_ms.extend(1000 * (ready - arrivals[i]) for i in batch)
            previous = ready
            if step + 1 < output_tokens:
                prefix = token if cached else torch.cat((prefix, token), dim=1)
        if cache:
            peak_cache = max(peak_cache, sum(t.nbytes for pair in cache for t in pair))
        for i, ids in zip(batch, torch.cat(emitted, dim=1).tolist(), strict=True):
            outputs[i] = ids
    elapsed = time.perf_counter() - start
    return {
        "queue_p95_ms": p95(queue_ms),
        "first_token_p95_ms": p95(first_ms),
        "inter_token_p95_ms": p95(gaps_ms),
        "output_tokens_per_second": len(arrivals) * output_tokens / elapsed,
        "elapsed_s": elapsed,
        "peak_logical_cache_bytes": peak_cache,
        "batch_size_histogram": dict(sorted(batches.items())),
    }, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("windows", type=Path)
    args = parser.parse_args()
    config = Config(
        **json.loads(args.weights.with_name("run.json").read_text())["config"]
    )
    model = LanguageModel(config)
    model.load_state_dict(
        torch.load(args.weights, weights_only=True, map_location="cpu")
    )
    model.cuda().eval()
    data = torch.load(args.windows, weights_only=True, map_location="cpu")
    rows = data["inputs"]
    seed, requests, budget, cap = 62, 192, 32, 16
    rng = random.Random(seed)
    lengths = [8, 64, 192] * (requests // 3)
    rng.shuffle(lengths)
    # Use only real next-token targets throughout each prompt.
    row_ids = [
        rng.choice((data["targets"][:, :n] >= 0).all(1).nonzero().flatten().tolist())
        for n in lengths
    ]
    prompts = rows[row_ids].long().cuda()
    offsets = [0.0]
    for _ in range(requests - 1):
        offsets.append(offsets[-1] + rng.expovariate(1))
    unit_trace = [t * (requests - 1) / offsets[-1] for t in offsets]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "windows_sha256": hashlib.sha256(args.windows.read_bytes()).hexdigest(),
        "seed": seed,
        "requests": requests,
        "output_tokens": budget,
        "batch_cap": cap,
        "prompt_lengths": lengths,
        "validation_rows": row_ids,
        "unit_rate_offsets_s": unit_trace,
        "runs": [],
    }
    reference = None
    # Warm each bucket at both batch-size extremes before timing.
    for n in sorted(set(lengths)):
        for size in (1, cap):
            for cached in (False, True):
                replay(model, prompts[:size], [n] * size, [0.0] * size, cached, 3, cap)
    rates = [25, 100, 250, 500, 1000]
    for repeat in range(2):
        for index, rate in enumerate(rates if repeat == 0 else reversed(rates)):
            arrivals = [t / rate for t in unit_trace]
            first_cached = bool((index + repeat) % 2)
            for cached in (first_cached, not first_cached):
                result, outputs = replay(
                    model, prompts, lengths, arrivals, cached, budget, cap
                )
                if reference is None:
                    reference = outputs
                if outputs != reference:
                    raise RuntimeError("Greedy IDs differ across replays")
                report["runs"].append(
                    dict(rate=rate, repeat=repeat, cached=cached, **result)
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
