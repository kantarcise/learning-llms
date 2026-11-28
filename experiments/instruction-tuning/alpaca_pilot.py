"""Measure throughput and peak GPU memory for Exercise 7.3 (Stanford Alpaca)."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import tiktoken
import torch
from previous_chapters import GPTModel
from torch import nn
from torch.utils.data import DataLoader
from train import collate, prompt


def percentile(sorted_values: list[int], frac: float) -> int:
    index = min(len(sorted_values) - 1, int(frac * (len(sorted_values) - 1)))
    return sorted_values[index]


def measure_updates(
    model: GPTModel,
    batches: list[list[list[int]]],
    lr: float = 5e-5,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    tokens = 0
    widths: list[int] = []
    start = perf_counter()
    for batch in batches:
        inputs, targets = collate(batch)
        widths.append(int(inputs.shape[1]))
        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        optimizer.step()
        tokens += int(inputs.numel())
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated()
    return {
        "updates": len(batches),
        "batch_size": len(batches[0]),
        "seconds": seconds,
        "updates_per_second": len(batches) / seconds,
        "tokens_processed": tokens,
        "tokens_per_second": tokens / seconds,
        "min_width": min(widths),
        "mean_width": sum(widths) / len(widths),
        "max_width": max(widths),
        "peak_allocated_bytes": peak_bytes,
    }


def main(cache: Path, run: Path, batch_size: int, random_updates: int) -> None:
    records = json.loads((cache / "alpaca_data.json").read_text(encoding="utf-8"))
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = [
        tokenizer.encode(prompt(r) + "\n\n### Response:\n" + r["output"])
        for r in records
    ]
    train_end = int(0.85 * len(encoded))
    test_end = train_end + int(0.10 * len(encoded))
    train_encoded = encoded[:train_end]
    lengths = sorted(len(ids) for ids in train_encoded)
    over_1024 = sum(1 for n in lengths if n > 1024)

    config = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["config"]
    model = GPTModel(config)
    model.load_state_dict(
        torch.load(cache / "gpt2-medium-355M.pth", mmap=True, weights_only=True)
    )
    model.cuda()

    # Warm up CUDA allocator and kernels with 2 short batches before timing.
    warmup_loader = DataLoader(
        train_encoded[: batch_size * 2],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: b,
    )
    _ = measure_updates(model, list(warmup_loader))

    # Reload clean weights before the timed pilots.
    model.load_state_dict(
        torch.load(cache / "gpt2-medium-355M.pth", mmap=True, weights_only=True)
    )
    torch.manual_seed(123)
    rand_loader = DataLoader(
        train_encoded,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: b,
    )
    sampled_batches: list[list[list[int]]] = []
    for batch in rand_loader:
        sampled_batches.append(batch)
        if len(sampled_batches) == random_updates:
            break

    random_pilot = measure_updates(model, sampled_batches)

    # Worst-case memory test: longest batches (capped at 1,024 tokens by collate).
    longest_sorted = sorted(train_encoded, key=len, reverse=True)
    long_batches = [
        longest_sorted[i : i + batch_size] for i in range(0, batch_size * 5, batch_size)
    ]
    long_result: dict[str, object]
    try:
        long_result = measure_updates(model, long_batches)
        long_result["status"] = "ok"
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        long_result = {"status": "oom", "error": str(exc)}

    batches_per_epoch = len(train_encoded) // batch_size
    sec_per_update = float(random_pilot["seconds"]) / random_updates
    est_one_epoch_min = (batches_per_epoch * sec_per_update) / 60.0
    est_two_epochs_min = 2.0 * est_one_epoch_min

    summary = {
        "gpu": torch.cuda.get_device_name(),
        "total_records": len(records),
        "split_sizes": {
            "train": train_end,
            "test": test_end - train_end,
            "val": len(encoded) - test_end,
        },
        "train_length_stats": {
            "min": lengths[0],
            "median": percentile(lengths, 0.50),
            "mean": sum(lengths) / len(lengths),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": lengths[-1],
            "over_1024": over_1024,
        },
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "random_pilot": random_pilot,
        "longest_batches_pilot": long_result,
        "projected_minutes_one_epoch": est_one_epoch_min,
        "projected_minutes_two_epochs": est_two_epochs_min,
    }
    out_path = run / f"alpaca-pilot-bs{batch_size}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-updates", type=int, default=50)
    args = parser.parse_args()
    main(args.cache, args.run, args.batch_size, args.random_updates)
