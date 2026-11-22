"""Compare identical greedy continuations with and without a per-request KV cache."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import torch
from model import Config, LanguageModel


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def verify(model: LanguageModel, prompt: torch.Tensor, count: int) -> float:
    """Compare all vocabulary logits and greedy IDs at every prediction."""
    prefix = prompt
    cache = None
    largest_error = 0.0
    for _ in range(count):
        reference = model(prefix)[:, -1]
        uncached = model(prefix, last_only=True)
        scores, cache = model.cached_logits(
            prefix if cache is None else prefix[:, -1:], cache
        )
        for actual in (uncached, scores):
            torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(actual.argmax(-1), reference.argmax(-1))
            largest_error = max(largest_error, (actual - reference).abs().max().item())
        prefix = torch.cat((prefix, reference.argmax(-1, keepdim=True)), dim=1)
    return largest_error


@torch.inference_mode()
def measure(
    model: LanguageModel, prompt: torch.Tensor, count: int, cached: bool
) -> dict:
    """Time prefill with first selection, then count-1 decode steps. Batch size one."""
    device = prompt.device
    synchronize(device)
    start = perf_counter()
    if cached:
        scores, cache = model.cached_logits(prompt)
    else:
        scores = model(prompt, last_only=True)
    token = scores.argmax(-1, keepdim=True)
    synchronize(device)
    first = perf_counter()
    prefix = prompt
    for _ in range(count - 1):
        if cached:
            scores, cache = model.cached_logits(token, cache)
        else:
            prefix = torch.cat((prefix, token), dim=1)
            scores = model(prefix, last_only=True)
        token = scores.argmax(-1, keepdim=True)
    synchronize(device)
    elapsed = perf_counter() - first
    return {
        "prefill_ms": (first - start) * 1000,
        "decode_ms_per_token": elapsed * 1000 / (count - 1),
        "logical_cache_bytes": sum(
            t.numel() * t.element_size() for pair in cache for t in pair
        )
        if cached
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights", type=Path, help="Inference state_dict beside run.json"
    )
    parser.add_argument("windows", type=Path, help="Prepared validation.pt")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = Config(
        **json.loads(args.weights.with_name("run.json").read_text())["config"]
    )
    model = LanguageModel(config)
    model.load_state_dict(
        torch.load(args.weights, weights_only=True, map_location="cpu")
    )
    model.to(device).eval()
    row = torch.load(args.windows, weights_only=True, map_location="cpu")["inputs"][0]
    report = {
        "device": str(device),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "dtype": str(model.token_embedding.weight.dtype),
        "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "config": model.configuration(),
        "generated_tokens": 32,
        "lengths": [],
    }
    for length in (8, 64, 192):
        prompt = row[:length].long().unsqueeze(0).to(device)
        error = verify(model, prompt, 32)
        results = {
            "prompt_ids": prompt[0].tolist(),
            "max_abs_error": error,
            "cached": [],
            "uncached": [],
        }
        for cached in (False, True):
            measure(model, prompt, 32, cached)
        for repeat in range(6):
            for cached in (bool(repeat % 2), not bool(repeat % 2)):
                results["cached" if cached else "uncached"].append(
                    measure(model, prompt, 32, cached)
                )
        report["lengths"].append(results)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
