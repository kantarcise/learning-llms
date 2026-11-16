"""Compare uninterrupted training with a stopped/resumed run in fresh processes.

Uses real prepared data by default. Reports exact state equality, not just a
similar loss. All checkpoints and logs remain under the chosen local output.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


def compare_state(left: object, right: object, path: str = "checkpoint") -> int:
    """Require exact nested equality and return the number of compared tensors."""
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or not torch.equal(left, right)
        ):
            raise AssertionError(f"Tensor mismatch: {path}")
        return 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Dictionary mismatch: {path}")
        return sum(
            compare_state(value, right[key], f"{path}.{key}")
            for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"Sequence mismatch: {path}")
        return sum(
            compare_state(a, b, f"{path}[{index}]")
            for index, (a, b) in enumerate(zip(left, right, strict=True))
        )
    if left != right:
        raise AssertionError(f"Value mismatch: {path}: {left!r} != {right!r}")
    return 0


def main() -> None:
    """Run three bounded training processes and persist evidence of continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "artifacts/data"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interrupt-after", type=int, default=37)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()
    if not 0 < args.interrupt_after < args.steps or args.eval_every < 1:
        parser.error(
            "Require 0 < interrupt-after < steps and a positive evaluation interval"
        )
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    commands: dict[str, list[str]] = {}
    for name, extra in [
        ("uninterrupted", []),
        ("interrupted", ["--stop-after", str(args.interrupt_after)]),
        ("resumed", ["--resume", str(args.output / "interrupted/latest.pt")]),
    ]:
        command = [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--data",
            str(args.data),
            "--output",
            str(args.output / name),
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--eval-every",
            str(args.eval_every),
            *extra,
        ]
        commands[name] = command
        print(f"Running {name}: {' '.join(command)}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (args.output / f"{name}.log").write_text(result.stdout + result.stderr)
        print(result.stdout, result.stderr, flush=True)
        result.check_returncode()
    full = torch.load(
        args.output / "uninterrupted/latest.pt", map_location="cpu", weights_only=True
    )
    resumed = torch.load(
        args.output / "resumed/latest.pt", map_location="cpu", weights_only=True
    )
    compared_tensors = compare_state(full, resumed)
    full_best = torch.load(
        args.output / "uninterrupted/best.pt", map_location="cpu", weights_only=True
    )
    resumed_best = torch.load(
        args.output / "resumed/best.pt", map_location="cpu", weights_only=True
    )
    compare_state(full_best, resumed_best)
    reports = {
        name: json.loads((args.output / name / "run.json").read_text())
        for name in commands
    }
    for key in [
        "history",
        "before",
        "after",
        "best_step",
        "training_target_tokens_seen",
        "validation_indices",
    ]:
        compare_state(reports["uninterrupted"][key], reports["resumed"][key], key)
    report = {
        "device": args.device,
        "steps": args.steps,
        "interrupted_after": args.interrupt_after,
        "eval_every": args.eval_every,
        "exact_latest_state_match": True,
        "exact_best_state_match": True,
        "latest_checkpoint_tensors_compared": compared_tensors,
        "matched_state": [
            "model",
            "optimizer",
            "sampler_rng",
            "rng",
            "identity",
            "progress including best snapshot, history and partial interval",
        ],
        "report_samples_and_history_match": True,
        "identity": full["identity"],
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("train.py"),
                Path(__file__).with_name("checkpoint.py"),
                Path(__file__).with_name("model.py"),
            ]
        },
        "commands": commands,
        "runs": reports,
        "scope": "Graceful process stop at a completed optimizer update, followed by fresh-process resume on the same device/software stack. Does not simulate power loss or establish cross-device reproducibility.",
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: exact latest/best checkpoint state and report samples/history match; {compared_tensors} latest-state tensors compared.",
        flush=True,
    )


if __name__ == "__main__":
    main()
