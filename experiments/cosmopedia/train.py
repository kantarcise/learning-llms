"""Run a bounded training experiment on the prepared Cosmopedia sample.

A step samples a batch of document windows WITH replacement. Steps are optimizer
updates, not epochs. Validation uses the same fixed sampled windows every time;
it is a cheap diagnostic, not a full-corpus benchmark. Save weights locally and
record provenance, measured speed, losses, and best-checkpoint text in run.json.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from model import Config, LanguageModel
from tokenizers import Tokenizer
from torch.nn import functional as F


def loss_for(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token loss: float scores [B,T,V], int64 targets [B,T].

    B=batch windows, T=positions, V=vocabulary size. Conceptually ask how much
    probability went to the correct token, then take its negative natural log.
    PyTorch applies log-softmax internally. -100 padding targets are excluded
    from the average; other targets are token IDs, never probabilities.
    """
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Measure fixed validation windows without gradients or weight updates.

    inputs/targets are int64 [N,T]: N=selected windows, T=context positions.
    Batch scores have [B,T,V]: B=batch windows, V=vocabulary entries.
    Average over non-padding targets, not equally over potentially uneven batches.
    """
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(inputs), batch_size):
        x = inputs[start : start + batch_size].to(device)
        y = targets[start : start + batch_size].to(device)
        valid = int((y != -100).sum().item())
        total += loss_for(model(x), y).item() * valid
        count += valid
    return total / count


@torch.no_grad()
def generate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    new_tokens: int,
    eos: int,
    device: torch.device,
) -> str:
    """Greedy continuation from one prompt, stopping at EOS or the token budget.

    IDs [1,T] -> scores [1,T,V] -> final scores [1,V] -> one selected token ID.
    T=context positions, V=vocabulary size. Crop long contexts and reset position
    indices, as in the earlier toy model. This is not a KV-cache implementation.
    """
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        raise ValueError("Prompt must encode to at least one token")
    for _ in range(new_tokens):
        x = torch.tensor([ids[-model.config.context_length :]], device=device)
        next_id = int(model(x)[0, -1].argmax().item())
        ids.append(next_id)
        if next_id == eos:
            break
    return tokenizer.decode(ids)


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous device work before reporting elapsed wall time."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    """Train from random weights; keep data/checkpoints under ignored artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "artifacts/data"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--validation-windows", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    args = parser.parse_args()
    if (
        min(args.steps, args.batch_size, args.eval_every, args.validation_windows) < 1
        or args.learning_rate <= 0
    ):
        parser.error("Step counts, batch sizes, and learning rate must be positive")
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    device_name = args.device
    if device_name == "auto":
        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    manifest_bytes = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    training = torch.load(args.data / "train.pt", weights_only=True)
    validation = torch.load(args.data / "validation.pt", weights_only=True)
    tokenizer = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    config = Config(
        vocabulary_size=manifest["vocabulary_size"],
        context_length=manifest["context_length"],
    )
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    # Separate generators keep evaluation choices from changing training sampling.
    sampling = torch.Generator().manual_seed(args.seed)
    validation_indices = torch.randperm(
        len(validation["inputs"]),
        generator=torch.Generator().manual_seed(args.seed + 1),
    )[: args.validation_windows]
    vx, vy = (
        validation["inputs"][validation_indices],
        validation["targets"][validation_indices],
    )
    args.output.mkdir(parents=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"Device: {device}; parameters: {parameter_count:,}; vocabulary: {config.vocabulary_size:,}",
        flush=True,
    )
    print(
        f"Architecture: {config.layers} blocks, width {config.width}, {config.heads} heads, context {config.context_length}",
        flush=True,
    )
    print(
        f"Training windows: {len(training['inputs'])}; fixed validation windows: {len(vx)}",
        flush=True,
    )
    prompt = "Once upon a time"
    before = generate(model, tokenizer, prompt, 40, manifest["eos_id"], device)
    initial_loss = evaluate(model, vx, vy, args.batch_size, device)
    print(f"Step 0: validation={initial_loss:.4f}", flush=True)
    best_loss, best_step = initial_loss, 0
    torch.save(model.state_dict(), args.output / "best.pt")
    history: list[dict[str, int | float | None]] = [
        {"step": 0, "train_batch_loss": None, "validation_loss": initial_loss}
    ]
    processed = 0
    interval_loss = 0.0
    interval_tokens = 0
    synchronize(device)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        selected = torch.randint(
            len(training["inputs"]), (args.batch_size,), generator=sampling
        )
        x = training["inputs"][selected].to(device)
        y = training["targets"][selected].to(device)
        # Fresh gradients each step; learned weights and AdamW state persist.
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss; stop instead of saving misleading results"
            )
        loss.backward()
        # Limit total gradient norm before the update; useful for stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        valid = int((y != -100).sum().item())
        interval_loss += loss.item() * valid
        interval_tokens += valid
        processed += valid
        if step % args.eval_every == 0 or step == args.steps:
            validation_loss = evaluate(model, vx, vy, args.batch_size, device)
            if not math.isfinite(validation_loss):
                raise RuntimeError("Non-finite validation loss")
            train_loss = interval_loss / interval_tokens
            history.append(
                {
                    "step": step,
                    "train_batch_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            print(
                f"Step {step}: recent training={train_loss:.4f} validation={validation_loss:.4f}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                torch.save(model.state_dict(), args.output / "best.pt")
            interval_loss, interval_tokens = 0.0, 0
    synchronize(device)
    elapsed = time.perf_counter() - started
    model.load_state_dict(
        torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    )
    after = generate(model, tokenizer, prompt, 80, manifest["eos_id"], device)
    report = {
        "config": model.configuration(),
        "parameter_count": parameter_count,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_revision": manifest["dataset_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "data_token_counts": manifest["token_counts"],
        "validation_indices": validation_indices.tolist(),
        "validation_targets": int((vy != -100).sum()),
        "history": history,
        "best_step": best_step,
        "before": before,
        "after": after,
        "generation": "Greedy; prompt Once upon a time; 40 new tokens before, 80 after; EOS can stop early",
        "training_target_tokens_seen": processed,
        "loop_seconds_including_evaluation_and_saving": elapsed,
        "training_targets_per_second_including_evaluation_and_saving": processed
        / elapsed,
        "loss_note": "Training is recent sampled batches at changing weights; validation is fixed windows at current weights. Not directly identical measurement procedures.",
        "checkpoint_note": "best.pt is an inference-only state_dict; architecture and tokenizer provenance in report/data manifest. No optimizer state for resuming.",
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Before training:\n{before}\nBest checkpoint (step {best_step}):\n{after}",
        flush=True,
    )
    print(
        f"Loop: {elapsed:.1f}s; {processed / elapsed:.0f} training target tokens/s including evaluation/saving",
        flush=True,
    )
    print(f"Saved report and inference-only checkpoint in {args.output}", flush=True)


if __name__ == "__main__":
    main()
