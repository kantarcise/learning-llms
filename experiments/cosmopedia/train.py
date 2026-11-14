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
from checkpoint import (
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    write_checkpoint,
)
from events import Events, estimated_training_flops
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
    parser.add_argument(
        "--resume", type=Path, help="Resume a versioned latest.pt or best.pt"
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Stop at this absolute update, preserving the planned evaluation schedule",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total target updates, including completed updates",
    )
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
        min(
            args.steps,
            args.batch_size,
            args.eval_every,
            args.validation_windows,
            args.checkpoint_every,
        )
        < 1
        or args.learning_rate <= 0
    ):
        parser.error("Step counts, batch sizes, and learning rate must be positive")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("--stop-after must be between 1 and --steps")
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    events = Events()
    events.emit("setup", "start")
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    # Hash actual artifacts, not only their descriptive manifest.
    identity = {
        "config": model.configuration(),
        "data_sha256": {
            name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
            for name in ["manifest.json", "tokenizer.json", "train.pt", "validation.pt"]
        },
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "eval_every": args.eval_every,
        "validation_indices": validation_indices.tolist(),
        "device_type": device.type,
        "torch_version": str(torch.__version__),
    }
    prompt = "Once upon a time"
    checkpoint = load_checkpoint(args.resume, identity) if args.resume else None
    if checkpoint is not None:
        progress = checkpoint["progress"]
        start_step = progress["step"]
        if args.steps <= start_step or (
            args.stop_after is not None and args.stop_after <= start_step
        ):
            parser.error(
                "The requested stopping update must exceed the checkpoint update"
            )
        restore_checkpoint(checkpoint, model, optimizer, sampling, device)
        history = progress["history"]
        before = progress["before"]
        best_loss, best_step = progress["best_loss"], progress["best_step"]
        best_snapshot = progress.get("best_checkpoint", checkpoint)
        best_model = best_snapshot["model"]
        processed = progress["processed"]
        interval_loss = progress["interval_loss"]
        interval_tokens = progress["interval_tokens"]
    else:
        start_step = 0
        with events.phase("generation", step=0):
            before = generate(model, tokenizer, prompt, 40, manifest["eos_id"], device)
        with events.phase("validation", step=0):
            initial_loss = evaluate(model, vx, vy, args.batch_size, device)
        events.emit("loss", "value", step=0, validation_loss=initial_loss)
        if not math.isfinite(initial_loss):
            raise RuntimeError("Non-finite initial validation loss")
        best_loss, best_step = initial_loss, 0
        best_model = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        history: list[dict[str, int | float | None]] = [
            {"step": 0, "train_batch_loss": None, "validation_loss": initial_loss}
        ]
        processed = 0
        interval_loss = 0.0
        interval_tokens = 0
    args.output.mkdir(parents=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"Device: {device}; parameters: {parameter_count:,}; start update: {start_step}",
        flush=True,
    )
    print(
        f"Validation loss at last evaluation: {history[-1]['validation_loss']:.4f}",
        flush=True,
    )

    def save(path: Path, step: int, include_best: bool = False) -> None:
        """Snapshot a completed update, including an unfinished reporting interval."""
        state = {
            "step": step,
            "history": history,
            "before": before,
            "best_loss": best_loss,
            "best_step": best_step,
            "processed": processed,
            "interval_loss": interval_loss,
            "interval_tokens": interval_tokens,
        }
        if include_best:
            state["best_checkpoint"] = best_snapshot
        with events.phase("checkpoint", step=step, file=path.name):
            save_checkpoint(path, model, optimizer, sampling, device, identity, state)

    # Each new output directory is self-contained, including after resuming.
    if checkpoint is None:
        save(args.output / "best.pt", 0)
        with events.phase("checkpoint_read", step=0):
            best_snapshot = load_checkpoint(args.output / "best.pt", identity)
    else:
        with events.phase("checkpoint", step=start_step, file="best.pt"):
            write_checkpoint(args.output / "best.pt", best_snapshot)
    events.emit("setup", "end", status="ok", parameter_count=parameter_count)
    segment_start_tokens = processed
    stop_step = args.stop_after if args.stop_after is not None else args.steps
    synchronize(device)
    started = time.perf_counter()
    for step in range(start_step + 1, stop_step + 1):
        step_started = time.monotonic_ns()
        events.emit("train_step", "start", step=step)
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
        events.emit(
            "train_step",
            "end",
            step=step,
            status="ok",
            loss=loss.item(),
            non_padding_targets=valid,
            duration_ns=time.monotonic_ns() - step_started,
            estimated_flops=estimated_training_flops(
                model.configuration(), len(x), x.shape[1]
            ),
        )
        if step % args.eval_every == 0 or step == args.steps:
            with events.phase("validation", step=step):
                validation_loss = evaluate(model, vx, vy, args.batch_size, device)
            if not math.isfinite(validation_loss):
                raise RuntimeError("Non-finite validation loss")
            train_loss = interval_loss / interval_tokens
            events.emit(
                "loss",
                "value",
                step=step,
                validation_loss=validation_loss,
                train_loss=train_loss,
            )
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
                best_model = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                improved = True
            else:
                improved = False
            interval_loss, interval_tokens = 0.0, 0
            if improved:
                save(args.output / "best.pt", step)
                with events.phase("checkpoint_read", step=step):
                    best_snapshot = load_checkpoint(args.output / "best.pt", identity)
        if step % args.checkpoint_every == 0 or step == stop_step:
            save(args.output / "latest.pt", step, include_best=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_model)
    with events.phase("checkpoint", step=stop_step, file="best-weights.pt"):
        torch.save(best_model, args.output / "best-weights.pt")
    with events.phase("generation", step=stop_step):
        after = generate(model, tokenizer, prompt, 80, manifest["eos_id"], device)
    report = {
        "config": model.configuration(),
        "parameter_count": parameter_count,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "steps": stop_step,
        "planned_steps": args.steps,
        "segment_start_step": start_step,
        "resumed_from": str(args.resume) if args.resume else None,
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
        "segment_training_target_tokens": processed - segment_start_tokens,
        "loop_seconds_including_evaluation_and_saving": elapsed,
        "training_targets_per_second_including_evaluation_and_saving": (
            processed - segment_start_tokens
        )
        / elapsed,
        "loss_note": "Training is recent sampled batches at changing weights; validation is fixed windows at current weights. Not directly identical measurement procedures.",
        "checkpoint_note": "latest.pt resumes the latest saved update; best.pt resumes the historical validation-best update. Both contain model/optimizer/RNG/progress/data identity. best-weights.pt contains inference-only historical best weights. Timing/throughput cover this invocation only; token totals and history are cumulative.",
    }
    if device.type == "cuda":
        report["cuda"] = {
            "device_name": torch.cuda.get_device_name(device),
            "build_version": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "memory_scope": "PyTorch allocator across setup, training, evaluation and generation; excludes other processes and non-allocator GPU memory.",
        }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Before training:\n{before}\nBest checkpoint (step {best_step}):\n{after}",
        flush=True,
    )
    print(
        f"Segment loop: {elapsed:.1f}s; {(processed - segment_start_tokens) / elapsed:.0f} training target tokens/s including evaluation/saving",
        flush=True,
    )
    print(f"Saved report and checkpoints in {args.output}", flush=True)


if __name__ == "__main__":
    main()
