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
import signal
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
from live import LiveLog
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
        "--session", type=Path, help="Parent session folder for the live dashboard"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Updates per recent training-loss point",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Soft loop-time budget for this invocation, checked between updates",
    )
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
            args.log_every,
        )
        < 1
        or args.learning_rate <= 0
    ):
        parser.error("Step counts, batch sizes, and learning rate must be positive")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("--stop-after must be between 1 and --steps")
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    if args.max_seconds is not None and (
        not math.isfinite(args.max_seconds) or args.max_seconds <= 0
    ):
        parser.error("--max-seconds must be finite and positive")
    if args.session is not None:
        if (
            args.output.resolve() == args.session.resolve()
            or not args.output.resolve().is_relative_to(args.session.resolve())
        ):
            parser.error("--output must be a child of --session")
        args.session.mkdir(parents=True, exist_ok=True)
    requested_stop: str | None = None

    def request_stop(signum: int, frame: object) -> None:
        nonlocal requested_stop
        requested_stop = signal.Signals(signum).name

    for signum in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(signum, request_stop)
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
        if progress.get("log_every", args.log_every) != args.log_every:
            parser.error("Keep --log-every unchanged when resuming")
        training_curve = progress.get(
            "training_curve",
            [
                {"step": row["step"], "loss": row["train_batch_loss"]}
                for row in history
                if row["train_batch_loss"] is not None
            ],
        )
        plot_interval_loss = progress.get("plot_interval_loss", 0.0)
        plot_interval_tokens = progress.get("plot_interval_tokens", 0)
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
        training_curve: list[dict[str, int | float]] = []
        plot_interval_loss = 0.0
        plot_interval_tokens = 0
    args.output.mkdir(parents=True)
    live = LiveLog(args.output, args.session)
    last_saved_step: int | None = start_step if args.resume else None
    last_saved_path = str(args.resume) if args.resume else None
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"Device: {device}; parameters: {parameter_count:,}; start update: {start_step}",
        flush=True,
    )
    print(
        f"Validation loss at last evaluation: {history[-1]['validation_loss']:.4f}",
        flush=True,
    )

    def publish(status: str, step: int, reason: str | None = None) -> None:
        """Expose cumulative curves without changing the sampler or reporting intervals."""
        pending = (
            {"step": step, "loss": plot_interval_loss / plot_interval_tokens}
            if plot_interval_tokens
            else None
        )
        live.update(
            status=status,
            step=step,
            planned_steps=args.steps,
            segment_start_step=start_step,
            history=history,
            training_curve=training_curve,
            pending_training_point=pending,
            best_validation_loss=best_loss,
            best_step=best_step,
            latest_checkpoint_step=last_saved_step,
            latest_checkpoint=last_saved_path,
            training_target_tokens_seen=processed,
            stop_reason=reason,
            log_every=args.log_every,
            data_token_counts=manifest["token_counts"],
        )

    def save(path: Path, step: int, include_best: bool = False) -> None:
        """Snapshot a completed update, including an unfinished reporting interval."""
        nonlocal last_saved_step, last_saved_path
        publish("saving", step)
        state = {
            "step": step,
            "history": history,
            "before": before,
            "best_loss": best_loss,
            "best_step": best_step,
            "processed": processed,
            "interval_loss": interval_loss,
            "interval_tokens": interval_tokens,
            "training_curve": training_curve,
            "plot_interval_loss": plot_interval_loss,
            "plot_interval_tokens": plot_interval_tokens,
            "log_every": args.log_every,
        }
        if include_best:
            state["best_checkpoint"] = best_snapshot
        with events.phase("checkpoint", step=step, file=path.name):
            save_checkpoint(path, model, optimizer, sampling, device, identity, state)
        if path.name == "latest.pt":
            last_saved_step, last_saved_path = step, str(path)
        publish("running", step)

    publish("initializing", start_step)
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
    last_step = start_step
    reason: str | None = None

    def stopping_reason() -> str | None:
        """Check cooperative stop requests without disrupting an in-flight update/save."""
        if requested_stop:
            return requested_stop
        if (args.output / "STOP").exists():
            return "stop file"
        if (
            args.max_seconds is not None
            and time.perf_counter() - started >= args.max_seconds
        ):
            return "time budget"
        return None

    publish("running", start_step)
    for step in range(start_step + 1, stop_step + 1):
        reason = stopping_reason()
        if reason:
            break
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
        plot_interval_loss += loss.item() * valid
        plot_interval_tokens += valid
        last_step = step
        if step % args.log_every == 0:
            training_curve.append(
                {"step": step, "loss": plot_interval_loss / plot_interval_tokens}
            )
            plot_interval_loss, plot_interval_tokens = 0.0, 0
            publish("running", step)
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
        reason = stopping_reason()
        if step % args.checkpoint_every == 0 or step == stop_step or reason:
            save(args.output / "latest.pt", step, include_best=True)
        if reason:
            break
    if last_saved_step != last_step or not (args.output / "latest.pt").exists():
        save(args.output / "latest.pt", last_step, include_best=True)
    stop_step = last_step
    reason = reason or (
        "stop-after budget" if stop_step < args.steps else "target reached"
    )
    publish("stopping" if stop_step < args.steps else "saving", stop_step, reason)
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
        "stop_reason": reason,
        "training_curve": training_curve,
        "pending_training_point": {
            "step": stop_step,
            "loss": plot_interval_loss / plot_interval_tokens,
        }
        if plot_interval_tokens
        else None,
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
    publish("stopped" if stop_step < args.steps else "completed", stop_step, reason)
    live.finish("stopped" if stop_step < args.steps else "completed")
    print(f"Saved report, checkpoints and loss.svg in {args.output}", flush=True)


if __name__ == "__main__":
    main()
