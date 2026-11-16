"""Versioned, atomic training snapshots; tensors load on CPU before restoration."""

from pathlib import Path

import torch
from model import LanguageModel


def save_checkpoint(
    path: Path,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    sampling: torch.Generator,
    device: torch.device,
    identity: dict,
    progress: dict,
) -> None:
    """Save weights, AdamW, sampler, Torch RNGs and loop/evaluation progress.

    CPU RNG and sampler states are uint8 vectors; device RNG is saved only for
    the active accelerator. Python/NumPy RNGs are not used by this trainer.
    Replace the destination only after serialization completes.
    """
    rng = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        rng["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        rng["device"] = torch.mps.get_rng_state()
    write_checkpoint(
        path,
        {
            "format_version": 1,
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler_rng": sampling.get_state(),
            "rng": rng,
            "progress": progress,
        },
    )


def write_checkpoint(path: Path, checkpoint: dict) -> None:
    """Atomically serialize an existing snapshot, including a historical best."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, identity: dict) -> dict:
    """Reject weights-only files or changed data/settings before restoring state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            "Expected a resumable checkpoint; weights-only files cannot resume AdamW/RNG state"
        )
    if checkpoint["identity"] != identity:
        changed = sorted(
            key for key in identity if checkpoint["identity"].get(key) != identity[key]
        )
        raise ValueError(f"Resume identity mismatch: {', '.join(changed)}")
    return checkpoint


def restore_checkpoint(
    checkpoint: dict,
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    sampling: torch.Generator,
    device: torch.device,
) -> None:
    """Restore RNGs last, after constructing/loading model and optimizer."""
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    sampling.set_state(checkpoint["sampler_rng"])
    torch.set_rng_state(checkpoint["rng"]["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(checkpoint["rng"]["device"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(checkpoint["rng"]["device"])
