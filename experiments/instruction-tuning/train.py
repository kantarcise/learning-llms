"""Chapter 7.6, two epochs on CUDA. Invoked by run-3090.sh."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import tiktoken
import torch
from previous_chapters import GPTModel, calc_loss_loader, generate, train_model_simple
from torch.utils.data import DataLoader


def prompt(record: dict[str, str]) -> str:
    sections = [
        (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        ),
        f"### Instruction:\n{record['instruction']}",
    ]
    if record["input"]:
        sections.append(f"### Input:\n{record['input']}")
    return "\n\n".join(sections)


def collate(batch: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CUDA int64 inputs and next-token targets shaped [examples, positions]."""
    width = min(max(map(len, batch)), 1024)
    padded = torch.full((len(batch), width + 1), 50256, dtype=torch.long)
    for row, ids in enumerate(batch):
        kept = ids[: width + 1]
        padded[row, : len(kept)] = torch.tensor(kept)
    targets = padded[:, 1:].clone()
    for row, ids in enumerate(batch):
        targets[row, len(ids) :] = -100
    return padded[:, :-1].cuda(), targets.cuda()


def plot_losses(
    updates: list[int],
    training: list[float],
    validation: list[float],
    batches_per_epoch: int,
    path: Path,
) -> None:
    """Use the Chapter 6 two-axis plot with actual optimizer update counts."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8), layout="constrained")
    ax.plot(updates, training, "o-", color="#18685d", markersize=3, label="Training")
    ax.plot(
        updates, validation, "s--", color="#a94b24", markersize=3, label="Validation"
    )
    ax.set(
        xlabel="Optimizer updates",
        ylabel="Mean next-token cross-entropy",
        ylim=(0, None),
    )
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x / batches_per_epoch, lambda x: x * batches_per_epoch),
    )
    top.set_xlabel("Epochs completed")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def main(cache: Path, run: Path) -> None:
    records = json.loads((cache / "instruction-data.json").read_text())
    assert len(records) == 1100
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = [
        tokenizer.encode(prompt(r) + "\n\n### Response:\n" + r["output"])
        for r in records
    ]
    train = DataLoader(
        encoded[:935], batch_size=8, shuffle=True, drop_last=True, collate_fn=collate
    )
    val = DataLoader(encoded[1045:], batch_size=8, collate_fn=collate)
    config = {
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "vocab_size": 50257,
        "context_length": 1024,
    }
    model = GPTModel(config)
    model.load_state_dict(
        torch.load(cache / "gpt2-medium-355M.pth", mmap=True, weights_only=True)
    )
    model.cuda().eval()
    start_context = prompt(records[1045])
    ids = torch.tensor([tokenizer.encode(start_context)], device="cuda")

    def response() -> str:
        model.eval()
        with torch.no_grad():
            outputs = generate(model, ids, 35, 1024, eos_id=50256)
        return tokenizer.decode(outputs[0, ids.shape[1] :].tolist()).strip()

    before = response()
    torch.manual_seed(123)
    with torch.no_grad():
        initial = [
            calc_loss_loader(loader, model, "cuda", 5) for loader in (train, val)
        ]
    print("Initial training and validation losses:", initial)
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = perf_counter()
    training, validation, tokens = train_model_simple(
        model,
        train,
        val,
        optimizer,
        "cuda",
        num_epochs=2,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
        tokenizer=tokenizer,
    )
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    updates = list(range(1, 2 * len(train) + 1, 5))
    assert len(updates) == len(training) == len(validation) == len(tokens)
    after = response()
    torch.save(model.state_dict(), run / "model.pth")
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "config": config,
        "epochs": 2,
        "optimizer_updates": 2 * len(train),
        "seconds": seconds,
        "peak_allocated_bytes": peak,
        "updates": updates,
        "tokens_seen": tokens,
        "initial_losses": initial,
        "training": training,
        "validation": validation,
        "prompt": start_context,
        "reference": records[1045]["output"],
        "before": before,
        "after": after,
    }
    (run / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_losses(updates, training, validation, len(train), run / "loss.svg")
    print(f"Finished in {seconds / 60:.2f} minutes. Chef response: {after}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
