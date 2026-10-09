"""Inspect first, then train a tiny model on six fictional log sequences."""

import argparse
import json
from pathlib import Path

import torch
from data import CharacterWindows, decode, encode, load_data
from model import TinyTransformer
from torch.nn import functional as F
from torch.utils.data import DataLoader


def loss_for(logits, targets):
    # Cross-entropy receives logits, not probabilities. One target per position.
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for inputs, targets in loader:
        count = targets.numel()
        total_loss += loss_for(model(inputs), targets).item() * count
        total_targets += count
    return total_loss / total_targets


@torch.no_grad()
def generate(model, vocabulary, prompt="job=", new_characters=120):
    model.eval()
    ids = torch.tensor([encode(prompt, vocabulary)], dtype=torch.long)
    for _ in range(new_characters):
        context = ids[:, -model.context_length :]
        next_id = model(context)[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=1)
    return decode(ids[0].tolist(), vocabulary)


def inspect(model, texts, vocabulary):
    # Exactly reproduce first-batch.json, rather than two adjacent windows.
    inputs = torch.tensor([encode(text[:8], vocabulary) for text in texts[:2]])
    targets = torch.tensor([encode(text[1:9], vocabulary) for text in texts[:2]])
    for x, y in zip(inputs, targets, strict=True):
        print(f"Input: {decode(x.tolist(), vocabulary)!r}  IDs: {x.tolist()}")
        print(f"Target: {decode(y.tolist(), vocabulary)!r} IDs: {y.tolist()}")
    model.eval()
    with torch.no_grad():
        embeddings = model.token_embedding(inputs)
        logits = model(inputs)
        print(f"Input/target shapes: {list(inputs.shape)} / {list(targets.shape)}")
        print(f"Embedding shape: {list(embeddings.shape)}")
        print(f"Vocabulary scores shape: {list(logits.shape)}")
        print(f"Untrained batch loss: {loss_for(logits, targets).item():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "artifacts"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    train_texts, validation_texts, vocabulary = load_data()
    config = {
        "vocabulary_size": len(vocabulary),
        "context_length": 32,
        "width": 32,
        "hidden_width": 64,
    }
    model = TinyTransformer(**config)  # CPU first: small, portable, inspectable.
    print(f"Vocabulary: {len(vocabulary)} characters")
    print(f"Learned parameters: {sum(p.numel() for p in model.parameters()):,}")
    inspect(model, train_texts, vocabulary)
    if args.inspect_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best.pt"
    if checkpoint.exists():
        parser.error(
            "Output already has a checkpoint; choose a fresh --output directory."
        )
    train_data = CharacterWindows(train_texts, vocabulary, config["context_length"])
    validation_data = CharacterWindows(
        validation_texts, vocabulary, config["context_length"]
    )
    training = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    train_evaluation = DataLoader(train_data, batch_size=args.batch_size)
    validation = DataLoader(validation_data, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    before = generate(model, vocabulary)
    history = []
    best_validation = float("inf")
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            for inputs, targets in training:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(model(inputs), targets)
                loss.backward()
                optimizer.step()
        # Both curves measure fixed weights over the whole respective split.
        train_loss = evaluate(model, train_evaluation)
        validation_loss = evaluate(model, validation)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:2}: train={train_loss:.4f} validation={validation_loss:.4f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "vocabulary": vocabulary,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                checkpoint,
            )

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    after = generate(model, vocabulary)
    result = {
        "seed": args.seed,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "config": config,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "best_epoch": saved["epoch"],
        "history": history,
        "before": before,
        "after": after,
        "generation": "Greedy, prompt job=, 120 new characters, no end token.",
    }
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for name in ("train_loss", "validation_loss"):
        ax.plot(
            [row["epoch"] for row in history],
            [row[name] for row in history],
            label=name,
        )
    ax.set(
        xlabel="Epoch (0 = untrained)",
        ylabel="Mean next-character cross-entropy",
        title="Six fictional sequences: pipeline demonstration",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "loss.png")
    plt.close(fig)
    print(
        f"\nBefore training:\n{before}\n\nBest checkpoint (epoch {saved['epoch']}):\n{after}"
    )
    print(f"\nSaved run.json, loss.png, and inference-only best.pt in {args.output}")


if __name__ == "__main__":
    main()
