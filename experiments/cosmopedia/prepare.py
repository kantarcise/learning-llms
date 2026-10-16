"""Stream a bounded Cosmopedia v2 subset and tokenize with the SmolLM tokenizer.

Split documents by a stable text hash BEFORE making windows. Exact duplicate
texts are removed across both splits. No window crosses a document boundary.
Raw text/tokenized data stay under ignored artifacts/. Revisions and selected
text hashes are saved for provenance. This is a small sample, not the full corpus.
"""

import argparse
import hashlib
import json
from collections.abc import Generator
from contextlib import closing
from pathlib import Path

import torch
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from pyarrow import parquet
from tokenizers import Tokenizer

DATASET = "HuggingFaceTB/smollm-corpus"
TOKENIZER = "HuggingFaceTB/SmolLM-135M"


def documents(revision: str) -> Generator[str, None, None]:
    """Read only text columns via remote byte ranges, with explicit cleanup.

    Synchronous Arrow batches avoid background-reader shutdown hangs when a
    bounded sample stops midway through a shard. Shards/rows retain source order.
    """
    files = sorted(
        name
        for name in HfApi().list_repo_files(
            DATASET, repo_type="dataset", revision=revision
        )
        if name.startswith("cosmopedia-v2/") and name.endswith(".parquet")
    )
    filesystem = HfFileSystem()
    for name in files:
        with filesystem.open(f"datasets/{DATASET}@{revision}/{name}", "rb") as source:
            reader = parquet.ParquetFile(source, pre_buffer=False)
            try:
                for batch in reader.iter_batches(
                    batch_size=100, columns=["text"], use_threads=False
                ):
                    yield from batch.column(0).to_pylist()
            finally:
                reader.close()


def windows(
    ids: list[int], context: int, eos: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return padded input/target rows [N,T] for ONE document including EOS.

    N=windows in this document, T=context length. Each target is shifted one
    token ahead. -100 marks target padding ignored by cross-entropy. Padding is
    on the right, so causal attention cannot expose it to real earlier tokens.
    """
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(ids) - 1, context):
        chunk = ids[start : start + context + 1]
        valid = len(chunk) - 1
        inputs.append(chunk[:-1] + [eos] * (context - valid))
        targets.append(chunk[1:] + [-100] * (context - valid))
    return inputs, targets


def main() -> None:
    """Download only enough documents for bounded token budgets and save tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "artifacts/data"
    )
    parser.add_argument("--train-tokens", type=int, default=1_000_000)
    parser.add_argument("--validation-tokens", type=int, default=65_536)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    args = parser.parse_args()
    if min(args.train_tokens, args.validation_tokens, args.context) < 1:
        parser.error("Budgets and context must be positive")
    if args.output.exists():
        parser.error("Choose a fresh output directory (existing data is preserved)")
    api = HfApi()
    dataset_revision = args.dataset_revision or api.dataset_info(DATASET).sha
    tokenizer_revision = args.tokenizer_revision or api.model_info(TOKENIZER).sha
    tokenizer = Tokenizer.from_file(
        hf_hub_download(TOKENIZER, "tokenizer.json", revision=tokenizer_revision)
    )
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        raise ValueError("Expected SmolLM end-of-text token")
    budgets = {"train": args.train_tokens, "validation": args.validation_tokens}
    counts = {"train": 0, "validation": 0}
    rows: dict[str, list[list[int]]] = {"train": [], "validation": []}
    labels: dict[str, list[list[int]]] = {"train": [], "validation": []}
    hashes: dict[str, list[str]] = {"train": [], "validation": []}
    seen: set[str] = set()
    args.output.mkdir(parents=True)
    tokenizer.save(str(args.output / "tokenizer.json"))
    print(
        f"Dataset revision: {dataset_revision}\nTokenizer revision: {tokenizer_revision}",
        flush=True,
    )
    with (
        closing(documents(dataset_revision)) as stream,
        (args.output / "documents.jsonl").open("w") as output,
    ):
        for document in stream:
            text = document.strip()
            digest = hashlib.sha256(text.encode()).hexdigest()
            if not text or digest in seen:
                continue
            seen.add(digest)
            split = "validation" if int(digest[:8], 16) % 10 == 0 else "train"
            remaining = budgets[split] - counts[split]
            if remaining <= 0:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            # Truncate only the last selected document to respect the token cap.
            # Reserve one token for an explicit document end marker.
            ids = ids[: max(0, remaining - 1)] + [eos]
            x, y = windows(ids, args.context, eos)
            rows[split].extend(x)
            labels[split].extend(y)
            hashes[split].append(digest)
            counts[split] += len(ids)
            output.write(
                json.dumps(
                    {
                        "split": split,
                        "sha256": digest,
                        "text": text,
                        "used_tokens": len(ids),
                    }
                )
                + "\n"
            )
            if sum(len(v) for v in hashes.values()) % 100 == 0:
                print(f"Token counts: {counts}", flush=True)
            if all(counts[k] >= budgets[k] for k in counts):
                break
    if not all(rows.values()) or any(counts[k] < budgets[k] for k in counts):
        raise RuntimeError("Stream ended before requested budgets were collected")
    for split in counts:
        torch.save(
            {
                "inputs": torch.tensor(rows[split], dtype=torch.long),
                "targets": torch.tensor(labels[split], dtype=torch.long),
            },
            args.output / f"{split}.pt",
        )
    metadata = {
        "dataset": DATASET,
        "subset": "cosmopedia-v2",
        "dataset_revision": dataset_revision,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": tokenizer_revision,
        "vocabulary_size": tokenizer.get_vocab_size(),
        "eos_id": eos,
        "context_length": args.context,
        "token_counts": counts,
        "window_counts": {k: len(rows[k]) for k in rows},
        "document_hashes": hashes,
        "sampling": "Stream order; SHA256(text) prefix modulo 10 assigns validation; exact text deduplication",
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {counts} tokens in {args.output}", flush=True)


if __name__ == "__main__":
    main()
