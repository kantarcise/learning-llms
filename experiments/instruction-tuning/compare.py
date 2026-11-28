"""Run Chapter 7 comparisons with the local §7.6 checkpoint and verified inputs."""

import argparse
import json
import math
from functools import partial
from pathlib import Path
from time import perf_counter

import tiktoken
import torch
from previous_chapters import GPTModel, generate, train_model_simple
from torch import nn
from torch.utils.data import DataLoader
from train import collate, prompt

Record = dict[str, str]
Example = tuple[list[int], int]


def templates(entry: Record) -> dict[str, str]:
    request = "\n".join(filter(None, (entry["instruction"], entry["input"])))
    qwen = f"<|im_start|>user\n{request}<|im_end|>\n<|im_start|>assistant\n"
    return {
        "alpaca": prompt(entry) + "\n\n### Response:\n",
        "phi": f"<|user|>\n{request}<|end|>\n<|assistant|>\n",
        "mistral": f"[INST] {request} [/INST]",
        "gemma2": f"<start_of_turn>user\n{request}<end_of_turn>\n<start_of_turn>model\n",
        "gemma4": f"<|turn>user\n{request}<turn|>\n<|turn>model\n",
        "qwen2": qwen,
        "qwen3": qwen + "<think>\n\n</think>\n\n",
    }


def encode(entry: Record, tokenizer: tiktoken.Encoding, style: str) -> Example:
    prefix = templates(entry)[style]
    ids = tokenizer.encode(prefix + entry["output"])
    prefix_ids = tokenizer.encode(prefix)
    # A BPE token can straddle the text boundary. Score it with the answer.
    boundary = len(prefix_ids)
    while ids[:boundary] != prefix_ids[:boundary]:
        boundary -= 1
    return ids, boundary


def batch_examples(
    batch: list[Example], masked: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int64 CUDA [batch, position] inputs and shifted targets."""
    inputs, targets = collate([ids for ids, _ in batch])
    if masked:
        for row, (_, boundary) in enumerate(batch):
            targets[row, : max(0, boundary - 1)] = -100
    return inputs, targets


class Adapter(nn.Module):
    """Appendix E additive update with rank 16 and direct multiplier 16."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(base.in_features, 16))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.b = nn.Parameter(torch.zeros(16, base.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project float [batch, position, input] to [batch, position, output]."""
        return self.base(x) + 16 * ((x @ self.a) @ self.b)


def add_adapters(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, Adapter(child))
        else:
            add_adapters(child)


@torch.inference_mode()
def reply(model: GPTModel, tokenizer: tiktoken.Encoding, text: str) -> str:
    ids = torch.tensor([tokenizer.encode(text)], device="cuda")
    output = generate(model, ids, 256, 1024, eos_id=50256)
    return tokenizer.decode(output[0, ids.shape[1] :].tolist()).strip()


def save(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main(run: Path, mode: str) -> None:
    cache = run.parent / "inputs"
    records = json.loads((cache / "instruction-data.json").read_text())
    assert len(records) == 1100
    test = records[935:1045]
    tokenizer = tiktoken.get_encoding("gpt2")
    style = "phi" if mode == "phi" else "alpaca"
    encoded = [encode(entry, tokenizer, style) for entry in records]
    config = json.loads((run / "metrics.json").read_text())["config"]
    model = GPTModel(config)
    checkpoint = (
        run / "model.pth" if mode == "baseline" else cache / "gpt2-medium-355M.pth"
    )
    model.load_state_dict(torch.load(checkpoint, mmap=True, weights_only=True))
    if mode == "lora":
        model.requires_grad_(False)
        torch.manual_seed(123)
        add_adapters(model)
    model.cuda()
    result: dict[str, object] = {
        "mode": mode,
        "style": style,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "mean_train_tokens": sum(len(ids) for ids, _ in encoded[:935]) / 935,
    }
    prompt_text = (
        (lambda entry: templates(entry)[style]) if mode in {"phi", "masked"} else prompt
    )
    if mode != "baseline":
        batch = partial(batch_examples, masked=mode == "masked")
        train = DataLoader(
            encoded[:935], batch_size=8, shuffle=True, drop_last=True, collate_fn=batch
        )
        val = DataLoader(encoded[1045:], batch_size=8, collate_fn=batch)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.1)
        torch.manual_seed(123)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = perf_counter()
        training, validation, tokens_seen = train_model_simple(
            model,
            train,
            val,
            optimizer,
            "cuda",
            2,
            5,
            5,
            prompt_text(records[1045]),
            tokenizer,
        )
        torch.cuda.synchronize()
        result["seconds"] = perf_counter() - start
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["training"] = training
        result["validation"] = validation
        result["tokens_seen"] = tokens_seen
        if mode == "lora":
            torch.save(
                {
                    n: p.detach().cpu()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                },
                run / "adapters.pth",
            )
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for inputs, targets in DataLoader(
            encoded[935:1045],
            batch_size=8,
            collate_fn=partial(batch_examples, masked=True),
        ):
            loss_sum += nn.functional.cross_entropy(
                model(inputs).flatten(0, 1), targets.flatten(), reduction="sum"
            ).item()
            count += targets.ne(-100).sum().item()
    result["answer_target_count"] = count
    result["answer_cross_entropy"] = loss_sum / count
    result["chef_response"] = (
        reply(model, tokenizer, prompt_text(records[1045]))
        .removeprefix("### Response:")
        .strip()
    )
    torch.cuda.synchronize()
    gen_start = perf_counter()
    for entry in test:
        entry["model_response"] = (
            reply(model, tokenizer, prompt_text(entry))
            .removeprefix("### Response:")
            .strip()
        )
    torch.cuda.synchronize()
    result["generation_seconds"] = perf_counter() - gen_start
    save(run / f"{mode}-responses.json", test)
    if mode == "baseline":
        result["spelling"] = [
            {
                "instruction": entry["instruction"],
                "input": entry["input"],
                "reference": entry["output"],
                "responses": {
                    name: reply(model, tokenizer, text)
                    for name, text in templates(entry).items()
                },
            }
            for entry in test
            if "spell" in entry["instruction"].lower()
        ]
    save(run / f"{mode}-comparison.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("mode", choices=["baseline", "phi", "masked", "lora"])
    args = parser.parse_args()
    main(args.run, args.mode)
