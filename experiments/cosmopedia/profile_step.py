"""Optional intrusive PyTorch operator/kernel trace for one synthetic training step.

Separate from hardware monitoring: trace collection adds overhead. Operator FLOP
counts cover supported operators only, not measured hardware instructions.
"""

import argparse
import json
from pathlib import Path

import torch
from events import estimated_training_flops
from model import Config, LanguageModel
from torch.profiler import ProfilerActivity, profile, record_function
from train import loss_for, synchronize


def main() -> None:
    """Warm up the existing full architecture, then trace one synthetic update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    device = torch.device(args.device)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    config = Config(vocabulary_size=49152)
    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    inputs = torch.randint(49152, (4, 256), device=device)
    targets = torch.randint(49152, (4, 256), device=device)

    def update() -> None:
        optimizer.zero_grad(set_to_none=True)
        with record_function("forward_and_loss"):
            loss = loss_for(model(inputs), targets)
        with record_function("backward"):
            loss.backward()
        with record_function("clip_and_adamw"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    for _ in range(3):
        update()
    synchronize(device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    args.output.mkdir(parents=True)
    with profile(
        activities=activities, record_shapes=True, with_flops=True
    ) as profiler:
        update()
        synchronize(device)
    profiler.export_chrome_trace(str(args.output / "trace.json"))
    averages = profiler.key_averages()
    rows = [
        {
            "operator": item.key,
            "calls": item.count,
            "self_cpu_us": item.self_cpu_time_total,
            "self_device_us": item.self_device_time_total,
            "supported_operator_flops": item.flops or 0,
        }
        for item in averages
    ]
    (args.output / "operators.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "operators.txt").write_text(
        averages.table(
            sort_by="self_cuda_time_total"
            if device.type == "cuda"
            else "self_cpu_time_total",
            row_limit=40,
        )
    )
    summary = {
        "device": str(device),
        "torch": str(torch.__version__),
        "config": model.configuration(),
        "batch": 4,
        "context": 256,
        "warmup_updates": 3,
        "profiled_updates": 1,
        "synthetic_inputs": True,
        "supported_operator_flops": sum(
            row["supported_operator_flops"] for row in rows
        ),
        "estimated_training_matmul_flops": estimated_training_flops(
            model.configuration(), 4, 256
        ),
        "notes": [
            "Not a training baseline or hardware FLOP-counter measurement.",
            "PyTorch FLOPs use formulas for supported operators; fused attention/backward coverage may be incomplete.",
            "Trace overhead perturbs timing. Use the CUDA kernel track to inspect execution, not operator FLOPs as an occupancy measurement.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Operator table: {args.output / 'operators.txt'}\nChrome/Perfetto trace: {args.output / 'trace.json'}"
    )


if __name__ == "__main__":
    main()
