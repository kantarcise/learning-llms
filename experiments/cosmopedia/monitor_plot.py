"""Render saved telemetry without a live GPU; matplotlib is already in experiments."""

import argparse
import base64
import html
import json
import math
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

COLORS = {
    "idle": "#dbe4ee",
    "setup": "#b7c6e0",
    "train_step": "#2e9d78",
    "validation": "#f2bd50",
    "checkpoint": "#d46755",
    "checkpoint_read": "#bd88b5",
    "generation": "#5aa6ce",
    "cooldown": "#c5dce5",
}


def read_jsonl(path: Path) -> list[dict]:
    """Ignore only an incomplete final record left by a killed writer."""
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    result: list[dict] = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or raw.endswith("\n"):
                raise
    return result


def spans(events: list[dict], start: int, end: int) -> list[dict]:
    """Pair phase boundaries, clipping unfinished phases to the observation end."""
    active: dict[tuple, dict] = {}
    result: list[dict] = []
    for event in sorted(events, key=lambda row: row["monotonic_ns"]):
        key = (event["phase"], event.get("step"), event.get("file"))
        if event["kind"] == "start":
            active[key] = event
        elif event["kind"] == "end" and key in active:
            opened = active.pop(key)
            result.append(
                {
                    **opened,
                    "start_s": (opened["monotonic_ns"] - start) / 1e9,
                    "end_s": (event["monotonic_ns"] - start) / 1e9,
                    "status": event.get("status", "ok"),
                }
            )
    for opened in active.values():
        result.append(
            {
                **opened,
                "start_s": (opened["monotonic_ns"] - start) / 1e9,
                "end_s": (end - start) / 1e9,
                "status": "unfinished",
            }
        )
    return result


def board_energy(samples: list[dict]) -> float | None:
    """Trapezoidal gross GPU board energy over adjacent valid samples, in joules."""
    energy = 0.0
    valid = False
    for left, right in pairwise(samples):
        a, b = left.get("gpu", {}).get("power_w"), right.get("gpu", {}).get("power_w")
        if a is not None and b is not None:
            valid = True
            energy += (a + b) / 2 * (right["monotonic_ns"] - left["monotonic_ns"]) / 1e9
    return energy if valid else None


def draw_phases(axes: list[plt.Axes], intervals: list[dict]) -> None:
    """Batch phase markers so artist count stays bounded for long runs."""
    phase_names = list(COLORS)
    # One collection per phase, rather than one artist per optimizer update.
    # Rasterize dense collections so long-run SVGs remain practical to open.
    for phase in phase_names:
        selected = [item for item in intervals if item["phase"] == phase]
        axes[0].broken_barh(
            [
                (item["start_s"], max(0.001, item["end_s"] - item["start_s"]))
                for item in selected
            ],
            (phase_names.index(phase) - 0.35, 0.7),
            facecolors=COLORS[phase],
            rasterized=True,
        )
        if phase in ["checkpoint", "checkpoint_read", "cooldown"]:
            vertices = [
                [
                    (item["start_s"], 0),
                    (item["start_s"], 1),
                    (item["end_s"], 1),
                    (item["end_s"], 0),
                ]
                for item in selected
            ]
            for axis in axes[1:]:
                axis.add_collection(
                    PolyCollection(
                        vertices,
                        transform=axis.get_xaxis_transform(),
                        facecolors=COLORS[phase],
                        alpha=0.09,
                        linewidth=0,
                        rasterized=True,
                    )
                )
    axes[0].set_ylim(-0.5, len(phase_names) - 0.5)


def render(directory: Path) -> None:
    """Build PNG/SVG, a portable HTML report and machine-readable summary."""
    metadata = json.loads((directory / "metadata.json").read_text())
    samples = read_jsonl(directory / "samples.jsonl")
    events = read_jsonl(directory / "lifecycle.jsonl") + read_jsonl(
        directory / "events.jsonl"
    )
    start = metadata["start_monotonic_ns"]
    end = max([start, *[row["monotonic_ns"] for row in samples + events]])
    intervals = spans(events, start, end)
    t = [(row["monotonic_ns"] - start) / 1e9 for row in samples]
    fig, axes = plt.subplots(
        11,
        1,
        figsize=(15, 25),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.3, 1, 1, 1.3, 1, 1, 1, 1, 1, 1, 1]},
    )
    fig.suptitle(
        "Training observation · idle → training/checkpoints → cooldown", fontsize=18
    )

    def line(axis: plt.Axes, values: list, label: str, **kwargs: object) -> None:
        if any(value is not None and math.isfinite(value) for value in values):
            axis.plot(
                t,
                [np.nan if value is None else value for value in values],
                label=label,
                linewidth=1.2,
                **kwargs,
            )

    def gpu(key: str) -> list:
        return [row.get("gpu", {}).get(key) for row in samples]

    phase_names = list(COLORS)
    draw_phases(list(axes), intervals)
    axes[0].set_yticks(range(len(phase_names)), phase_names)
    axes[0].set_title(
        "Phase boundaries from the trainer; command lifetime includes interpreter startup and teardown",
        loc="left",
    )
    line(axes[1], gpu("busy_percent"), "GPU kernel-active time (%)")
    line(axes[1], gpu("memory_busy_percent"), "GPU memory read/write-active time (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("GPU activity %")
    cores = sorted(
        {key for row in samples for key in row["cpu_percent"] if key != "cpu"},
        key=lambda key: int(key[3:]),
    )
    line(
        axes[2],
        [row["cpu_percent"].get("cpu") for row in samples],
        "Whole host CPU busy",
    )
    line(
        axes[2],
        [
            row["process"].get("cpu_percent_one_core", 0) / max(1, len(cores))
            if row["process"].get("cpu_percent_one_core") is not None
            else None
            for row in samples
        ],
        "Live command tree (% machine capacity)",
    )
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("CPU busy %")
    if samples and cores:
        values = np.array(
            [
                [
                    row["cpu_percent"].get(core)
                    if row["cpu_percent"].get(core) is not None
                    else np.nan
                    for row in samples
                ]
                for core in cores
            ]
        )
        edges = [*t, t[-1] + metadata["interval_seconds"]]
        axes[3].pcolormesh(
            edges,
            np.arange(len(cores) + 1),
            values,
            vmin=0,
            vmax=100,
            cmap="YlGnBu",
            shading="flat",
            rasterized=True,
        )
        axes[3].set_yticks(np.arange(len(cores)) + 0.5, cores, fontsize=7)
    axes[3].set_ylabel("Logical CPUs\n0–100% busy")
    line(axes[4], gpu("power_w"), "GPU board power (W)")
    line(axes[4], gpu("power_limit_w"), "GPU configured ceiling (W)", linestyle=":")
    packages = {key for row in samples for key in row.get("cpu_package_power_w", {})}
    for key in sorted(packages):
        line(
            axes[4],
            [row.get("cpu_package_power_w", {}).get(key) for row in samples],
            f"CPU {key} (W)",
        )
    axes[4].set_ylabel("Power W")
    line(axes[5], gpu("temperature_c"), "GPU")
    sensor_keys = sorted({key for row in samples for key in row["sensors"]})
    for key in sensor_keys:
        if key.endswith("/c") and (
            "k10temp/" in key or "coretemp/" in key or "CPU Temperature" in key
        ):
            line(
                axes[5],
                [row["sensors"].get(key) for row in samples],
                key.removesuffix("/c"),
            )
    axes[5].set_ylabel("Temperature °C")
    line(axes[6], gpu("fan_percent"), "GPU fan (%)")
    axes[6].set_ylim(0, 105)
    axes[6].set_ylabel("GPU fan %")
    fan_axis = axes[6].twinx()
    for key in sensor_keys:
        if key.endswith("/rpm") and any(
            row["sensors"].get(key, 0) > 0 for row in samples
        ):
            line(
                fan_axis,
                [row["sensors"].get(key) for row in samples],
                key.removesuffix("/rpm"),
                linestyle="--",
            )
    fan_axis.set_ylabel("Host fans/pumps RPM")
    line(
        axes[7],
        [v / 1024 if v is not None else None for v in gpu("memory_used_mib")],
        "Device VRAM in use",
    )
    line(
        axes[7],
        [
            (row["memory_bytes"]["MemTotal"] - row["memory_bytes"]["MemAvailable"])
            / 2**30
            for row in samples
        ],
        "Host RAM unavailable",
    )
    line(
        axes[7],
        [
            row["process"].get("summed_rss_bytes", 0) / 2**30
            if row["process"]
            else None
            for row in samples
        ],
        "Live process-tree summed RSS",
    )
    axes[7].set_ylabel("Memory GiB")
    line(axes[8], gpu("sm_clock_mhz"), "GPU SM clock")
    line(axes[8], gpu("memory_clock_mhz"), "GPU memory clock")
    axes[8].set_ylabel("Clock MHz")
    losses = [row for row in events if row["phase"] == "loss"]
    for key, label in [
        ("train_loss", "Recent training"),
        ("validation_loss", "Fixed-window validation"),
    ]:
        points = [row for row in losses if key in row]
        if points:
            axes[9].plot(
                [(row["monotonic_ns"] - start) / 1e9 for row in points],
                [row[key] for row in points],
                "o-",
                label=label,
            )
    axes[9].set_ylabel("Loss")
    updates = [
        row
        for row in events
        if row["phase"] == "train_step"
        and row["kind"] == "end"
        and row.get("duration_ns", 0) > 0
    ]
    if updates:
        x = [(row["monotonic_ns"] - start) / 1e9 for row in updates]
        axes[10].scatter(
            x,
            [row["estimated_flops"] / row["duration_ns"] / 1000 for row in updates],
            label="Estimated matmul training TFLOP/s",
            s=8,
            rasterized=True,
        )
        tokens_axis = axes[10].twinx()
        tokens_axis.scatter(
            x,
            [row["non_padding_targets"] * 1e9 / row["duration_ns"] for row in updates],
            color="#9f5c15",
            alpha=0.5,
            label="Non-padding targets/s (step)",
            s=7,
            rasterized=True,
        )
        tokens_axis.set_ylabel("Targets/s")
        tokens_axis.legend(loc="upper right", fontsize=7)
    axes[10].set_ylabel("Estimated TFLOP/s")
    axes[10].set_xlabel("Seconds since monitoring started · actual sample timestamps")
    for axis in [*axes, fan_axis]:
        axis.grid(alpha=0.15)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                loc="upper right" if axis is fan_axis else "upper left",
                fontsize=7,
            )
    axes[0].legend(
        handles=[
            Patch(color=COLORS[k], label=k)
            for k in ["checkpoint", "checkpoint_read", "cooldown"]
        ],
        loc="upper right",
        fontsize=7,
    )
    axes[-1].set_xlim(0, max(1, (end - start) / 1e9))
    fig.savefig(directory / "timeline.png", dpi=140)
    fig.savefig(directory / "timeline.svg")
    plt.close(fig)
    errors = sorted(
        {error for row in samples for error in row["errors"]}
        | {row["gpu"]["error"] for row in samples if "error" in row["gpu"]}
    )
    phase_seconds = {
        phase: sum(
            item["end_s"] - item["start_s"]
            for item in intervals
            if item["phase"] == phase
        )
        for phase in phase_names
    }
    summary = {
        "samples": len(samples),
        "duration_seconds": (end - start) / 1e9,
        "phase_seconds": phase_seconds,
        "gross_sampled_gpu_energy_j": board_energy(samples),
        "max_gpu_power_w": max(
            (v for v in gpu("power_w") if v is not None), default=None
        ),
        "max_gpu_temperature_c": max(
            (v for v in gpu("temperature_c") if v is not None), default=None
        ),
        "max_collection_seconds": max(
            (row["collection_seconds"] for row in samples), default=None
        ),
        "errors": errors,
        "notes": [
            "Phase durations can overlap (setup includes initial validation/generation/checkpoint).",
            "GPU energy is approximate gross device energy over sampled intervals, includes idle/cooldown/desktop, and is not whole-machine wall power.",
            "GPU activity percent is time with kernels active, not percent of cores occupied or percent of peak FLOPs.",
            "FLOPs are an algorithmic estimate: 3x forward matmul work; full-square attention; excludes embeddings/norms/activations/softmax/loss/optimizer. No hardware FLOP or occupancy counters collected.",
            "Short phases may fall between sensor samples; driver metrics have their own averaging windows. Cooldown is fixed duration, not a guarantee of thermal equilibrium.",
        ],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    image = base64.b64encode((directory / "timeline.png").read_bytes()).decode()
    rows = "".join(
        f"<tr><td>{html.escape(item['phase'])}</td><td>{item.get('step', '')}</td><td>{html.escape(str(item.get('file', '')))}</td><td>{item['start_s']:.3f}</td><td>{item['end_s'] - item['start_s']:.3f}</td><td>{item['status']}</td></tr>"
        for item in intervals
        if item["phase"] != "train_step"
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in summary["notes"])
    energy_label = (
        f"approximately {summary['gross_sampled_gpu_energy_j']:.1f} J"
        if summary["gross_sampled_gpu_energy_j"] is not None
        else "unavailable"
    )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Training observation</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#1e293b}}img{{width:100%}}pre{{white-space:pre-wrap;background:#f1f5f9;padding:16px}}td,th{{text-align:left;padding:6px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse;width:100%}}</style>
<h1>Training observation</h1><p>{html.escape(metadata["started_at_utc"])} · command exit {metadata.get("command_exit_code")} · {len(samples)} samples</p>
<pre>{html.escape(" ".join(metadata["command"]))}</pre><p>GPU energy across the observation: {energy_label}. CPU package power is shown only when readable.</p>
<ul>{notes}</ul><img alt="Aligned hardware metrics, training phases and losses" src="data:image/png;base64,{image}">
<details><summary>Phase durations and checkpoint events</summary><table><tr><th>Phase</th><th>Step</th><th>File</th><th>Start s</th><th>Duration s</th><th>Status</th></tr>{rows}</table></details>
<details><summary>Unavailable sensors and collection errors</summary><pre>{html.escape(chr(10).join(errors) or "None")}</pre></details></html>"""
    (directory / "report.html").write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
