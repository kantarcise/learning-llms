"""Regenerate the completed November 14 figures from this directory’s saved evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
output = root
run = json.loads((root / "run.json").read_text())
summary = json.loads((root / "monitor-summary.json").read_text())
meta = json.loads((root / "monitor-metadata.json").read_text())
import gzip

samples = [
    json.loads(line)
    for line in gzip.decompress((root / "samples.jsonl.gz").read_bytes())
    .decode()
    .splitlines()
]
start = meta["start_monotonic_ns"]
t = np.array([(row["monotonic_ns"] - start) / 1e9 / 60 for row in samples])


def gpu(key: str) -> np.ndarray:
    return np.array([row["gpu"].get(key, np.nan) for row in samples], dtype=float)


def avg(x: np.ndarray, n: int = 30) -> np.ndarray:
    # Centered rolling mean, with a shorter window at the observed boundaries.
    a = np.ones(n)
    return np.convolve(x, a, "same") / np.convolve(np.ones(len(x)), a, "same")


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
    }
)
fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained", sharex=False)
fig.suptitle(
    "RTX 3090 · completed 10M-token experiment\nContinuation: updates 1,000–500,000 · includes pre-run idle and post-run cooldown",
    fontsize=17,
)
a = axes[0, 0]
a.plot(t, gpu("power_w"), alpha=0.22, color="#2563eb")
a.plot(
    t, avg(gpu("power_w")), color="#2563eb", label="GPU board power · 30-sample mean"
)
a.axhline(420, color="#64748b", ls=":", label="Configured ceiling: 420 W")
a.set(
    title="GPU power · peak 410 W · gross energy 0.670 kWh",
    ylabel="Watts",
    ylim=(0, 450),
)
a.legend(fontsize=8)
a = axes[0, 1]
a.plot(t, avg(gpu("busy_percent")), label="GPU kernel-active time", color="#2563eb")
a.plot(
    t,
    avg(np.array([r["cpu_percent"].get("cpu") or 0 for r in samples])),
    label="Whole host CPU busy",
    color="#d97706",
)
a.set(
    title="Activity · 30-sample means",
    ylabel="Percent of sampling time / CPU capacity",
    ylim=(0, 105),
)
a.legend(fontsize=8)
a = axes[1, 0]
a.plot(t, gpu("temperature_c"), label="GPU", color="#dc2626")
cpu_key = "k10temp/Tctl/c"
a.plot(
    t,
    avg(np.array([r["sensors"].get(cpu_key, np.nan) for r in samples])),
    label="CPU Tctl · 30-sample mean",
    color="#d97706",
)
a.set(title="Temperature · GPU peak 82°C", ylabel="°C")
a.legend(fontsize=8)
a = axes[1, 1]
a.plot(t, gpu("fan_percent"), color="#059669", label="GPU fan")
a.set(title="GPU fan and SM clock", ylabel="GPU fan %", ylim=(0, 105))
b = a.twinx()
b.plot(t, avg(gpu("sm_clock_mhz")), color="#7c3aed", label="SM clock · 30-sample mean")
b.set_ylabel("SM clock MHz")
b.grid(False)
a.legend(loc="upper left", fontsize=8)
b.legend(loc="lower left", fontsize=8)
a = axes[2, 0]
a.plot(t, gpu("memory_used_mib") / 1024, label="GPU-wide VRAM", color="#2563eb")
ram = np.array(
    [
        (r["memory_bytes"]["MemTotal"] - r["memory_bytes"]["MemAvailable"]) / 2**30
        for r in samples
    ]
)
a.plot(t, ram, label="Host RAM unavailable", color="#d97706")
a.set(
    title="Memory · includes other applications",
    ylabel="GiB",
    xlabel="Minutes since monitoring started",
)
a.legend(fontsize=8)
a = axes[2, 1]
keep = t >= t[-1] - 1.5
a.plot(t[keep], gpu("power_w")[keep], color="#2563eb", label="GPU board power")
a.set(
    title="Final 90 seconds · completion and cooldown",
    ylabel="Watts",
    xlabel="Minutes since monitoring started",
)
b = a.twinx()
b.plot(t[keep], gpu("temperature_c")[keep], color="#dc2626", label="GPU temperature")
b.set_ylabel("°C")
b.grid(False)
a.legend(loc="upper left", fontsize=8)
b.legend(loc="upper right", fontsize=8)
# The final panel is a zoom, so it must not share the full-run x limits.
for axis in axes.flat:
    axis.set_xlim(0, t[-1])
axes[2, 1].set_xlim(t[keep][0], t[-1])
fig.savefig(output / "usage.png", dpi=150)
plt.close(fig)

operations = json.loads((root / "operations-summary.json").read_text())
buckets = {row["minute"]: row for row in operations["per_minute_estimated_work"]}
mins = sorted(buckets)
wall_rate = [
    buckets[m]["flops"] / min(60, summary["duration_seconds"] - m * 60) / 1e12
    for m in mins
]
step_rate = [buckets[m]["flops"] / buckets[m]["step_seconds"] / 1e12 for m in mins]
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
fig.suptitle("Where the time went, and estimated model work", fontsize=17)
labels = [
    "Training updates\n(CPU + GPU step wall time)",
    "Checkpoint writes",
    "Validation",
    "Checkpoint reads",
    "Other / idle / cooldown",
]
phase = summary["phase_seconds"]
values = [
    phase["train_step"],
    phase["checkpoint"],
    phase["validation"],
    phase["checkpoint_read"],
]
values.append(summary["duration_seconds"] - sum(values))
axes[0].grid(False)
axes[0].pie(
    values,
    labels=None,
    colors=["#059669", "#dc6b53", "#e5b644", "#b88ccb", "#cbd5e1"],
    autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
    startangle=90,
    wedgeprops={"width": 0.45},
)
axes[0].legend(
    [f"{label}: {value / 60:.1f} min" for label, value in zip(labels, values)],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    fontsize=9,
)
axes[0].set_title("Recorded continuation phases · 114.2 min observed")
axes[1].plot(
    mins, wall_rate, label="Estimated work / observed wall time", color="#2563eb"
)
axes[1].plot(
    mins,
    step_rate,
    label="Estimated work / training-step time",
    color="#059669",
    alpha=0.7,
)
axes[1].set(
    title="Per-minute aggregates · estimates, not hardware counters",
    xlabel="Minutes since monitoring started",
    ylabel="Estimated matmul TFLOP/s",
    ylim=(0, None),
)
axes[1].legend(fontsize=8)
fig.savefig(output / "operations.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
train = run["training_curve"]
val = run["history"]
for a in axes:
    a.plot(
        [r["step"] for r in train],
        [r["loss"] for r in train],
        color="#2563eb",
        alpha=0.35,
        label="Recent 100-update training window",
    )
    a.plot(
        [r["step"] for r in val],
        [r["validation_loss"] for r in val],
        color="#d97706",
        label="All 300 fixed validation windows",
    )
    a.axvline(run["best_step"], color="#059669", ls="--", lw=1)
    a.set(xlabel="Global optimizer update", ylabel="Cross-entropy loss")
axes[0].set_title("Complete trajectory, including resumed calibration")
axes[1].set(
    title="Detail after initial learning · best at 89,000 updates",
    xlim=(10000, 500000),
    ylim=(2, 5.5),
)
axes[1].scatter([89000], [4.030756540104804], color="#059669", zorder=5)
axes[0].legend(fontsize=8)
fig.suptitle(
    "Training kept improving; validation was best well before the final update",
    fontsize=16,
)
fig.savefig(output / "loss.png", dpi=150)
plt.close(fig)
