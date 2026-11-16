# Observe a training run

The monitor records a Linux host and one NVIDIA GPU on the same monotonic timeline as the Cosmopedia trainer. It starts before launching the command, keeps sampling through training and checkpoint saves, then observes a fixed cooldown window after the command exits. It does not change power limits, clocks or fans.

This uses **existing experiment dependencies**: Python's standard library, PyTorch and matplotlib. Hardware readings come from `nvidia-smi`, `/proc` and readable `/sys/class/hwmon`/powercap files. No daemon, dashboard service, account, extra package or root access is required. GPU access must be available in the shell running the command.

## Run it

From the repository root, choose two fresh output directories:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/monitor.py \
  --output experiments/cosmopedia/artifacts/observation-01 \
  --gpu 0 --interval 0.5 --pre-seconds 10 --post-seconds 30 \
  -- .venv/bin/python experiments/cosmopedia/train.py \
  --device cuda --steps 100 --eval-every 25 --checkpoint-every 25 \
  --output experiments/cosmopedia/artifacts/observed-training-01
```

Open `observation-01/report.html` afterward. It is a standalone report with the chart embedded, phase durations, and missing-sensor diagnostics. `timeline.png` and `timeline.svg` can be shared separately. JSONL files are flushed as sampling progresses; `command.log` holds trainer output. Reports are generated on success, command failure and a normal Ctrl-C, with the command's exit status preserved. Ctrl-C terminates the launched process group; resume training from its last successfully saved checkpoint. A force-killed monitor cannot render a report, but completed raw records remain usable.

The wrapper can observe other commands too; the detailed phase markers require a cooperating trainer. It sets `COSMOPEDIA_EVENTS` to an event-file path. The Cosmopedia trainer emits setup, training-step, validation, checkpoint-write/read, generation and loss markers without using RNGs. `--gpu` selects the observed physical GPU by index/UUID; it does not change the command's device selection. On a multi-GPU host, align that selection with the training environment yourself.

## What the chart means

| Track | Measurement and limits |
| --- | --- |
| Phase timeline | Actual trainer markers, including individual updates and named checkpoint files. Setup contains nested validation/generation/save work, so phase totals overlap. |
| GPU activity | Driver-reported fraction of a sampling period with kernels active, plus memory read/write-active time. Neither is a count of occupied cores or percentage of peak FLOPs. |
| CPU | Aggregate and per-logical-CPU busy time from `/proc/stat`; idle and I/O-wait are excluded from busy time. Process-tree CPU is also recorded (100% = one logical CPU) and plotted as a fraction of machine capacity. Short-lived children between samples can be missed. |
| Power | GPU board watts and configured power ceiling. CPU package watts are derived from readable energy counters, with wraparound handled. Permission-restricted/missing values remain unavailable. This is not whole-PC wall power. |
| Temperatures and fans | GPU temperature/fan percentage, CPU temperature and exposed host fan/pump RPM. Raw data retains all sensor labels, including potentially unconnected or implausible motherboard readings. |
| Memory | GPU-wide VRAM use, host RAM unavailable, and summed live-process-tree RSS. Shared pages can be counted more than once in the tree sum; GPU-wide metrics include the desktop and other applications. |
| Clocks | Driver-reported SM and memory clock frequencies, in MHz. |
| Loss | Recent training interval loss and fixed-window validation loss, located at evaluation time. Their measurement procedures differ. |
| Step throughput | Non-padding targets/sec and an explicitly approximate matmul TFLOP/s estimate over each completed step's wall time. Points are not connected across checkpoint pauses. |

Each hardware row records start time, UTC time and collection duration. Sensors are queried sequentially, not simultaneously. Requested sampling intervals are a target; use recorded timestamps when queries take longer. Driver metrics have their own averaging windows, so shorter phases can be missed even at 0.5-second sampling. The recorder adds some overhead; compare unmonitored runs separately for performance claims. A fixed cooldown window shows the recovery trajectory but does not establish thermal equilibrium.

The energy summary trapezoidally integrates adjacent valid GPU power samples. It is **gross sampled GPU-board energy across the whole observation**, including idle, startup, checkpoint pauses, cooldown and other GPU applications—not training-only energy or electricity measured at the wall. Gaps with missing power are excluded.

Raw output:

- `metadata.json`: exact command, requested observation settings and exit status.
- `samples.jsonl`: actual timestamps, GPU, CPU/per-core, RAM, live process tree, sensor readings and sensor errors.
- `events.jsonl`: cooperating trainer phases, losses and per-step counters.
- `lifecycle.jsonl`: idle, command and cooldown boundaries.
- `summary.json`: phase durations, sampled power/temperature peaks, gross sampled board energy and limitations.

To regenerate graphs without rerunning training or accessing a GPU:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/monitor_plot.py \
  experiments/cosmopedia/artifacts/observation-01
```

## FLOPs and deeper kernel profiling

The step estimate counts a multiply-add as two FLOPs. Forward matmul work is approximated by:

```text
2 B T [L (4 D² + 2 D H) + D V] + 4 B L T² D
```

Here B is batch size, T context length, L blocks, D model width, H feedforward width and V vocabulary size. Training is approximated as three times forward work. For batch 4 and context 256, this model yields approximately **99.86 billion estimated matmul FLOPs per update**. Full-square attention can overestimate real causal/fused work. Embeddings, normalization, GELU, softmax, loss and optimizer operations are excluded. Padding still incurs compute. Dividing this estimate by synchronized step wall time produces an algorithmic throughput estimate, not a hardware instruction count, SM occupancy measurement or model FLOP utilization percentage.

For an optional, more intrusive operator/kernel trace:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/profile_step.py \
  --device cuda --output experiments/cosmopedia/artifacts/profile-01
```

This uses the same full architecture with synthetic tokens (batch 4/context 256), warms up three optimizer updates and profiles one. It writes `operators.txt`, `operators.json`, `summary.json` and a Chrome/Perfetto-compatible `trace.json`. The trace separates forward/loss, backward and clipping/AdamW and includes CUDA kernels. It is a diagnostic, not the real-data training baseline. PyTorch's `with_flops` estimates work for supported operators; fused/backward coverage is incomplete. Do not add nested operator/kernel durations together as though they were disjoint. Profiler overhead changes timing.

Actual SM occupancy, instruction mix and Tensor Core/hardware FLOP counters require a dedicated tool such as Nsight Compute. This host has NVIDIA profiling tools installed, but this script does not collect those counters or change their permissions. Counter access and version compatibility must be verified separately. GPU utilization alone cannot supply those measurements.

References: [NVIDIA query definitions](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries), [nvidia-smi reference](https://docs.nvidia.com/deploy/nvidia-smi/), [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html), [Nsight Compute profiling and permissions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## First actual observation

On November 14, the monitor sampled the RTX 3090 at a requested 0.5-second interval around a 100-update run, with 10 seconds requested before launch and 30 seconds afterward. The record contains **174 samples over 87.77 seconds**. Validation again ended at **7.6613** and generation remained repetitive.

The training loop took **42.03 seconds**. Across the observation, checkpoint-write markers totaled **40.14 seconds**, while completed training-step markers totaled **1.26 seconds**. These scopes differ slightly (initial/final saves are outside the loop); they nevertheless show checkpoint serialization dominating this small run. The chart aligns long checkpoint intervals with low GPU activity, while brief update bursts appear between saves. This is evidence about this run, not a claim that the GPU is generally underused.

Sampled GPU power peaked at **197.6 W** and temperature at **57°C**. Gross sampled board energy was about **6.77 kJ** across the entire observation. CPU package power was unavailable because the energy counter was permission-restricted. Desktop GPU activity continued during idle/cooldown. Comparing the monitored run with the earlier unmonitored resume verification found **exact equality across all 358 latest-checkpoint tensors and metadata**. The separate one-step profiler produced **193 CUDA kernel events**; supported-operator formula counts totaled about **96.65 billion FLOPs**, versus the **99.86-billion** model estimate. Neither is a hardware FLOP counter.

The [saved chart](results/rtx3090-monitor/timeline.png), [summary](results/rtx3090-monitor/summary.json), and raw records in that directory preserve this observation. Run the rendering command above against `experiments/cosmopedia/results/rtx3090-monitor` to recreate the portable report locally. Checkpoints and the large profiler trace stay in ignored artifacts.
