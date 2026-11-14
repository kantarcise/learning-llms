"""Record Linux/NVIDIA telemetry before, during and after an arbitrary command.

Example: monitor.py --output artifacts/observed -- python train.py ...
Uses existing nvidia-smi, /proc and readable sysfs sensors; never changes fan,
clock, power or system permissions. No additional Python dependencies.
"""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

GPU_FIELDS = [
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "fan.speed",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "pstate",
]
GPU_KEYS = [
    "index",
    "uuid",
    "name",
    "busy_percent",
    "memory_busy_percent",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "fan_percent",
    "power_w",
    "power_limit_w",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "pstate",
]


def number(value: str) -> float | None:
    """Unsupported sensor readings stay missing, never become fabricated zeros."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def gpu_snapshot(gpu: str) -> dict:
    """Query one explicitly selected NVIDIA GPU with a bounded timeout."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu,
                f"--query-gpu={','.join(GPU_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode:
            return {"error": result.stderr.strip() or result.stdout.strip()}
        rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != len(GPU_KEYS):
            return {"error": "Unexpected nvidia-smi output"}
        return {
            key: value.strip() if key in ["uuid", "name", "pstate"] else number(value)
            for key, value in zip(GPU_KEYS, rows[0], strict=True)
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def cpu_ticks(text: str) -> dict[str, tuple[int, int]]:
    """Return busy and total jiffies, excluding duplicated guest accounting."""
    result: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("cpu"):
            values = [int(v) for v in parts[1:9]]
            total = sum(values)
            result[parts[0]] = (total - values[3] - values[4], total)
    return result


def cpu_percent(before: dict, after: dict) -> dict[str, float | None]:
    """Compute aggregate and per-logical-CPU busy time between snapshots."""
    result: dict[str, float | None] = {}
    for key, (busy, total) in after.items():
        old_busy, old_total = before.get(key, (busy, total))
        delta = total - old_total
        result[key] = (
            max(0.0, min(100.0, 100 * (busy - old_busy) / delta)) if delta > 0 else None
        )
    return result


def sensors() -> tuple[dict[str, float], list[str]]:
    """Read temperature °C and fan RPM; retain sensor labels, even suspect ones."""
    readings: dict[str, float] = {}
    errors: list[str] = []
    for folder in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (folder / "name").read_text().strip()
        except OSError:
            continue
        for pattern, scale, unit in [
            ("temp*_input", 1000, "c"),
            ("fan*_input", 1, "rpm"),
        ]:
            for source in folder.glob(pattern):
                try:
                    prefix = source.name.removesuffix("_input")
                    label_path = folder / f"{prefix}_label"
                    label = (
                        label_path.read_text().strip()
                        if label_path.exists()
                        else prefix
                    )
                    readings[f"{name}/{label}/{unit}"] = (
                        float(source.read_text()) / scale
                    )
                except (OSError, ValueError) as error:
                    errors.append(f"{source}: {error}")
    return readings, errors


class Sampler:
    """Keep interval counters for system CPUs, command process tree and energy."""

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu
        self.previous_cpu: dict = {}
        self.previous_process: dict[tuple[int, int], int] = {}
        self.previous_time: float | None = None
        self.previous_energy: dict[str, tuple[float, float]] = {}

    def process_tree(self, pid: int | None, elapsed: float) -> dict:
        """Approximate live-tree CPU/RSS/I/O; short-lived exited children can be missed."""
        if pid is None:
            self.previous_process = {}
            return {}
        table: dict[int, tuple[int, int, int, int]] = {}
        for folder in Path("/proc").glob("[0-9]*"):
            try:
                fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
                table[int(folder.name)] = (
                    int(fields[1]),
                    int(fields[19]),
                    int(fields[11]) + int(fields[12]),
                    int(fields[21]),
                )
            except (OSError, ValueError, IndexError):
                continue
        selected = {pid}
        while True:
            expanded = selected | {p for p, row in table.items() if row[0] in selected}
            if expanded == selected:
                break
            selected = expanded
        ticks = 0
        rss = 0
        io: dict[str, int] = {}
        current: dict[tuple[int, int], int] = {}
        for p in selected & table.keys():
            _, start, cpu, pages = table[p]
            key = (p, start)
            current[key] = cpu
            ticks += max(0, cpu - self.previous_process.get(key, cpu))
            rss += pages * os.sysconf("SC_PAGE_SIZE")
            try:
                for line in Path(f"/proc/{p}/io").read_text().splitlines():
                    name, value = line.split(":")
                    if name in ["read_bytes", "write_bytes"]:
                        io[name] = io.get(name, 0) + int(value)
            except OSError:
                pass
        self.previous_process = current
        return {
            "cpu_percent_one_core": 100 * ticks / os.sysconf("SC_CLK_TCK") / elapsed
            if elapsed > 0
            else None,
            "summed_rss_bytes": rss,
            "live_processes": len(current),
            "io_bytes_live_processes": io,
        }

    def sample(self, pid: int | None) -> dict:
        """Take a timestamped snapshot; sensor queries are sequential, not simultaneous."""
        started = time.monotonic_ns()
        now = time.monotonic()
        elapsed = now - self.previous_time if self.previous_time is not None else 0
        ticks = cpu_ticks(Path("/proc/stat").read_text())
        cpu = cpu_percent(self.previous_cpu, ticks)
        self.previous_cpu, self.previous_time = ticks, now
        memory = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.split(":")[0] in ["MemTotal", "MemAvailable"]
        }
        sensor_values, sensor_errors = sensors()
        energy: dict[str, float | None] = {}
        errors = sensor_errors
        for folder in Path("/sys/class/powercap").glob("intel-rapl:*"):
            # Top-level packages only: nested domains would double-count energy.
            if folder.name.count(":") != 1:
                continue
            try:
                label = (folder / "name").read_text().strip()
                value = float((folder / "energy_uj").read_text())
                maximum = float((folder / "max_energy_range_uj").read_text())
                old = self.previous_energy.get(label)
                energy[label] = (
                    ((value - old[0]) % maximum) / 1e6 / (now - old[1])
                    if old and now > old[1]
                    else None
                )
                self.previous_energy[label] = (value, now)
            except (OSError, ValueError) as error:
                errors.append(f"CPU energy unavailable: {error}")
        process = self.process_tree(pid, elapsed)
        gpu = gpu_snapshot(self.gpu)
        return {
            "monotonic_ns": started,
            "wall_time_utc": datetime.now(UTC).isoformat(),
            "collection_seconds": (time.monotonic_ns() - started) / 1e9,
            "cpu_percent": cpu,
            "memory_bytes": memory,
            "process": process,
            "sensors": sensor_values,
            "cpu_package_power_w": energy,
            "gpu": gpu,
            "errors": errors,
        }


def main() -> None:
    """Own the command lifecycle and save partial data even when it fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--pre-seconds", type=float, default=10.0)
    parser.add_argument("--post-seconds", type=float, default=30.0)
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU index or UUID to observe; does not change device visibility",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (
        not command
        or not all(
            math.isfinite(v)
            for v in [args.interval, args.pre_seconds, args.post_seconds]
        )
        or args.interval < 0.1
        or min(args.pre_seconds, args.post_seconds) < 0
    ):
        parser.error(
            "Provide a command, interval >= 0.1s, and nonnegative pre/post durations"
        )
    if not all(Path(name).is_file() for name in ("/proc/stat", "/proc/meminfo")):
        parser.error(
            "Hardware monitoring requires Linux with /proc/stat and /proc/meminfo"
        )
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    start_ns = time.monotonic_ns()
    metadata = {
        "command": command,
        "gpu_selection": args.gpu,
        "interval_seconds": args.interval,
        "pre_seconds": args.pre_seconds,
        "post_seconds": args.post_seconds,
        "start_monotonic_ns": start_ns,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "schema_version": 1,
        "scope": "System/GPU metrics include other applications; process CPU is per logical core (can exceed 100%), tree RSS can double-count shared pages. GPU busy time is not core occupancy or FLOPs. Sensors are sequential snapshots. CPU energy is optional.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    interrupted = False

    def stop(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    old_handlers = {
        sig: signal.signal(sig, stop) for sig in [signal.SIGINT, signal.SIGTERM]
    }
    sampler = Sampler(args.gpu)
    process: subprocess.Popen | None = None
    next_sample = time.monotonic()
    exit_code = 130
    with (
        (args.output / "samples.jsonl").open("w") as samples,
        (args.output / "command.log").open("w") as log,
        (args.output / "lifecycle.jsonl").open("w") as lifecycle,
    ):

        def event(phase: str, kind: str, **values: object) -> None:
            lifecycle.write(
                json.dumps(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "phase": phase,
                        "kind": kind,
                        **values,
                    }
                )
                + "\n"
            )
            lifecycle.flush()

        def collect(until: float | None = None) -> None:
            nonlocal next_sample
            while not interrupted:
                if until is not None and time.monotonic() >= until:
                    return
                if until is None and process is not None and process.poll() is not None:
                    return
                row = sampler.sample(
                    process.pid if process and process.poll() is None else None
                )
                samples.write(json.dumps(row) + "\n")
                samples.flush()
                next_sample = max(next_sample + args.interval, time.monotonic())
                deadline = min(next_sample, until) if until is not None else next_sample
                while not interrupted and time.monotonic() < deadline:
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))

        try:
            event("idle", "start")
            collect(time.monotonic() + args.pre_seconds)
            event("idle", "end")
            if not interrupted:
                environment = os.environ.copy()
                environment["COSMOPEDIA_EVENTS"] = str(
                    (args.output / "events.jsonl").resolve()
                )
                event("command", "start")
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                collect()
                if interrupted and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
                event("command", "end", exit_code=exit_code)
                event("cooldown", "start")
                collect(time.monotonic() + args.post_seconds)
                event("cooldown", "end")
        except OSError as error:
            metadata["error"] = str(error)
            exit_code = 127
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            metadata.update(command_exit_code=exit_code, interrupted=interrupted)
            (args.output / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    from monitor_plot import render

    render(args.output)
    print(f"Monitoring report: {args.output / 'report.html'}", flush=True)
    raise SystemExit(
        130 if interrupted else exit_code if exit_code >= 0 else 128 - exit_code
    )


if __name__ == "__main__":
    main()
