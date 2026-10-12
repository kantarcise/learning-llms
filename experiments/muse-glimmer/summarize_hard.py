"""Summarize request-phase samples without treating them as exact peaks."""

import argparse
import json
from itertools import pairwise
from pathlib import Path


def summarize(directory):
    record = json.loads((directory / "request.json").read_text())
    all_samples = [
        json.loads(line)
        for line in (directory / "resources.jsonl").read_text().splitlines()
    ]
    samples = [s for s in all_samples if s["phase"] == "request"]
    gpu = [[float(v) for v in s["gpu"].split(",")[1:]] for s in samples]
    process = [s for s in samples if "process" in s]
    cpu = [
        100
        * (b["process"]["cpu_seconds"] - a["process"]["cpu_seconds"])
        / (b["monotonic"] - a["monotonic"])
        for a, b in pairwise(process)
    ]
    pid_memory = []
    for sample in samples:
        for line in sample["compute-apps"].splitlines():
            pid, memory = line.split(",")
            if int(pid) == record["pid"]:
                pid_memory.append(float(memory))
    result = {
        "model": record["model"],
        "request_samples": len(samples),
        "request_sample_gap_max_seconds": max(
            b["monotonic"] - a["monotonic"] for a, b in pairwise(samples)
        ),
        "all_phase_samples": len(all_samples),
        "all_phase_process_RSS_peak_MiB": max(
            s["process"]["status"]["VmRSS"] for s in all_samples if "process" in s
        )
        / 1024,
        "all_phase_system_used_memory_peak_MiB": max(
            s["system_memory"]["MemTotal_KiB"] - s["system_memory"]["MemAvailable_KiB"]
            for s in all_samples
            if "system_memory" in s
        )
        / 1024,
        "telemetry_errors": [s for s in all_samples if any("error" in k for k in s)],
        "sampled_device_memory_peak_MiB": max(row[0] for row in gpu),
        "sampled_process_gpu_memory_peak_MiB": max(pid_memory),
        "sampled_gpu_utilization_mean_percent": sum(row[2] for row in gpu) / len(gpu),
        "sampled_gpu_utilization_peak_percent": max(row[2] for row in gpu),
        "sampled_gpu_power_peak_W": max(row[4] for row in gpu),
        "sampled_gpu_temperature_peak_C": max(row[5] for row in gpu),
        "sampled_process_RSS_peak_MiB": max(
            s["process"]["status"]["VmRSS"] for s in process
        )
        / 1024,
        "sampled_system_used_memory_peak_MiB": max(
            s["system_memory"]["MemTotal_KiB"] - s["system_memory"]["MemAvailable_KiB"]
            for s in samples
        )
        / 1024,
        "process_cpu_interval_peak_percent_one_core": max(cpu),
        "process_cpu_mean_percent_one_core": 100
        * (process[-1]["process"]["cpu_seconds"] - process[0]["process"]["cpu_seconds"])
        / (process[-1]["monotonic"] - process[0]["monotonic"]),
    }
    result.update(
        {
            k: v
            for k, v in record.items()
            if k.endswith("seconds")
            or k in {"usage", "timings", "finish_reason", "error"}
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
