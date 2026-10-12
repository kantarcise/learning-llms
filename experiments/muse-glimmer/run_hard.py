"""Stream one harder task and sample host resources; standard library only."""

import argparse
import datetime
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def utc_now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def capture(pid):
    sample = {"utc": utc_now(), "monotonic": time.monotonic()}
    for name, fields in {
        "gpu": "timestamp,memory.used,memory.total,utilization.gpu,"
        "utilization.memory,power.draw,temperature.gpu",
        "compute-apps": "pid,used_gpu_memory",
    }.items():
        result = subprocess.run(
            ["nvidia-smi", f"--query-{name}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        sample[name] = result.stdout.strip()
        if result.returncode:
            sample[name + "_error"] = result.stderr.strip()
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            mem[key + "_KiB"] = int(value.split()[0])
    sample["system_memory"] = mem
    sample["load_average"] = list(os.getloadavg())
    try:
        status = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize", "Threads"}:
                status[key] = int(value.split()[0])
        # Fields after the process name start with field 3 (state).
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        sample["process"] = {
            "pid": pid,
            "status": status,
            "cpu_seconds": (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
            "io": Path(f"/proc/{pid}/io").read_text(),
        }
    except OSError as error:
        sample["process_error"] = str(error)
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--template-kwargs", default='{"reasoning_strength":"low"}')
    parser.add_argument("--reasoning-effort")
    args = parser.parse_args()
    if urlparse(args.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Use a loopback endpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    phase = ["waiting_for_health"]
    stop = threading.Event()

    def monitor():
        with (args.output / "resources.jsonl").open("w") as target:
            while not stop.is_set():
                started = time.monotonic()
                try:
                    sample = capture(args.pid)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    sample = {"utc": utc_now(), "error": repr(error)}
                sample["phase"] = phase[0]
                target.write(json.dumps(sample) + "\n")
                target.flush()
                stop.wait(max(0, 0.5 - (time.monotonic() - started)))

    worker = threading.Thread(target=monitor)
    worker.start()
    record = {"model": args.model, "pid": args.pid, "started_utc": utc_now()}
    try:
        health_start = time.monotonic()
        while True:
            try:
                with urllib.request.urlopen(args.base_url + "/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        break
            except OSError:
                if time.monotonic() - health_start > 240:
                    raise TimeoutError("Model did not become healthy") from None
            time.sleep(0.5)
        record["health_wait_seconds"] = time.monotonic() - health_start
        phase[0] = "baseline"
        time.sleep(1)
        prompt = Path(__file__).with_name("hard-prompt.txt").read_text()
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 6144,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": args.top_k,
            "seed": 42,
            "chat_template_kwargs": json.loads(args.template_kwargs),
        }
        if args.reasoning_effort:
            payload["reasoning_effort"] = args.reasoning_effort
        record["request"] = payload
        req = urllib.request.Request(
            args.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        phase[0] = "request"
        record["request_started_utc"] = utc_now()
        started = time.monotonic()
        answer, reasoning = [], []
        with (
            (args.output / "stream.jsonl").open("w") as target,
            urllib.request.urlopen(req, timeout=600) as response,
        ):
            record["headers_seconds"] = time.monotonic() - started
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                elapsed = time.monotonic() - started
                target.write(json.dumps({"seconds": elapsed, "event": event}) + "\n")
                target.flush()
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    for key, parts, timing in [
                        ("content", answer, "first_answer_seconds"),
                        ("reasoning_content", reasoning, "first_reasoning_seconds"),
                    ]:
                        if delta.get(key):
                            parts.append(delta[key])
                            record.setdefault(timing, elapsed)
                            record.setdefault("first_text_seconds", elapsed)
                    if choice.get("finish_reason"):
                        record["finish_reason"] = choice["finish_reason"]
                for key in ("usage", "timings"):
                    if event.get(key):
                        record[key] = event[key]
        record["elapsed_seconds"] = time.monotonic() - started
        (args.output / "answer.py").write_text("".join(answer))
        (args.output / "reasoning.txt").write_text("".join(reasoning))
        phase[0] = "post_request"
        time.sleep(1)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record["error"] = repr(error)
    finally:
        record["ended_utc"] = utc_now()
        stop.set()
        worker.join()
        (args.output / "request.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "request"}, indent=2))
    if "error" in record:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
