"""Offline counter/phase checks and command failure handling for monitoring."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from events import Events, estimated_training_flops
from monitor import cpu_percent, cpu_ticks, gpu_snapshot, number
from monitor_plot import board_energy, read_jsonl, spans


class MonitorTests(unittest.TestCase):
    def test_cpu_guest_iowait_and_missing_counters(self) -> None:
        old = cpu_ticks("cpu 100 0 0 100 50 0 0 0 20 0\ncpu0 100 0 0 100 50 0 0 0 20 0")
        new = cpu_ticks(
            "cpu 150 0 0 150 100 0 0 0 40 0\ncpu0 150 0 0 150 100 0 0 0 40 0"
        )
        self.assertAlmostEqual(cpu_percent(old, new)["cpu"], 100 / 3)
        self.assertIsNone(cpu_percent(new, new)["cpu0"])
        self.assertIsNone(number("[N/A]"))
        self.assertIsNone(number("nan"))
        self.assertEqual(number("0"), 0)

    def test_gpu_unavailable_is_missing_not_zero(self) -> None:
        with mock.patch(
            "monitor.subprocess.run", side_effect=FileNotFoundError("nvidia-smi")
        ):
            self.assertIn("error", gpu_snapshot("0"))
        result = subprocess.CompletedProcess(
            [],
            0,
            "0, GPU-test, Test GPU, 90, 40, 512, 24576, 60, [N/A], 150, 350, 1500, 9000, P2\n",
            "",
        )
        with mock.patch("monitor.subprocess.run", return_value=result):
            row = gpu_snapshot("0")
        self.assertEqual(row["power_w"], 150)
        self.assertIsNone(row["fan_percent"])

    def test_phase_pairing_partial_log_and_sampled_energy(self) -> None:
        events = [
            {
                "monotonic_ns": 1_000_000_000,
                "phase": "checkpoint",
                "kind": "start",
                "step": 25,
            },
            {
                "monotonic_ns": 3_000_000_000,
                "phase": "checkpoint",
                "kind": "end",
                "step": 25,
            },
            {"monotonic_ns": 4_000_000_000, "phase": "generation", "kind": "start"},
        ]
        result = spans(events, 0, 5_000_000_000)
        self.assertEqual(result[0]["end_s"] - result[0]["start_s"], 2)
        self.assertEqual(result[1]["status"], "unfinished")
        samples = [
            {"monotonic_ns": 0, "gpu": {"power_w": 100}},
            {"monotonic_ns": 2_000_000_000, "gpu": {"power_w": 200}},
        ]
        self.assertEqual(board_energy(samples), 300)
        self.assertIsNone(board_energy([{"monotonic_ns": 0, "gpu": {}}]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text('{"phase":"train_step"}\n{"partial":')
            self.assertEqual(len(read_jsonl(path)), 1)
            path.write_text('{"partial":\n')
            with self.assertRaises(json.JSONDecodeError):
                read_jsonl(path)

    def test_failed_command_preserves_report_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "monitor"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("monitor.py")),
                    "--output",
                    str(output),
                    "--pre-seconds",
                    "0",
                    "--post-seconds",
                    "0",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["command_exit_code"], 7)
            self.assertTrue((output / "report.html").exists())
            self.assertTrue((output / "command.log").exists())

    def test_events_preserve_failed_phase_and_estimate_convention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            with mock.patch.dict(os.environ, {"COSMOPEDIA_EVENTS": str(path)}):
                events = Events()
                with (
                    self.assertRaisesRegex(RuntimeError, "failed"),
                    events.phase("validation", step=5),
                ):
                    raise RuntimeError("failed")
            recorded = read_jsonl(path)
            self.assertEqual(recorded[-1]["status"], "failed")
        config = {"width": 2, "hidden_width": 4, "layers": 1, "vocabulary_size": 8}
        self.assertEqual(estimated_training_flops(config, 1, 2), 672)


if __name__ == "__main__":
    unittest.main()
