"""Live dashboard, cooperative stop and aggregated-history regression checks."""

import json
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

import torch
from live import LiveLog, active_run, atomic_json, loss_svg
from test_resume import make_fixture
from verify_resume import compare_state


class LiveTests(unittest.TestCase):
    def wait_json(self, path: Path, predicate: object = None) -> dict:
        """Wait for a real child-process progress file rather than using fixed sleeps."""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if path.exists():
                value = json.loads(path.read_text())
                if predicate is None or predicate(value):
                    return value
            time.sleep(0.03)
        self.fail(f"Timed out waiting for {path}")

    def command(self, data: Path, output: Path, extra: list[str]) -> list[str]:
        return [
            sys.executable,
            str(Path(__file__).with_name("train.py")),
            "--data",
            str(data),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--steps",
            "60",
            "--eval-every",
            "20",
            "--checkpoint-every",
            "30",
            "--validation-windows",
            "4",
            "--log-every",
            "5",
            *extra,
        ]

    def test_dashboard_stop_resume_aggregates_exact_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = make_fixture(root)
            session = root / "session"
            server = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("live.py")),
                    "--session",
                    str(session),
                    "--port",
                    "0",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            trainer: subprocess.Popen | None = None
            try:
                url = self.wait_json(session / "dashboard.json")["url"]
                # Other browser origins cannot issue stop requests.
                bad = urllib.request.Request(
                    url + "/stop",
                    data=b"{}",
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "https://example.com",
                    },
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(bad, timeout=5)
                self.assertEqual(error.exception.code, 403)
                trainer = subprocess.Popen(
                    self.command(data, session / "part", ["--session", str(session)]),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.wait_json(
                    session / "part/state.json",
                    lambda row: row["step"] >= 5 and row["status"] == "running",
                )
                request = urllib.request.Request(
                    url + "/stop",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertTrue(json.load(response)["stop_requested"])
                _, stderr = trainer.communicate(timeout=30)
                self.assertEqual(trainer.returncode, 0, stderr)
                part = json.loads((session / "part/state.json").read_text())
                self.assertEqual(part["status"], "stopped")
                self.assertLess(part["step"], 60)
                self.assertEqual(part["latest_checkpoint_step"], part["step"])
                ElementTree.fromstring((session / "part/loss.svg").read_text())
                subprocess.run(
                    self.command(
                        data,
                        session / "resumed",
                        [
                            "--session",
                            str(session),
                            "--resume",
                            str(session / "part/latest.pt"),
                        ],
                    ),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                subprocess.run(
                    self.command(data, root / "full", []),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )
                compare_state(
                    torch.load(root / "full/latest.pt", weights_only=True),
                    torch.load(session / "resumed/latest.pt", weights_only=True),
                )
                self.assertEqual(active_run(session), (session / "resumed").resolve())
                with urllib.request.urlopen(url + "/state", timeout=5) as response:
                    combined = json.load(response)
                self.assertEqual(combined["step"], 60)
                self.assertEqual(combined["status"], "completed")
                self.assertEqual(
                    [p["step"] for p in combined["training_curve"]],
                    list(range(5, 61, 5)),
                )
                with urllib.request.urlopen(url + "/loss.svg", timeout=5) as response:
                    ElementTree.fromstring(response.read())
            finally:
                if trainer is not None and trainer.poll() is None:
                    trainer.terminate()
                    trainer.communicate(timeout=30)
                server.terminate()
                server.communicate(timeout=10)

    def test_signal_and_time_budget_save_latest_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = make_fixture(root)
            output = root / "signal"
            trainer = subprocess.Popen(
                self.command(data, output, ["--steps", "100000"]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.wait_json(output / "state.json", lambda row: row["step"] >= 5)
                trainer.send_signal(signal.SIGTERM)
                _, stderr = trainer.communicate(timeout=30)
                self.assertEqual(trainer.returncode, 0, stderr)
                state = json.loads((output / "state.json").read_text())
                self.assertEqual(state["stop_reason"], "SIGTERM")
                self.assertEqual(state["step"], state["latest_checkpoint_step"])
                self.assertTrue((output / "loss.svg").exists())
            finally:
                if trainer.poll() is None:
                    trainer.terminate()
                    trainer.communicate(timeout=30)
            timed = root / "timed"
            subprocess.run(
                self.command(data, timed, ["--max-seconds", "0.000001"]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            self.assertEqual(
                json.loads((timed / "state.json").read_text())["stop_reason"],
                "time budget",
            )
            self.assertTrue((timed / "latest.pt").exists())

    def test_svg_pointer_and_single_writer(self) -> None:
        state = {
            "step": 9,
            "status": "stopped",
            "training_curve": [{"step": 5, "loss": 2.0}],
            "history": [{"step": 0, "validation_loss": 3.0}],
            "pending_training_point": {"step": 9, "loss": 1.5},
        }
        ElementTree.fromstring(loss_svg(state))
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary)
            first, second = session / "first", session / "second"
            first.mkdir()
            second.mkdir()
            writer = LiveLog(first, session)
            try:
                with self.assertRaisesRegex(ValueError, "already owns"):
                    LiveLog(second, session)
            finally:
                writer.lock.close()
            atomic_json(session / "active.json", {"run": "../escape"})
            with self.assertRaisesRegex(ValueError, "Invalid active"):
                active_run(session)


if __name__ == "__main__":
    unittest.main()
