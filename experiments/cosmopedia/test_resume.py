"""Offline process-boundary checks for complete training continuation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from model import Config, LanguageModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


class ResumeTests(unittest.TestCase):
    def assert_nested_equal(self, left: object, right: object) -> None:
        """Require exact tensor and scalar equality throughout serialized state."""
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right, strict=True):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def run_training(
        self, data: Path, output: Path, extra: list[str]
    ) -> subprocess.CompletedProcess[str]:
        """Run the real CLI in a fresh CPU process, using synthetic offline data."""
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("train.py")),
                "--device",
                "cpu",
                "--data",
                str(data),
                "--output",
                str(output),
                "--steps",
                "6",
                "--eval-every",
                "3",
                "--checkpoint-every",
                "2",
                "--validation-windows",
                "4",
                *extra,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def test_cli_interrupted_between_evaluations_matches_uninterrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            vocab = {
                "<|endoftext|>": 0,
                "[UNK]": 1,
                "Once": 2,
                "upon": 3,
                "a": 4,
                "time": 5,
            }
            vocab.update({f"t{i}": i for i in range(6, 32)})
            tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.save(str(data / "tokenizer.json"))
            ids = torch.arange(12 * 9).reshape(12, 9) % 32
            tensors = {"inputs": ids[:, :-1].clone(), "targets": ids[:, 1:].clone()}
            tensors["targets"][::2, -2:] = -100
            for split in ["train", "validation"]:
                torch.save(tensors, data / f"{split}.pt")
            manifest = {
                "vocabulary_size": 32,
                "context_length": 8,
                "eos_id": 0,
                "dataset_revision": "offline-fixture",
                "tokenizer_revision": "offline-fixture",
                "token_counts": {"train": 108, "validation": 108},
            }
            (data / "manifest.json").write_text(json.dumps(manifest))
            self.run_training(data, root / "full", [])
            self.run_training(data, root / "part", ["--stop-after", "2"])
            part = torch.load(root / "part/latest.pt", weights_only=True)
            self.assertEqual(part["progress"]["step"], 2)
            self.assertGreater(part["progress"]["interval_tokens"], 0)
            self.assertEqual(len(part["progress"]["history"]), 1)
            # latest.pt is self-contained: no companion checkpoint is required.
            (root / "part/best.pt").unlink()
            self.run_training(
                data, root / "resumed", ["--resume", str(root / "part/latest.pt")]
            )
            full = torch.load(root / "full/latest.pt", weights_only=True)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=True)
            self.assert_nested_equal(full, resumed)
            full_report = json.loads((root / "full/run.json").read_text())
            report = json.loads((root / "resumed/run.json").read_text())
            self.assertEqual(full_report["history"], report["history"])
            self.assertEqual(full_report["after"], report["after"])
            self.assertLess(
                report["segment_training_target_tokens"],
                report["training_target_tokens_seen"],
            )
            # best.pt also contains a coherent optimizer/RNG snapshot.
            self.run_training(
                data,
                root / "from-best",
                ["--resume", str(root / "resumed/best.pt"), "--steps", "9"],
            )
            self.run_training(data, root / "full-nine", ["--steps", "9"])
            self.assert_nested_equal(
                torch.load(root / "from-best/latest.pt", weights_only=True),
                torch.load(root / "full-nine/latest.pt", weights_only=True),
            )
            best = torch.load(root / "resumed/best.pt", weights_only=True)
            self.assertEqual(best["progress"]["step"], best["progress"]["best_step"])
            # A changed tensor file must fail even with an unchanged manifest.
            tensors["inputs"][0, 0] = 31
            torch.save(tensors, data / "train.pt")
            with self.assertRaises(subprocess.CalledProcessError) as error:
                self.run_training(
                    data,
                    root / "wrong-data",
                    ["--resume", str(root / "part/latest.pt")],
                )
            self.assertIn("data_sha256", error.exception.stderr)
            self.assertFalse((root / "wrong-data").exists())

    def test_rng_restoration_and_rejection_of_incompatible_checkpoints(self) -> None:
        torch.manual_seed(7)
        model = LanguageModel(
            Config(16, context_length=4, width=8, heads=2, layers=1, hidden_width=16)
        )
        optimizer = torch.optim.AdamW(model.parameters())
        sampling = torch.Generator().manual_seed(17)
        device = torch.device("cpu")
        identity = {
            "config": model.configuration(),
            "batch_size": 4,
            "device_type": "cpu",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            save_checkpoint(
                path, model, optimizer, sampling, device, identity, {"step": 0}
            )
            expected_rng = torch.rand(4)
            expected_sample = torch.randint(100, (4,), generator=sampling)
            torch.manual_seed(99)
            sampling.manual_seed(99)
            restore_checkpoint(
                load_checkpoint(path, identity), model, optimizer, sampling, device
            )
            self.assertTrue(torch.equal(torch.rand(4), expected_rng))
            self.assertTrue(
                torch.equal(
                    torch.randint(100, (4,), generator=sampling), expected_sample
                )
            )
            previous = path.read_bytes()
            with (
                mock.patch("checkpoint.torch.save", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                save_checkpoint(
                    path, model, optimizer, sampling, device, identity, {"step": 1}
                )
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            for key, value in [
                ("batch_size", 8),
                ("device_type", "cuda"),
                ("config", {}),
            ]:
                with self.assertRaisesRegex(ValueError, key):
                    load_checkpoint(path, identity | {key: value})
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "weights-only"):
                load_checkpoint(path, identity)


if __name__ == "__main__":
    unittest.main()
