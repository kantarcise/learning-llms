"""Behavior checks for the learning example; run with unittest discovery."""

import json
import unittest
from pathlib import Path

import torch
from data import CharacterWindows, encode, load_data
from model import TinyTransformer
from torch.utils.data import DataLoader
from train import evaluate, loss_for


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.train, self.validation, self.vocabulary = load_data()
        self.model = TinyTransformer(len(self.vocabulary)).eval()

    def test_inspected_fixture_matches_data(self):
        fixture = json.loads(Path(__file__).with_name("first-batch.json").read_text())
        self.assertEqual(fixture["vocabulary"], self.vocabulary)
        for text, example in zip(self.train, fixture["examples"]):
            self.assertEqual(encode(text[:8], self.vocabulary), example["input_ids"])
            self.assertEqual(encode(text[1:9], self.vocabulary), example["target_ids"])

    def test_windows_do_not_cross_records(self):
        windows = CharacterWindows(["abcd", "dcba"], list("abcd"), 2)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[1][0].tolist(), [1, 2])
        self.assertEqual(windows[1][1].tolist(), [2, 3])
        self.assertEqual(windows[2][0].tolist(), [3, 2])

    def test_future_tokens_cannot_change_earlier_logits(self):
        x = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = x.clone()
        changed[:, 3:] = torch.tensor([7, 8, 9])
        with torch.no_grad():
            torch.testing.assert_close(self.model(x)[:, :3], self.model(changed)[:, :3])

    def test_batch_examples_are_independent(self):
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[5, 6, 7, 8]])
        with torch.no_grad():
            torch.testing.assert_close(self.model(a), self.model(torch.cat((a, b)))[:1])

    def test_validation_does_not_update_parameters(self):
        loader = DataLoader(
            CharacterWindows(self.validation, self.vocabulary, 8), batch_size=16
        )
        before = {
            name: value.clone() for name, value in self.model.state_dict().items()
        }
        self.assertGreater(evaluate(self.model, loader), 0)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_training_step_reaches_embeddings_and_both_sublayers(self):
        windows = CharacterWindows(self.train, self.vocabulary, 8)
        x, y = next(iter(DataLoader(windows, batch_size=2)))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_for(self.model(x), y).backward()
        weights = [
            self.model.token_embedding.weight,
            self.model.block.attention.query.weight,
            self.model.block.feedforward[0].weight,
            self.model.block.norm2.weight,
        ]
        before = [p.detach().clone() for p in weights]
        for p in weights:
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertTrue(all(not torch.equal(p, old) for p, old in zip(weights, before)))


if __name__ == "__main__":
    unittest.main()
