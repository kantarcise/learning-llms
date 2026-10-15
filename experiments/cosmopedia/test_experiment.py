"""Offline checks for causal isolation, document windows, and training gradients."""

import unittest

import torch
from model import Config, LanguageModel
from prepare import windows
from train import loss_for


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = LanguageModel(
            Config(32, context_length=8, width=16, heads=4, layers=2, hidden_width=32)
        )

    def test_future_and_other_batch_examples_cannot_change_prefix(self) -> None:
        ids = torch.randint(0, 32, (2, 8))
        original = self.model(ids)
        changed = ids.clone()
        changed[0, 4:] = (changed[0, 4:] + 1) % 32
        changed[1] = (changed[1] + 2) % 32
        actual = self.model(changed)
        torch.testing.assert_close(original[0, :4], actual[0, :4])
        self.assertEqual(actual.shape, (2, 8, 32))

    def test_gradients_and_optimizer_change_parameters(self) -> None:
        ids = torch.randint(0, 32, (2, 8))
        targets = torch.randint(0, 32, (2, 8))
        before = self.model.token_embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        loss_for(self.model(ids), targets).backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        torch.testing.assert_close(before, self.model.token_embedding.weight)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.token_embedding.weight))

    def test_document_tail_padding_and_eos_targets(self) -> None:
        inputs, targets = windows([3, 4, 5, 6, 7, 0], context=4, eos=0)
        self.assertEqual(inputs, [[3, 4, 5, 6], [7, 0, 0, 0]])
        self.assertEqual(targets, [[4, 5, 6, 7], [0, -100, -100, -100]])
        logits = torch.randn(2, 4, 32)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 32)[:5], torch.tensor([4, 5, 6, 7, 0])
        )
        torch.testing.assert_close(loss_for(logits, torch.tensor(targets)), expected)

    def test_context_limit_and_head_width_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))
        with self.assertRaises(ValueError):
            LanguageModel(Config(32, width=15, heads=4))


if __name__ == "__main__":
    unittest.main()
