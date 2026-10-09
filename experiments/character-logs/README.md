# Character-level fictional logs 🧩

Preparation started October 8, 2026. **A tiny model and training pipeline are implemented; the assistant ran a ten-epoch CPU demonstration.** This is preparation outside the book chapters, not a completed independent exercise.

This small companion experiment starts with a question: how do readable log sequences become character IDs, shifted targets, and a batch? The book remains the main implementation path. No independent exercise completion is claimed; the assistant-run demonstration is recorded below.

## Data and scope

[sequences.json](sequences.json) contains six original, assistant-authored fictional job sequences: four assigned to training and two to validation. They are synthetic teaching examples, not employer logs. Entire sequences are assigned before making windows; windows must stay within their source sequence.

The validation sequences use different job names but deliberately share the same event templates. They can check limited generalization within these templates, not general language ability, unseen workflow understanding, or operational correctness. This handful of examples is for inspecting the pipeline; a meaningful training experiment will need a separately agreed data design and success criterion.

## First inspectable batch

[first-batch.json](first-batch.json) records the character vocabulary, exact IDs, and two examples. The vocabulary is the sorted set of characters appearing in **training** text, with zero-based indices as IDs. It contains 25 characters, including newline, space, `=`, and `1`. Validation introduces no unknown characters. There is no special end token in this initial fixture; sequence boundaries are represented by separate records.

Use batch size 2 and input length 8. Take the first 9 characters from `train-orders` and `train-users`. The first 8 become inputs, and the last 8 become targets:

| Example | Input text | Target text |
| --- | --- | --- |
| A | `job=orde` | `ob=order` |
| B | `job=user` | `ob=users` |

```text
Input IDs:
[[12, 17, 5, 3, 17, 19, 7, 8],
 [12, 17, 5, 3, 22, 20, 8, 19]]

Target IDs:
[[17, 5, 3, 17, 19, 7, 8, 19],
 [17, 5, 3, 22, 20, 8, 19, 20]]
```

Both arrays have shape **2 × 8**. The implemented model produces **2 × 8 × 25 logits**, one 25-class score vector for each of 16 target positions. With width 32, embedding lookup produces **2 × 8 × 32**. The inspection command verifies these shapes. Attention remains causal within each example and does not mix examples. The saved JSON fixture describes data only; the script performs the model calculations.

Reproduce the fixture by sorting the training character set, enumerating it into IDs, taking the specified nine-character slices, and encoding each slice's `[:-1]` and `[1:]` portions. The full vocabulary is saved in the JSON so this mapping is inspectable.

## Read and run it 🔧

From the repository root, install the locked optional experiment group and inspect the first batch:

```bash
uv sync --locked --group experiments
uv run --locked --group experiments python experiments/character-logs/train.py --inspect-only
```

Then run the small training demonstration:

```bash
uv run --locked --group experiments python experiments/character-logs/train.py --epochs 10 --output experiments/character-logs/artifacts/my-run
```

Use a fresh output directory for each run. Outputs are local and ignored by Git: `run.json` records losses and generated samples, `loss.png` plots both curves, and `best.pt` stores the best validation checkpoint. The checkpoint includes model weights, architecture configuration, and vocabulary for inference; it does **not** include optimizer/RNG state for resuming training.

Read the implementation in this order:

1. [data.py](data.py): training-only character vocabulary and shifted windows within each sequence.
2. [model.py](model.py): explicit Q/K/V, causal mask, softmax, value mixing, residual additions, and feedforward layers.
3. [train.py](train.py): cross-entropy, optimizer updates, full-split validation, checkpoint selection, and generation.

The model has one block, one head, width 32, feedforward width 64, and a 32-character context. It has **11,104 learned parameters**. Unlike our simplified arithmetic, it includes learned position embeddings added to token embeddings. The two LayerNorm modules have separate scale/shift parameters. We omit dropout for this first experiment. All parameters train from initialization using AdamW; no pretrained weights are used.

The initial inspection uses our two eight-character examples. Training then uses all stride-one, 32-character windows from the four training sequences, shuffled in batches of 16. After each epoch, both curves evaluate the whole respective split with fixed weights and no gradients. Loss is averaged by target count, including the smaller final batch. Validation examples never update parameters.

Generation starts with `job=`, greedily selects characters, and stops after 120 new characters. There is no end token. For longer sequences it uses only the last 32 characters and resets position indices within that window; KV caching is not implemented.

## First demonstration: observed, not a benchmark

The assistant ran on CPU with one PyTorch thread, seed 7, learning rate 0.003, AdamW weight decay 0.01, and PyTorch 2.14.1. Dependencies are resolved in the repository's uv lockfile. No timing or GPU comparison was measured.

| Checkpoint | Training loss | Validation loss |
| --- | --- | --- |
| Untrained | 3.3557 | 3.3809 |
| Epoch 9, best validation | 0.7870 | 1.9128 |
| Epoch 10 | 0.6789 | 1.9296 |

Loss decreased, but greedy generated logs remained malformed. One late validation increase is not enough to establish a reliable overfitting trend. The tiny, shared-template dataset is a demonstration of mechanics and limited held-out prediction, not proof of language or workflow understanding. Results may differ across software versions and hardware. Local details are in `artifacts/first-run/run.json`.

## Verification and next question

```bash
uv run --locked --group experiments ruff check experiments/character-logs
uv run --locked --group experiments ruff format --check experiments/character-logs
uv run --locked --group experiments python -m unittest discover -s experiments/character-logs -p 'test_*.py' -v
```

Six behavior checks cover fixture alignment, sequence boundaries, future-token isolation, independence between batch examples, validation leaving weights unchanged, and gradients/updates reaching embeddings, attention, normalization, and feedforward parameters.

Resume by running `--inspect-only` and following `job=orde` → `ob=order` through IDs, embeddings, and logits, one step at a time. Then read `TransformerBlock.forward` together and connect its two residual additions to our diagram, before following the training loop. Sezai has already correctly identified `o` as the next-character target at `=` in the first inspected example; implementation understanding has not yet been checked.

The implementation was written for this preparation experiment using standard PyTorch operations. References for the shared concepts: [author's GPT implementation](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py) and [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
