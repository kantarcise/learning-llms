# A small Cosmopedia language model 🌱

This separate experiment trains **randomly initialized weights** on a bounded sample of [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), synthetic educational text in the SmolLM corpus. It reuses the [SmolLM-135M tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM-135M), not that model's pretrained weights. See [issue #12](https://github.com/kantarcise/learning-llms/issues/12).

The character-log experiment remains unchanged. This is an additional experiment, not a prerequisite for starting the book.

## Read the code in this order

1. `prepare.py::windows`: how one document becomes input/next-token target windows, including right padding.
2. `model.py::LanguageModel.forward`: token and position lookup, four blocks, final normalization, vocabulary scores.
3. `model.py::Attention.forward`: Q/K/V, four heads, causal attention, concatenate, output projection.
4. `train.py::loss_for`, then the training loop: clear gradients → loss → backward → clip gradients → AdamW update.

| Architecture choice | Value |
| --- | ---: |
| Vocabulary entries | 49,152 |
| Context positions | 256 subword tokens |
| Numbers per token vector | 256 |
| Transformer blocks | 4 |
| Attention heads per block | 4 |
| Numbers per head | 64 |
| Feedforward | 256 → 1,024 → 256, GELU |
| Unique learned parameters | 15,803,904 |

The context and width happen to both be 256; they remain independent hyperparameters. Input IDs `[B,T]` become vectors `[B,T,256]`, then scores `[B,T,49152]`. B is batch examples and T is input positions. A token may be a word fragment, a word, or punctuation; it is no longer necessarily one character.

Each attention head sees its own learned Q/K/V coordinates. We reshape a 256-coordinate projection into four groups of 64. Each group computes its own position mixing, and the groups are concatenated back to 256 coordinates. PyTorch's `scaled_dot_product_attention` computes the familiar scaled scores, causal masking, softmax, and weighted values. Heads are not manually assigned linguistic roles.

The token embedding and output projection **share** their 49,152-by-256 weight table (weight tying). Both lookup and output scoring train the same parameter. This saves a second 12,582,912-parameter table; vocabulary size determines rows and vector width determines coordinates per row.

## Prepare data

From the repository root:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/prepare.py
```

Default local destination: `experiments/cosmopedia/artifacts/data`. The directory must not already exist. The initial download is already prepared on this computer; do not rerun it unless choosing a fresh output directory.

This streams the dataset's `text` field, not its generation prompt. It collects one million training tokens and 65,536 validation tokens, including document end markers. Network transfers use Parquet ranges and can exceed the selected text size; this is a token budget, not a strict byte-download limit. No model weights are downloaded.

A SHA256 text hash assigns roughly 10% of documents to validation, before tokenization and windowing. Exact duplicate documents are removed across both splits; near duplicates are not detected. Sampling takes documents in stream order, not a representative random sample of the entire corpus. Validation is for checkpoint selection, not an untouched test set. Synthetic text can contain factual errors.

Each document ends with EOS. Windows never cross document boundaries. Short tail windows are padded on the right, and target padding uses `-100`, excluded from loss. The last selected document in each split can be truncated to fit the token budget, with an artificial end marker at that cutoff. This bounded-sample simplification affects at most those two documents.

The manifest records dataset/tokenizer commit revisions, selected text hashes, token counts, context length, and EOS ID. To reproduce the initial sample elsewhere, use the revisions in `results/smoke-100steps/data-manifest.json`:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/prepare.py \
  --dataset-revision 3ba9d605774198c5868892d7a8deda78031a781f \
  --tokenizer-revision 1d461723eec654e65efdc40cf49301c89c0c92f4
```

Raw documents, tokenizer files, token tensors, and checkpoints stay in ignored `artifacts/`. The committed manifest contains hashes and metadata, not downloaded text. Attribute Cosmopedia/SmolLM as above; the corpus card lists ODC-BY.

## Train

A short check (already completed by the assistant):

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/train.py \
  --steps 100 --eval-every 25 \
  --output "experiments/cosmopedia/artifacts/check-$(date +%Y%m%d-%H%M%S)"
```

The next user-run experiment:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/train.py \
  --steps 1000 --eval-every 100 \
  --output "experiments/cosmopedia/artifacts/run-1000steps-$(date +%Y%m%d-%H%M%S)"
```

Defaults: batch size 4, AdamW learning rate 0.0003, weight decay 0.01, gradient-norm clipping at 1.0, seed 7. The script selects CUDA, then MPS, then CPU; `--device cpu` overrides that choice. No mixed precision or scheduler is used in this small baseline.

**Steps are updates, not epochs.** Each step samples four windows with replacement. The 1,000-step run starts from scratch; it does not resume the saved checkpoint. Reported training loss averages recent sampled batches at changing weights. Validation measures the same 32 deterministically chosen windows at fixed weights. Therefore the curves have different measurement procedures; do not interpret their gap exactly like our full-split character-log curves. Increase `--validation-windows` to evaluate more windows when appropriate.

The output directory holds `run.json` and an inference-only `best.pt`. The checkpoint is just a state dictionary: use the report's config and prepared tokenizer to reconstruct it. It does not contain optimizer/RNG state for resuming. Generation uses the checkpoint with the lowest measured validation loss, including step 0 if nothing improves.

## Actual first check: 100 updates

The assistant ran this on the Mac's MPS device. Data: 1,393 training documents / 4,587 windows and 89 validation documents / 300 windows. Evaluation used 32 fixed validation windows.

| Step | Recent training loss | Validation loss |
| --- | ---: | ---: |
| 0 | — | 10.8452 |
| 25 | 9.8804 | 9.0234 |
| 50 | 8.3737 | 8.0052 |
| 75 | 7.7016 | 7.7779 |
| 100 | 7.6274 | 7.6613 |

The loop took approximately 12.8 seconds, including periodic validation and checkpoint saving, at roughly 6,698 non-padding training targets/second. Download, setup, initial evaluation, and generated samples are outside that timer. Hardware/software affect speed and numerical results; this is not a portable benchmark.

Generation began `Once upon a time, and the the the ...` and kept repeating. Loss decreased, but this short run does not demonstrate fluent language. It processed only about 86,000 training targets, sampled with replacement, from the million-token sample. The next run tests whether longer training helps; it is not promised to become conversational.

The full unmodified samples and measurements are in [run.json](results/smoke-100steps/run.json). Data provenance is in [data-manifest.json](results/smoke-100steps/data-manifest.json). Model weights are not committed. A tokenizer provides reusable text segmentation; it does not transfer the pretrained model's language ability.

## Checks

```bash
uv run --locked --group experiments --group cosmopedia ruff check experiments/cosmopedia
uv run --locked --group experiments --group cosmopedia ruff format --check experiments/cosmopedia
uv run --locked --group experiments --group cosmopedia python -m unittest discover -s experiments/cosmopedia -p 'test_*.py' -v
```

Tests are offline: changing future tokens or another batch example cannot affect an earlier prediction; padding and EOS targets are handled correctly; gradients are finite and parameter updates occur; context/head constraints are checked. Actual prepared-document hashes were also checked for disjoint splits and exact deduplication.
