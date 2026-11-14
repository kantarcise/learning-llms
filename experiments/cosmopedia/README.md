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

### Linux RTX 3090 environment

For [issue #51](https://github.com/kantarcise/learning-llms/issues/51), Linux x86-64 uses the official **PyTorch 2.14.1 CUDA 12.6 build** through the explicit `pytorch-cu126` uv index. Install the locked environment with:

```bash
uv sync --locked --group experiments --group cosmopedia
```

The previous lock selected PyTorch's CUDA 13 dependencies from PyPI; CUDA 13 was not an experiment requirement. The local RTX 3090 host has NVIDIA driver 560.35.03, which supports CUDA 12. CUDA 13 requires driver 580 or newer. Selecting the CUDA 12.6 build keeps the same PyTorch release and works with the installed driver. See [PyTorch's published builds](https://pytorch.org/get-started/previous-versions/) and [NVIDIA's driver compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/latest/595/minor-version-compatibility.html).

The CUDA version shown by `nvidia-smi` describes driver support; `torch.version.cuda` identifies the runtime used to build PyTorch. Installing this wheel supplies its runtime libraries in the Python environment. It does not upgrade the system driver. macOS continues to resolve PyTorch from PyPI and use MPS when available.

The November 14 Linux [preflight record](results/rtx3090-preflight.json) captures the starting commit, hardware/software inventory, a successful CUDA tensor operation, and one synthetic update of the full 15.8M-parameter model. The synthetic check used random token IDs; it is not a Cosmopedia baseline.

The sample was then regenerated on Linux using the pinned revisions above. Its manifest matches the original byte for byte, including the ordered document hashes: 1,393 training documents and 89 validation documents. Verification recomputed every text hash, checked split assignment and deduplication, and retokenized every selected document to reconstruct and compare all saved input/target tensors. The five prepared files total **29,481,912 bytes**, matching the original file sizes. The [data verification record](results/rtx3090-data-verification.json) includes file SHA256 fingerprints for later transfers and the 32 fixed validation-window indices selected by the existing training script with seed 7. Original tensor-file hashes were not recorded, so the new fingerprints identify this regenerated copy.

### Training commands

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

The output directory holds `run.json`, resumable `latest.pt` and `best.pt`, and inference-only `best-weights.pt`. Generation uses the weights with the lowest measured validation loss, including step 0 if nothing improves. Historical Mac and initial CUDA runs below predate resumable checkpoints; their original `best.pt` files remain weights-only.

### Stop and resume

For a graph that updates during training, a Stop & save button, and a continuous curve across resumed runs, see [Live loss, stop and resume](LIVE_TRAINING.md). That guide also records the 10M-token longer-run setup and its completed 1,000-update calibration.

`--steps` is the **total target update count**, including updates already completed. Use a fresh output directory for each invocation. For example, stop at update 37 of a planned 100-update run, then continue to 100:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/train.py \
  --device cuda --steps 100 --stop-after 37 --eval-every 25 \
  --output experiments/cosmopedia/artifacts/resume-part

uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/train.py \
  --device cuda --steps 100 --eval-every 25 \
  --resume experiments/cosmopedia/artifacts/resume-part/latest.pt \
  --output experiments/cosmopedia/artifacts/resume-continued
```

The trainer saves `latest.pt` every `--checkpoint-every` updates (default 25) and at the stopping update. It saves `best.pt` when validation improves. Both snapshots contain model weights, AdamW state, completed updates, Torch CPU/active-device RNG state, the separate batch-sampler RNG, validation indices, loss history, best result, and partial interval loss/token totals. A latest snapshot also embeds the historical best snapshot, so it can be transferred alone and still recreate both files. This uses additional disk space to make each continuation self-contained. Model and optimizer restoration follow [PyTorch's general checkpoint pattern](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Resume checks architecture, actual data/tokenizer/manifest file hashes, seed, batch size, optimizer settings, evaluation interval/selection, device type and PyTorch version before creating an output directory. Keep these settings unchanged. Old inference-only files are rejected: loading those weights with a fresh optimizer would be a **warm start**, not a resume; this CLI does not offer a warm-start mode.

`--stop-after` stops after a completed update without adding an evaluation at that boundary. This preserves the planned evaluation schedule and unfinished reporting interval. To reproduce a planned run exactly, keep the same `--steps` on both invocations and use `--stop-after` for the earlier stop. Extending an already completed run can introduce an extra evaluation at its previous final update. The report's loss history and token total are cumulative; elapsed time, throughput and GPU-memory peaks cover only the current invocation. Full checkpoint I/O is included in loop timing, so these timings are not directly comparable to the earlier weights-only baseline.

Snapshots are written to a temporary sibling and atomically replace the destination after serialization succeeds. A failed write leaves the previous snapshot intact. After an unexpected process exit, resume the last successfully saved update; unsaved updates must be repeated. This does not promise durability against power loss. Exact continuation is verified on the same software/device stack, not across CUDA, MPS and CPU.

Run the fresh-process comparison on prepared data with:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/verify_resume.py \
  --device cuda --steps 100 --interrupt-after 37 --eval-every 25 \
  --output experiments/cosmopedia/artifacts/resume-verification
```

This runs 100 uninterrupted updates, then 37 updates plus a fresh-process continuation to 100. It requires exact equality of latest/best checkpoint tensors and metadata, including AdamW, RNG/sampler and reporting state, and compares generated samples and loss history. Checkpoints/logs stay in ignored artifacts; `verification.json` records results and source fingerprints. Offline tests exercise the real CLI with synthetic data, changed-data/settings rejection, RNG restoration, resuming best snapshots, and failed-write preservation.

**Actual RTX 3090 verification (November 14):** the 100-update run and 37 + 63-update continuation matched exactly across **358 tensors** in the latest snapshots, all other checkpoint metadata, the separate best snapshots, loss history and generated samples. Validation again ended at **7.6613**. The [verification report](results/rtx3090-resume/verification.json) contains all three segment reports and source fingerprints; [provenance](results/rtx3090-resume/provenance.json) records the invocation. This verifies a graceful stop and fresh-process continuation on this CUDA stack, not an abrupt machine failure.

The uninterrupted loop took **56.9 seconds** with full checkpoint saving, compared with the earlier 9.82-second weights-only baseline. The interrupted and resumed segments took 17.4 and 57.6 seconds respectively. These timings include substantial snapshot I/O and are not isolated GPU-compute measurements. Measure the intended checkpoint cadence again before setting the two-hour update budget.

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

## RTX 3090 baseline: 100 updates

On November 14, the assistant ran the same 100-update workload on the Linux RTX 3090 using PyTorch 2.14.1+cu126. Seed 7, batch size 4, context 256, learning rate 0.0003, the data-manifest hash, and all 32 validation indices match the original MPS check. The run processed the same **85,926 non-padding training targets**, sampled with replacement.

| Step | Recent training loss | Validation loss |
| --- | ---: | ---: |
| 0 | — | 10.8452 |
| 25 | 9.8804 | 9.0234 |
| 50 | 8.3737 | 8.0052 |
| 75 | 7.7016 | 7.7780 |
| 100 | 7.6274 | 7.6613 |

The synchronized training loop took **9.82 seconds**, or approximately **8,750 non-padding target tokens/second**, including periodic validation and checkpoint saving. Download, setup, initial evaluation and generated samples are outside that timer. Peak PyTorch allocated GPU memory was **840.1 MiB**, with **902 MiB** reserved, across setup, training, evaluation and generation. These allocator measurements exclude other processes and non-allocator GPU memory; the desktop shared the GPU during the run.

The best checkpoint was step 100. The fixed-prompt continuation again began `Once upon a time, and the the the ...`; lower loss still does not demonstrate fluent generation. CUDA and MPS losses were close and the generated strings matched in this check. This is one short configuration-specific observation, not a general hardware ranking or a steady-state training benchmark.

The unchanged [CUDA run report](results/rtx3090-100steps/run.json) preserves full samples, losses, timing and memory measurements. The [provenance record](results/rtx3090-100steps/provenance.json) records the command, base commit and source/lockfile hashes. The inference-only checkpoint remains under ignored `artifacts/rtx3090-100steps-20261114/`. The resume workflow above addresses the checkpoint prerequisite from [issue #14](https://github.com/kantarcise/learning-llms/issues/14); the longer training experiment remains separate.

## How much data did we actually use? 📦

Measured on October 16 from the local files. **MB below means 1,000,000 bytes.** Source-text size counts only UTF-8 bytes of the text fields; file sizes also include formatting and metadata. These are logical byte sizes, not filesystem allocated blocks.

| Measurement | Character logs | Cosmopedia sample |
| --- | ---: | ---: |
| Training documents | 4 | 1,393 |
| Validation documents | 2 | 89 |
| Training source text | 372 bytes | 5,234,080 bytes (5.23 MB) |
| Validation source text | 191 bytes | 347,717 bytes (0.35 MB) |
| Total source text | **563 bytes** | **5,581,797 bytes (5.58 MB)** |
| Source JSON/JSONL file | 1,037 bytes | 5,828,130 bytes |

The Cosmopedia source-text count includes each full selected document. Only the final selected document in each split may be truncated to fit the token budget. The actual prepared budgets are **1,000,000 training tokens and 65,536 validation tokens**, including EOS markers. Character-log tokens are individual characters; Cosmopedia tokens are subword units, so token counts are not a like-for-like text-size comparison. Overlapping character-log windows also reuse the same small source text.

Cosmopedia's prepared folder totals **29,481,912 bytes (29.48 MB, approximately 28.12 MiB)**:

| Prepared file | Bytes |
| --- | ---: |
| Selected documents (`documents.jsonl`) | 5,828,130 |
| Training inputs and targets (`train.pt`) | 18,790,101 |
| Validation inputs and targets (`validation.pt`) | 1,230,717 |
| Tokenizer (`tokenizer.json`) | 3,522,656 |
| Provenance (`manifest.json`) | 110,308 |

Token tensors store integer inputs and targets plus padding, which explains why they occupy more space than the source text. These totals exclude model checkpoints, verification copies, library caches, and the Python environment. They are not a measurement of network bytes transferred. The earlier roughly 29 MB disk-usage answer was rounded filesystem allocation; this table uses exact file lengths.

The complete Cosmopedia v2 download is approximately 122 GB according to the [dataset metadata](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/main/README.md). We did not download the full corpus. Exact local measurements are saved in [data-sizes.json](results/data-sizes.json).

## Sezai's 1,000-update run

Sezai subsequently ran 1,000 updates on MPS. Validation loss fell from 10.8452 initially to 7.6613 at step 100 and **6.3611 at step 1,000**; all recorded validation points improved. The loop took 96.2 seconds and processed **866,275 non-padding training targets**, sampled with replacement. This is not one complete epoch, and it does not mean 866,275 distinct tokens were visited.

The best checkpoint was step 1,000. Its continuation included `Once upon a time, I was a time, I was a time` and continued repeating. Recognizable phrase fragments improved, but coherent sustained generation has not been demonstrated. The unchanged [user run report](results/user-1000steps/run.json) contains the full samples and all measurements.

The step-100 recent-training loss differs from the 100-step check because that check reported 25-step intervals, while this run reports 100-step intervals. Step-100 validation loss matches. Validation still uses only 32 fixed windows.

**Historical continuation limitation:** the Mac's original `best.pt` files save weights only. They cannot recover the optimizer and RNG state of those runs. The current trainer saves versioned resumable snapshots as described above; a command without `--resume` still initializes randomly.

Linux follow-up: [issue #14](https://github.com/kantarcise/learning-llms/issues/14) tracks checkpoint support, resource checks, and a bounded two-hour experiment. Book reading can proceed now.

## Completed longer run

The 10M-token experiment completed 500,000 updates. Validation was best at update 89,000 (4.0308) and worsened to 4.3837 by the final update while training loss kept falling. See the [complete loss curves, power/activity graphs and retained-checkpoint verification](results/long-10m-run/README.md).

## Checks

For continuous GPU/CPU telemetry, phase-aligned graphs, idle/cooldown capture and an optional one-step kernel trace, see [Observe a training run](MONITORING.md). The wrapper saves raw samples and a portable report using the existing experiment dependencies.

```bash
uv run --locked --group experiments --group cosmopedia ruff check experiments/cosmopedia
uv run --locked --group experiments --group cosmopedia ruff format --check experiments/cosmopedia
uv run --locked --group experiments --group cosmopedia python -m unittest discover -s experiments/cosmopedia -p 'test_*.py' -v
```

Tests are offline: changing future tokens or another batch example cannot affect an earlier prediction; padding and EOS targets are handled correctly; gradients are finite and parameter updates occur; context/head constraints are checked. Actual prepared-document hashes were also checked for disjoint splits and exact deduplication.
