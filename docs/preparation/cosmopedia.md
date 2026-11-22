# A first Cosmopedia training check 🌱

After finishing the character-log preparation cycle, Sezai chose to try a larger public dataset immediately. This is a separate experiment; starting the book is still possible alongside it. The assistant implemented and ran the initial check, not Sezai.

[Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) contains synthetic educational text, including explanations, tutorials, and stories. We sampled its text and reused the SmolLM tokenizer. **All model weights were initialized randomly**; using a pretrained tokenizer does not import a pretrained model's language ability.

## Model architecture, drawn from the code 🧱

<figure markdown="1">

[![Cosmopedia architecture: embeddings, transformer blocks, final normalization, vocabulary logits, and an expanded block showing both residual shortcuts.](../assets/model-architectures/cosmopedia.svg)](../assets/model-architectures/cosmopedia.svg)

<figcaption markdown="span">Open the drawing for a full-size view. The three-dimensional blocks are schematic, like a CNN architecture drawing: their physical sizes do not measure parameter counts. They represent computation stages, not image height, width, or channels.</figcaption>
</figure>

This figure follows the default architecture in [`experiments/cosmopedia/model.py`](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/model.py). The upper row shows the whole model; the middle row opens one transformer block. The lower panels distinguish stored weights from the computed vectors flowing through the model.

For four full training windows, IDs `[4,256]` become vectors `[4,256,256]`, then logits `[4,256,49152]`. The two occurrences of 256 in the representation mean different things: input positions and coordinates per token. Four attention heads each use 64 coordinates; their outputs are concatenated back to width 256.

**Shape reminder:** `B` is the number of batch examples, `T` is input positions per example, and the final representation axis is vector width. Logits replace that width with the number of vocabulary candidates. Each position predicts the **next** token. Both residual additions preserve the representation shape. These diagrams end at logits: softmax and token selection are not part of these models' `forward` methods.

## From our tiny model to this one

| Component | Character-log model | Cosmopedia model |
| --- | ---: | ---: |
| Vocabulary | 25 characters | 49,152 subword entries |
| Maximum context | 24 characters | 256 tokens |
| Vector width | 32 | 256 |
| Transformer blocks | 1 | 4 |
| Attention heads per block | 1 | 4 |
| Feedforward width | 64 | 1,024 |
| Learned parameters | 10,848 | 15,803,904 |

The forward path is still token/position lookup → transformer blocks → final normalization → vocabulary scores. Each block still has attention and feedforward sublayers with normalization and residual additions. Context length and vector width are independent even though both happen to be 256 in this configuration.

Four attention heads arrange each token's 256 projected coordinates into four groups of 64, compute a separate attention mixture per group, and join the results back into 256 coordinates. We have implemented this, but have not yet walked through the new mechanism together. The input embedding and output projection share a single learned weight table to avoid duplicating the large vocabulary matrix.

## Data and first measured result

We prepared 1,000,000 training tokens across 1,393 documents, plus 65,536 validation tokens across 89 documents. Documents are assigned to splits before windowing, exact duplicate text is removed, and windows do not cross document boundaries. Dataset and tokenizer revisions and document hashes are recorded. Only a bounded source-order sample was used, not the entire corpus.

The assistant ran 100 optimizer updates with batches of four windows on the Mac GPU (MPS). These are randomly sampled windows with replacement, so 100 steps does not mean 100 epochs. Each window has up to 256 positions; padding targets are excluded from loss.

| Update | Recent training loss | Fixed-sample validation loss |
| --- | ---: | ---: |
| 0 | — | 10.8452 |
| 25 | 9.8804 | 9.0234 |
| 50 | 8.3737 | 8.0052 |
| 75 | 7.7016 | 7.7779 |
| 100 | 7.6274 | 7.6613 |

The training loop took about 12.8 seconds including periodic evaluation and checkpoint saving. Setup, download, initial evaluation, and generation are excluded. This measures this particular machine and run, not expected performance elsewhere.

The best-checkpoint continuation began `Once upon a time, and the the the ...` and remained repetitive. Falling loss verifies learning progress on this check; it does not show fluent language. The run saw only about 86,000 training targets with replacement. Validation uses 32 fixed sampled windows, not the full validation set. Training loss averages recent batches at changing weights, so it differs from the full-split evaluation procedure in the character-log experiment.

## Next: run longer and inspect, not assume

The [experiment README](https://github.com/kantarcise/learning-llms/tree/main/experiments/cosmopedia) contains runnable download/training commands, exact revisions, shapes, and limitations. The [initial report](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/results/smoke-100steps/run.json) preserves both generated samples verbatim. Local checkpoints and downloaded data are excluded from Git.

The proposed 1,000-update user run has now been completed; see the follow-up below. It may improve text quality; there is no promise of conversation or fluency. Synthetic text can contain errors, near duplicates are not removed, one seed is not a benchmark, and validation selected the checkpoint rather than serving as an untouched final test set.

Track this separate experiment in [issue #12](https://github.com/kantarcise/learning-llms/issues/12). The completed character-log cycle remains documented in the [training comparison](training-comparison.md).


## Data sizes and the user-run follow-up

The character-log experiment used **563 bytes of text**: 372 training bytes and 191 validation bytes across six sequences. Its JSON file is 1,037 bytes including structure and metadata.

Our selected Cosmopedia source documents contain **5,581,797 UTF-8 text bytes (5.58 MB)**: 5,234,080 training bytes and 347,717 validation bytes. The final document in each split may be truncated for tokenization. Prepared data, including integer input/target tensors, tokenizer, and manifest, totals **29,481,912 bytes (29.48 MB)**. Checkpoints, caches, and verification copies are separate. These are measured logical file sizes, not network traffic. The [experiment README](https://github.com/kantarcise/learning-llms/tree/main/experiments/cosmopedia) includes the breakdown and measurement record.

Sezai has now run **1,000 updates** himself. Validation loss reached **6.3611**, with the best checkpoint at step 1,000. The loop took 96.2 seconds and processed 866,275 training targets with repeated sampling. Output improved to repeated phrase fragments such as `I was a time`, but remained incoherent. The [saved report](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/results/user-1000steps/run.json) preserves the actual output.

**Current handoff:** before a multi-hour Linux/3090 experiment, implement resumable latest/best checkpoints and benchmark that machine. Existing weights do not preserve AdamW or sampling state. This additional experiment can proceed alongside Chapter 1; it is not a new prerequisite for the book.

Track the bounded Linux run in [issue #14](https://github.com/kantarcise/learning-llms/issues/14).

## KV caching on the M5 Air

Suppose eight prompt tokens have just produced token nine. In each of our four transformer blocks, each head has already calculated a key and a value for every prompt position. Causal attention lets us keep them: appending token nine cannot change what an earlier position saw. **Prefill** is the first call that processes the prompt and selects token nine. **Decode** starts when we feed token nine back in to select token ten.

| During that first decode step | Full-prefix generation | KV-cached generation |
| --- | --- | --- |
| Input to the model | Positions 1–9 | Position 9 |
| Keys and values calculated per head | Nine of each | One of each |
| Positions visible to the newest query | All nine | All nine |

Each block’s key and value tensors grow from `[1, 4, 8, 64]` to `[1, 4, 9, 64]` (one request, four heads, processed positions, 64 numbers per head). The new query still reads keys and values across the prefix. Keeping them saves the repeated projections **and the work of passing the older positions through the blocks again**.

### What the Mac actually saved

We used the existing 1,000-step checkpoint on the M5 MacBook Air GPU (PyTorch MPS, float32), with a single request and fixed budgets of 32 greedy output tokens. The prompts are prefixes of the same validation window. Both paths project only the last position into vocabulary scores. The timing includes greedy selection and dynamic cache appends. We warmed both paths, synchronized the GPU around each timed phase, and alternated run order across six repetitions. Each decode measurement averages the 31 steps after prefill.

| Prompt tokens | Prefill ms, uncached / cached (median) | Uncached decode ms/token, median (range) | Cached decode ms/token, median (range) | Decode speedup | Final cache |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.26 / 1.19 | 1.034 (1.023–1.106) | 0.949 (0.946–0.957) | 1.09× | 312 KiB |
| 64 | 1.43 / 1.32 | 1.167 (1.161–1.238) | 0.965 (0.937–0.986) | 1.21× | 760 KiB |
| 192 | 1.71 / 1.74 | 1.607 (1.603–1.612) | 0.988 (0.985–0.994) | 1.63× | 1,784 KiB |

The first prediction still requires reading the prompt. The saving appears during repeated decoding and grew with prompt length in this run. Across all 96 predictions, both paths picked identical IDs. We also checked every vocabulary logit against the original full-output forward pass. The largest absolute difference was below 0.000009 (absolute and relative tolerances each 0.0001).

For this model, keeping one position costs **8 KiB**: two tensors × four blocks × four heads × 64 numbers × four bytes. At the end of the longest trial the cache holds 192 + 31 = 223 positions. The 32nd output was picked but never fed back in. These sizes count the logical K/V tensor contents. Peak device memory also includes weights, temporary tensors, allocator overhead, and copies made when appending. The results cover one model and one request at a time within its 256-position limit.

[Runnable benchmark](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/benchmark_cache.py) · [Individual timings, prompt IDs, and checkpoint fingerprint](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/results/kv-cache/m5-mps.json)

### What changed on the RTX 3090?

KV caching took slightly longer per decoded token in all five workloads on the RTX 3090. The table reports median decode time for a single request in float32 with PyTorch 2.14.1+cu126. Prefill produces the first output, leaving 31 timed decode steps for the short continuations and 191 for the long ones.

| Prompt tokens | Output tokens | Repeats | Uncached ms/token | Cached ms/token | Uncached / cached |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 6 | 0.928 | 0.963 | 0.964× |
| 64 | 32 | 6 | 0.937 | 0.962 | 0.975× |
| 192 | 32 | 6 | 0.954 | 0.968 | 0.985× |
| 8 | 192 | 20 | 1.020 | 1.036 | 0.984× |
| 64 | 192 | 20 | 0.971 | 0.989 | 0.982× |

The 32-output workload used the same prompt IDs and model configuration as the Mac run. The checkpoint fingerprints differ. The 3090 used the best weights from the local 100-step monitored run, while the Mac used the 1,000-step checkpoint. Each ratio compares two paths using identical weights on the 3090. Both paths matched the reference greedy ID at all 480 predictions across the five workloads. The largest absolute logit difference was 0.0000062.

[32-output timings](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/results/kv-cache/3090-matched.json) · [192-output timings](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/results/kv-cache/3090-long.json)

### What happens as requests arrive faster?

At 250 offered requests per second, the RTX 3090 produced about 7,000 output tokens/s with either path. A user waiting for the first token would still see a difference. With the cache, the 95th-percentile wait fell from 131 ms to 103 ms. At 500 requests/s, caching shortened that wait by about half and raised completed output throughput by 33%.

| Offered requests/s | p95 queue ms | p95 first-token ms | p95 inter-token ms | Output tokens/s | Peak cached K/V MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 69 / 67 | 71 / 68 | 1.43 / 1.36 | 798 / 798 | 5.23 |
| 100 | 96 / 100 | 98 / 101 | 1.45 / 1.42 | 3,037 / 3,057 | 12.20 |
| 250 | 129 / 102 | 131 / 103 | 2.35 / 1.38 | 6,979 / 6,994 | 20.91 |
| 500 | 296 / 143 | 298 / 145 | 2.79 / 1.36 | 8,653 / 11,531 | 27.88 |
| 1,000 | 431 / 256 | 432 / 257 | 2.81 / 1.40 | 8,969 / 12,170 | 27.88 |

Pairs show uncached / cached. Each entry averages two replays, except memory, which takes the larger peak logical cache size. Each replay had 192 requests with mixed 8-, 64-, and 192-token prompts, each generating 32 tokens. All 20 replays produced identical greedy IDs per request. Throughput includes time to drain the queue after the last arrival.

The worker admitted batches of up to 16 waiting requests with equal prompt length and finished each batch before admitting another. This policy explains why first-token waits can be much longer than the gaps between streamed tokens. At 1,000 requests/s the offered work was 32,000 output tokens/s, well above either observed completion rate. These finite bursts show how a queue develops under this policy. Estimating sustainable service capacity would require longer steady traffic and a target first-token and inter-token latency.

[Runnable arrival benchmark and methodology](https://github.com/kantarcise/learning-llms/tree/main/experiments/cosmopedia#replay-mixed-prompt-lengths-under-load-on-cuda) · [Trace and per-replay results](https://github.com/kantarcise/learning-llms/blob/main/experiments/cosmopedia/results/kv-cache/3090-arrivals.json)

### Whose memory does the cache use?

The RTX 3090 stores the CUDA cache tensors in its dedicated VRAM. On the M5 Air, the MPS tensors draw from the same physical memory pool as the CPU. The model weights are shared across requests on either machine. Each active sequence has its own keys and values. [Apple’s unified memory model](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)

Suppose 16 clients each send a 192-token prompt and receive 32 tokens. Just before selecting the final token, each cache holds 223 positions. Across the batch, that is **16 × 223 × 8 KiB = 27.875 MiB**, the largest logical cache we observed. The other requests waiting in the queue have no K/V tensors yet. Our worker releases the batch’s cache when generation finishes. Appending with `torch.cat` temporarily needs room for both old and new tensors, so a server also needs headroom above the logical total.

A practical admission rule would budget the sum of prompt tokens and allowed continuation tokens across active requests. An idle conversation presents a different choice. Keeping its token IDs lets a server release the cache and reconstruct it exactly on the next turn by running prefill again. Retaining its cache saves that prefill if the client returns, at the cost of memory that could serve another request.

### How does this relate to Claude’s prompt caching and chat compaction?

Our benchmark reuses a prefix **within one answer**. Anthropic’s prompt caching extends that idea **across API calls**. For example, if a 20,000-token conversation prefix gets a full cache hit and the user adds 50 tokens, the earlier 20,000 are billed at the cache-read rate and the 50 new tokens still need processing. All 20,050 tokens occupy context. Anthropic documents charges for writing and reading cached tokens, with a retention period for each entry. The published API prices do not establish a physical memory cost per token for Claude. [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) · [Cache pricing](https://platform.claude.com/docs/en/about-claude/pricing#prompt-caching)

Compaction changes what the model reads. Replacing 20,000 tokens of older turns with a 2,000-token summary frees 18,000 context positions for future turns. Generating the summary has an upfront cost, its tokens need prefill, and a detail left out may matter later. An unchanged prefix before the summary can still qualify for a cache hit. A server can evict idle K/V tensors to reclaim device memory. It can independently compact a conversation when the remaining context space or expected savings justify a shorter history. [Anthropic’s compaction overview](https://platform.claude.com/docs/en/build-with-claude/compaction)
