# A first Cosmopedia training check 🌱

After finishing the character-log preparation cycle, Sezai chose to try a larger public dataset immediately. This is a separate experiment; starting the book is still possible alongside it. The assistant implemented and ran the initial check, not Sezai.

[Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) contains synthetic educational text, including explanations, tutorials, and stories. We sampled its text and reused the SmolLM tokenizer. **All model weights were initialized randomly**; using a pretrained tokenizer does not import a pretrained model's language ability.

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

The next proposed user run is 1,000 updates from scratch, using the same prepared data. It may improve text quality; there is no promise of conversation or fluency. Synthetic text can contain errors, near duplicates are not removed, one seed is not a benchmark, and validation selected the checkpoint rather than serving as an untouched final test set.

Track this separate experiment in [issue #12](https://github.com/kantarcise/learning-llms/issues/12). The completed character-log cycle remains documented in the [training comparison](training-comparison.md).
