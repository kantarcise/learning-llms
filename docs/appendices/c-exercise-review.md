# Appendix C — Checking our exercises

This review compares the companion’s recorded work with the printed solutions on pages 300–312. It covers all 21 numbered exercises in Chapters 2–7. Chapter 1 has no numbered exercise in Appendix C. Numerical comparisons below use existing results. The audit did not rerun training.

## Chapters 2–4 · From text to a GPT model

| Exercise | What we checked against Appendix C |
| --- | --- |
| [2.1 · BPE](../chapters/02-working-with-text-data.md#exercise-21-inspect-each-piece) | The six token IDs agree exactly. Decoding them together recovers the original string, including the space. |
| [2.2 · Windows](../chapters/02-working-with-text-data.md#exercise-22-window-length-stride-and-overlap) | Same batch size of four, stride of two, and lengths of two and eight. Our guided toy example gives the same overlap and target shift. An independent run on the story is unrecorded. |
| [3.1 · Projection weights](../chapters/03-coding-attention-mechanisms.md#exercise-31-comparing-selfattentionv1-and-selfattention_v2) | Both solutions account for the transpose between a manually stored projection and a PyTorch linear layer before comparing outputs. |
| [3.2 · Two output coordinates](../chapters/03-coding-attention-mechanisms.md#exercise-32-returning-two-dimensional-embedding-vectors) | Two wrapper heads each return one coordinate. Concatenating them gives width two. |
| [3.3 · GPT-2 attention](../chapters/03-coding-attention-mechanisms.md#exercise-33-initializing-gpt-2-size-attention-modules) | The Small configuration agrees: context 1,024, width 768, 12 heads, 64 coordinates per head. |
| [4.1 · Parameter counts](../chapters/04-implementing-a-gpt-model.md#exercise-41-feedforward-versus-multi-head-attention) | Exact agreement: 4,722,432 feedforward parameters and 2,360,064 attention parameters per block. |
| [4.2 · Larger models](../chapters/04-implementing-a-gpt-model.md#exercise-42-initialize-the-larger-gpt-2-configurations) | Our four configurations use shape-only construction to count parameters. XL agrees exactly at 1,637,792,000 untied and 1,557,380,800 if tied. |
| [4.3 · Dropout](../chapters/04-implementing-a-gpt-model.md#exercise-43-give-each-dropout-site-its-own-setting) | Same three independent controls for embedding dropout, attention-weight dropout, and residual-branch dropout. Our distinct settings demonstrate the wiring. |

## Chapter 5 · Training and generation

| Exercise | Comparison |
| --- | --- |
| 5.1 · Sampling frequency | At temperature 5, both report 32 pizza draws in 1,000. The probability at index 6 is about 4.3%. The empirical count and underlying probability answer different questions. |
| 5.2 · Decoding settings | The companion records three continuations from one trained model and connects restricted sampling with predictable output and broader sampling with brainstorming. Appendix C gives the same trade-off. |
| 5.3 · Greedy decoding | Both give two routes: the explicit greedy branch or a single eligible token. Our threshold-based top-k implementation also documents what happens with tied maximum logits. |
| 5.4 · Resume training | Both restore model and optimizer. Our fresh-process check reproduced the prompt logits and advanced AdamW’s counters from 90 to 99 during the extra epoch. |
| 5.5 · Pretrained loss | Our measured train/validation losses were 3.75473 and 3.55960. Appendix C reports approximately 3.75475 and 3.55962. |
| 5.6 · Larger weights | The companion records generated text from both pretrained Small and XL. Appendix C gives the model-size and checkpoint switch. There is no prescribed output sentence. |

[Chapter 5 code and recorded runs](../chapters/05-pretraining-on-unlabeled-data.md)

## Chapter 6 · SMS classification

| Exercise | Our evidence | Appendix C reference |
| --- | --- | --- |
| 6.1 · Increase context | Checked 1,024-wide inputs, preserved labels, and padding. The accuracy comparison at the longer width remains open. | Test accuracy fell from 95.67% to 78.33% in the book’s run. |
| 6.2 · Unfreeze all layers | Our one-epoch Mac test accuracy rose from 85.0% to 93.3%. | 95.67% to 96.67%. Both experiments favor full fine-tuning here. Their training budgets differ. |
| 6.3 · Read the first position | Our one-epoch Mac test accuracy fell from 85.0% to 65.7% when we switched from last to first. | 95.67% to 75.00%. The earlier position has access to less of the message. |

[Chapter 6 setup and comparison tables](../chapters/06-finetuning-for-text-classification.md)

## Chapter 7 · RTX 3090 runs versus Appendix C

All four Chapter 7 exercises ran on the Linux RTX 3090 from the same pretrained GPT-2 Medium checkpoint and 85/10/5 split, scored by local `llama3:8b` (Ollama `0.20.0`, `Q4_0`, `temperature=0`, `seed=123`). Under that judge, the two-epoch Alpaca baseline scored **59.36/100** (`110/110`) with held-out answer cross-entropy **1.174** in **53.8 s** (8.28 GB peak VRAM). [Recorded RTX 3090 experiment metrics](../assets/chapter-07/rtx3090-experiments.json)

| Exercise | Our RTX 3090 evidence | Appendix C reference |
| --- | --- | --- |
| 7.1 · Prompt style | Mismatched templates at inference broke all 3 held-out spelling items (`0/3` across Phi-3, Mistral, Gemma 2/4, and Qwen 2.5/3.8). Training and evaluating with matching Phi-3 markers shortened mean training inputs by 22.9% (56.4 to 43.5 tokens), cut training time by 15.1% (45.7 s vs 53.8 s), reached answer cross-entropy **1.168**, and scored **57.11/100** (`109/110`; 56.77 with 1 prose reply). | Similar quality and about 17% faster training with shorter Phi-3 inputs. |
| 7.2 · Instruction masking | Masking the entire prompt including `\n\n### Response:\n` required supplying `### Response:\n` at inference (omitting it made the model emit `<|endoftext|>` immediately). With aligned prompts, two epochs took **52.7 s**, improved held-out answer cross-entropy from **1.174** to **1.133**, and scored **60.05/100** (`109/110`; 60.23 with 1 prose reply) — within judge variance of the 59.36 baseline. | The author’s judge score fell by about four points under his setup. |
| 7.3 · Original Alpaca | Timed pilots on all 52,002 records (44,201 train, mean 107.5 tokens, p99 319, max 1,510). In FP32, `batch_size=8` and `batch_size=4` both OOMed on the 1,024-token longest batches. `batch_size=2` succeeded at **16.12 GB** peak VRAM and projected **~40.6 min/epoch** (~81.2 min for two epochs). Full 52k training was held back pending that budget decision. | Switches to the 52k dataset and inspects generated replies. |
| 7.4 · LoRA | Freezing the 406.3M base weights and adding rank-16, alpha-16 adapters trained **7,898,384** parameters (1.94% of base), cut peak VRAM by **44.7%** (4.58 GB vs 8.28 GB) and time by **15.7%** (45.3 s vs 53.8 s), saved a **31.7 MB** adapter file (vs 1.73 GB full checkpoint), reached answer cross-entropy **1.219**, and scored **57.20/100** (`109/110`; 56.86 with 1 prose reply). | Freezes base weights with rank 16 and alpha 16; on an L4 reports 1.30 min vs 1.80 min for full tuning with judge score around 50. |

[Chapter 7 exercises and baseline](../chapters/07-finetuning-to-follow-instructions.md)

## The optional PyTorch exercises

Appendix C also answers A.1–A.4. A.1 and A.2 cover environment setup and installation checks, which the repository’s working environments address in practice. A.3 counts 752 trainable parameters in a specific small network. A.4 times matrix multiplication on CPU and GPU. Our parameter-count and hardware experiments use different workloads. We have no separate completion record for those appendix exercises.
