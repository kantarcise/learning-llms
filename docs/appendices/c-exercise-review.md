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

## Chapter 7 · Reference results for the Linux runs

The baseline two-epoch instruction fine-tune ran on our RTX 3090. These four exercise comparisons are pending in the handoff. The figures below describe the author's runs.

| Exercise | Appendix C comparison to make |
| --- | --- |
| 7.1 · Prompt style | Train and extract replies using matching Phi-3 markers. The author reports similar quality and about 17% faster training with shorter inputs. Trying a template only at inference answers a different question. |
| 7.2 · Instruction masking | Keep the prompt visible and exclude its targets from loss. The author’s judge score fell by about four points. Check the direction on our held-out replies. |
| 7.3 · Original Alpaca | The appendix switches to the larger instruction dataset and inspects generated replies. Our plan adds a throughput and memory pilot before the full comparison. |
| 7.4 · LoRA | The appendix freezes the base weights and uses rank 16 and alpha 16. On an L4 it reports 1.30 minutes versus 1.80 minutes for full tuning, with a judge score around 50. Measure time and quality anew on the 3090. |

[Chapter 7 exercises and baseline](../chapters/07-finetuning-to-follow-instructions.md)

## The optional PyTorch exercises

Appendix C also answers A.1–A.4. A.1 and A.2 cover environment setup and installation checks, which the repository’s working environments address in practice. A.3 counts 752 trainable parameters in a specific small network. A.4 times matrix multiplication on CPU and GPU. Our parameter-count and hardware experiments use different workloads. We have no separate completion record for those appendix exercises.
