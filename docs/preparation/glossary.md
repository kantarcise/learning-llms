# Learning glossary 🧩

A quick reference built from our preparation questions. Examples use the character-log experiment unless marked illustrative. Explanations are reference material, not claims that every concept has been mastered.

## Reading tensor shapes

| Symbol | Meaning | Our inspection |
| --- | --- | --- |
| B | Batch size: number of examples processed together | 2 |
| T | Input positions per example | 8 |
| D | Vector width: coordinates per token vector | 32 |
| V | Vocabulary size: number of possible token IDs | 25 |

`[B,T] = [2,8]` holds integer IDs. `[B,T,D] = [2,8,32]` holds a vector for each position. `[B,T,V] = [2,8,25]` holds next-character scores. Maximum input length is **24**; it is independent of vector width **32**.

## Corpus

A **corpus** is a collection of text gathered for study or training; its plural is **corpora**. Our six log sequences form a tiny corpus. A corpus does not imply a particular tokenizer or size. Available corpus tokens differ from total tokens processed during training, since training can revisit some examples or stop before covering them all.

## Self-supervised learning

Training where targets are constructed from the data itself. In next-token prediction, the actual next token is the target; people do not manually assign a label to every position. This still uses a loss and parameter updates. It does not mean the model invents its own correct answers.

## Autoregressive generation

**Autoregressive** means predicting the next element using earlier elements of the sequence. In language generation, the model uses its own previously generated tokens as part of the context for later predictions. For example, `the` → `the cat` → `the cat sleeps` illustrates successive token additions; actual token boundaries depend on the tokenizer.

Predict a next token from the preceding context, select it, append it, and repeat. Generated tokens become inputs to later predictions. Model weights stay fixed during this inference loop; generation is not an optimizer update. See the [Chapter 1 illustration](../chapters/01-understanding-large-language-models.md#autoregressive-generation-predict-append-repeat).

## Token, vocabulary, and encoding

A **token** is a unit chosen by the tokenizer. Here it is one Python string character. A **vocabulary** lists all available tokens. An **ID** is a token's integer index in that list, not a measure of its meaning.

**Encoding** maps text to IDs: with vocabulary `["a", "b", "c"]`, `"cab"` becomes `[2,0,1]`. **Decoding** reverses that mapping. Our actual vocabulary maps `j` to 12 and `o` to 17. It is fixed during model training.

## Tensor, shape, and dtype

A **tensor** is PyTorch's numerical array object. It can have zero, one, two, three, or more dimensions; it is not necessarily 3D. **Shape** lists the length of each axis. **dtype** specifies the numerical type; `torch.long` means 64-bit integers, used here for IDs.

`torch.tensor([[12,17],[17,12]])` constructs a tensor of shape `[2,2]`. The numbers retain their values. Tensor creation does not perform embedding lookup. For the list-to-array/tensor bridge, see [Sezai's numpyandpytorch examples](https://github.com/kantarcise/numpyandpytorch) and our [preparation discussion](first-conversation.md#why-turn-character-ids-into-tensors).

## Embedding and vector width

An **embedding** is a numerical vector representing an entry. The token embedding table stores one trainable vector per vocabulary entry. Lookup retrieves the row indexed by a token ID: ID 12 selects the row for `j`.

**The vocabulary size tells us how many rows. The width tells us how many coordinates per row.** Our table has 25 rows, each containing 32 floating-point numbers: shape `[25,32]`. Width is a chosen hyperparameter; the values are learned parameters.

## Position embedding and context length

A **position embedding** is a vector selected by the position index, rather than token ID. Our table has 24 position rows, each with 32 coordinates. At position 0, the input to the block is `token_table[token_id] + position_table[0]`, added coordinate by coordinate.

**Context length** here is the maximum supported input length: 24 tokens. The eight-character inspection uses position rows 0–7. Learned absolute position tables are this model's design; other models represent position differently.

## Parameter, weight, bias, and hyperparameter

A **parameter** is a stored number that training can adjust. **Weights** often refer to matrix entries or collectively to model parameters. A **bias** is a learned offset, such as the `b` in a linear transformation. Embedding coordinates and LayerNorm scale/shift are parameters too.

A **hyperparameter** is a configuration choice, such as width 32, context length 24, or learning rate 0.003. Our embedding tables start randomly; not every parameter does (LayerNorm scales start at one and shifts at zero).

## Activation and forward pass

An **activation** is a value computed while processing an input, such as a retrieved embedding or an attention output. It is not a newly stored learned parameter.

A **forward pass** computes outputs from inputs using current parameters. It does not itself update our model's weights. `model(token_ids)` runs the model's `forward` through PyTorch's module-call machinery.

## Logit, softmax, and probability

A **logit** is a raw prediction score before softmax. It can be negative, exceed one, and need not sum to one with the other scores. Higher logits correspond to higher probabilities within the same prediction.

Illustrative three-token vocabulary `["a", "b", "c"]`:

```text
Logits:                 [0,       1,       2]
After softmax:          [0.0900,  0.2447,  0.6652]  (rounded)
Highest-scoring token:   c
```

**Softmax** converts scores into positive probabilities summing to one: `p_i = exp(score_i) / sum(exp(scores))`. A deliberately masked score of negative infinity gives probability zero. Applying softmax does not select a token; selection is another step.

Our model returns `[2,8,25]` logits: 25 candidate next-character scores for each of eight positions in each of two examples. The cross-entropy function accepts logits directly, so we do not softmax before passing scores to it.

## Target, loss, and cross-entropy

A **target** is the correct answer used to score a prediction. For input `job=orde`, shifted targets are `ob=order`. At input position 3 (`=`), the correct next character is `o`.

**Loss** measures prediction error. **Cross-entropy** for one target is `-log(probability assigned to the correct token)`. A correct-token probability of 0.5 gives loss about 0.693; 0.1 gives about 2.303 (natural logarithm). Our batch loss averages over its target positions. We can compute validation loss without updating weights.

## Gradient, backward, optimizer, and no_grad

A **gradient** describes how a small change in a parameter affects loss locally. `loss.backward()` calculates gradients through the recorded operations. `optimizer.step()` uses gradients to update parameters. They are separate operations.

`torch.no_grad()` disables tracking operations for a later backward pass in its scope; predictions and loss can still be computed. `model.eval()` instead changes the behavior of certain layers, such as dropout. It does not itself disable gradient tracking.

## Attention, causal mask, and projection

**Attention** combines value vectors using mixing amounts derived from query–key scores. Q, K, and V are temporary vectors computed with learned projection matrices. **Projection** here means a learned linear transformation, not necessarily a geometric orthogonal projection.

For `[B,T,D]` queries and keys, `Q @ K.transpose(-2,-1)` gives `[B,T,T]` scores: each query compared with each key. `-2` and `-1` refer to the last two axes. A **causal mask** blocks future keys, so position t can access positions 0 through t. It still predicts the character after t.

## LayerNorm and residual connection

**LayerNorm** normalizes a token's coordinates using their mean and variance (with a small stabilizing epsilon), then applies learned scale and shift. The two norms inside our block have separate parameters.

A **residual connection** adds an update to the incoming representation: `y = x + attention(norm1(x))`. The shortcut carries x unchanged to the addition. Here x already includes token and position embeddings.

## Feedforward network and GELU

Our **feedforward network** applies `Linear(32,64) → GELU → Linear(64,32)` to each position, sharing weights across positions. It does not mix positions. **GELU** is a nonlinear activation function. The feedforward update is added through the second residual connection: `z = y + feedforward(norm2(y))`.

## Module, transformer block, and model

An `nn.Module` is a PyTorch component that can contain parameters and child modules. It is distinct from a Python module meaning a source file such as `model.py`.

Our **transformer block** contains attention, feedforward, two LayerNorms, and two residual additions. Our **model**, `TinyTransformer`, contains the embedding tables, one such block, final LayerNorm, and vocabulary projection. A block is one component of the complete model.

## Floating-point precision: FP32, TF32, FP16, and BF16

**Floating-point precision** is how many significant digits—and how much accuracy—a computer can store when representing real (fractional, very large, or very small) numbers in a finite number of bits. It exists because real numbers are infinite while computer memory is finite: by letting the decimal (binary) point "float" like scientific notation, one compact format can represent both microscopic gradients and huge numbers, at the cost of small rounding approximations (such as `0.1 + 0.2 = 0.30000000000000004` in binary float—which is why financial systems use exact decimal/integer math, whereas scientific computing uses 64-bit `FP64` and machine learning trades extra digits for speed and VRAM).

Following the IEEE 754 pattern, a floating-point number is stored in binary scientific notation: `± (mantissa) × 2^(exponent)`. For example, `6.5` in binary is `110.1₂ = 1.101₂ × 2²` (sign `+`, mantissa `1.101₂ = 1.625`, exponent `2`).

- **Sign bit:** positive or negative.
- **Exponent bits** set the **range** (how giant or tiny a number can be before overflowing to `inf`/`NaN` or underflowing to `0`).
- **Mantissa (significand) bits** set the **decimal sharpness** (how many significant digits are kept).

| Format | Stored size | Sign \| Exp \| Mantissa | Max value | Role in LLM training and serving |
| --- | :---: | :---: | :---: | --- |
| **FP32** | 4 bytes | `1 \| 8 \| 23` | `3.4 × 10³⁸` | Standard baseline: huge range and fine decimals, heavy on VRAM and bandwidth. |
| **FP16** | 2 bytes | `1 \| 5 \| 10` | `65,504` | Half the memory, narrow range; training gradients can overflow to `NaN` or underflow to `0` without loss scaling. |
| **BF16** (*Brain Floating Point 16*) | 2 bytes | `1 \| 8 \| 7` | `3.4 × 10³⁸` | Invented by Google Brain for TPUs: literally FP32 with the bottom 16 mantissa bits dropped. Keeps FP32's full range while halving memory and running on Tensor Cores. |
| **TF32** (*TensorFloat-32*) | 4 bytes (in RAM) | `1 \| 8 \| 10` *(compute)* | `3.4 × 10³⁸` | NVIDIA Ampere (RTX 3090+) Tensor Core mode: stores FP32 in memory and trims the mantissa to 10 bits on the fly during matrix multiplication. |

**Three concrete bridges for why BF16 works so well in LLMs:**

1. **Same simple numbers, tiny rounding on messy ones (`6.5` vs. `π`):** `6.5 = 1.101₂ × 2²` only needs 3 mantissa bits, so FP32, FP16, and BF16 all store `6.5` identically. For `π = 3.14159265...`, BF16 literally keeps the top 16 bits of FP32 and drops the bottom 16 mantissa bits:
   - **FP32** (`1 | 8 | 23`): `0 | 10000000 | 10010010000111111011011` → `3.1415927`
   - **BF16** (`1 | 8 | 7`): `0 | 10000000 | 1001001` → `(1 + 1/2 + 1/16 + 1/128) × 2 = 3.140625` (**0.031%** error)
2. **Range breaks training; tiny rounding does not (`70,000` and `10⁻⁸`):** Because an LLM multiplies and sums thousands of noisy weights across millions of operations, ±0.03% mantissa rounding averages out while halving memory and unlocking Tensor Core throughput. Running out of *exponent* range is catastrophic: in **FP16** (max `65,504`), an activation of `70,000` overflows to `inf` (and `inf / inf` becomes `NaN`, poisoning the next matmul) while a tiny `10⁻⁸` gradient underflows to `0.0`. In **BF16**, `70,000` stays finite (`70,144`) and `10⁻⁸` stays non-zero.
3. **The ruler analogy:** think of the **exponent** as the *span* of a ruler (can it measure both an ant and a skyscraper?) and the **mantissa** as the *tick marks* (millimeters vs. nanometers). Neural networks process both tiny gradients and large activations, so they need the wide span (8-bit exponent), not nanometer tick marks (23-bit mantissa). That is why BF16 matches FP32 quality on LLMs at half the memory (`~14.25 GiB` instead of `~28.5 GiB` for Qwen2.5-7B-Instruct on a 24 GiB RTX 3090).

## FlashAttention and PyTorch SDPA

Naive self-attention computes `Q @ Kᵀ`, writes the full `[B, n_heads, T, T]` score matrix to GPU VRAM (HBM), reads it back for `softmax`, writes it again, reads it for `@ V`, and keeps copies in VRAM until backpropagation. Because `[B, n_heads, T, T]` grows quadratically with sequence length (`O(T²)`), long prompts quickly exhaust VRAM.

**FlashAttention** computes the **exact same mathematical attention output** without ever materializing that `T × T` matrix in VRAM. Just as a CPU has fast L1/L2 caches in front of main RAM, a GPU has tiny, much faster on-chip **SRAM** in front of VRAM. FlashAttention loads small **tiles** (slices of tokens, e.g. 64–128 at a time) of `Q`, `K`, and `V` into SRAM, computes the exact softmax incrementally tile by tile, and writes only the final `[B, n_heads, T, d_head]` output back to VRAM (recomputing tiles in SRAM during backprop instead of storing the `T × T` matrix). Memory drops from `O(T²)` to `O(T)` and execution runs ~2–4× faster by eliminating slow VRAM round-trips.

**PyTorch SDPA** (`torch.nn.functional.scaled_dot_product_attention`) is the built-in abstraction that automatically dispatches to fused kernels such as FlashAttention-2 on Ampere+ GPUs (including the RTX 3090) when its preconditions are met—notably **FP16 or BF16** precision (FP32 tensors fall back to non-FlashAttention backends) and a standard causal mask (`is_causal=True`).

## LoRA (Low-Rank Adaptation)

**LoRA** adapts a pretrained model without updating its full weight matrices. Every pretrained linear weight `W` is frozen (`requires_grad=False`), and a small trainable side branch of two skinny matrices `A` (`d_in × r`) and `B` (`r × d_out`) is added in parallel: `y = Wx + α((x @ A) @ B)`. Initializing `B` to all zeros ensures `A @ B = 0` at step 0, so training starts from the exact pretrained behavior without random noise.

Even though all frozen base weights stay in GPU memory, LoRA cuts training VRAM sharply because FP32 AdamW stores **16 bytes per trainable parameter** across four buckets: weights (`4 B`), gradients (`4 B`), first moment `m` (`4 B`), and second moment `v` (`4 B`). Freezing the base weights eliminates the last three buckets (`12 B/param`) for ~98% of the model.

## Interval notation

`[0,V)` means **0 is included, V is excluded**. For integer IDs and V=25, valid values are 0 through 24. `positions <= t` means positions up to and including t.

## References

- [PyTorch tensors](https://docs.pytorch.org/docs/stable/tensors.html)
- [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch scaled_dot_product_attention (SDPA)](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [NVIDIA TensorRT accuracy considerations (FP32, TF32, FP16, BF16)](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [FlashAttention paper (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
