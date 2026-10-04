# From tokens to next-token predictions

Conversation notes and handoff, October 3–4, 2026. The book has not arrived yet. These are preparation notes from a guided discussion, not completed book chapters or independently completed exercises. Explanations are condensed rather than a verbatim transcript; quoted observations are Sezai's words.

## Start here on the other computer

We stopped after Sezai explained that the model is trained to predict the ending token too: “OH that makes sense, we train and predict end token too.”

Next: work through **greedy decoding versus sampling** with a tiny vocabulary and probabilities. Sampling has been explained briefly but understanding has not been checked. Then return to **why transformers need multiple blocks**, using Sezai's CS231N background. Do not jump straight into Q/K/V equations, pooling, or tensor notation without establishing the motivation.

Teaching preferences: concrete numerical examples, small steps, and a check before advancing. When discussing embeddings, write **embeddings (vectors)**. Clearly distinguish stored model parameters from temporary activations. Sezai already understands training versus inference from his graduation project; focus on the language-model-specific connections.

## What we are building

A readable, chapter-by-chapter companion to Sebastian Raschka's *Build a Large Language Model (From Scratch)*. Writing is part of learning; runnable examples and inference-engineering connections grow alongside the book. The study budget is six hours per week over six months. Chapter pages remain scaffolds.

The site is live at [learningllms.kantarcise.com](https://learningllms.kantarcise.com/). Cloudflare Pages builds and deploys after updates to main. GitHub Actions validates documentation on pull requests and main pushes; it does not deploy the site. CI failures only block merges if branch protection requires the check. Local development and CI use uv; Cloudflare's user-confirmed working command is `mkdocs build`. See [deployment instructions](https://github.com/kantarcise/learning-llms/blob/main/DEPLOYMENT.md) in the repository for setup history and dependency exports.

## The generation loop

Sezai's starting explanation: the prompt is decomposed into tokens, and the model guesses matching tokens to satisfy the request.

Refinement: the model predicts **one next token at a time**, using the prompt and previously generated tokens. Select a token, append its ID, and repeat. Parameters stay fixed during ordinary inference; the input sequence grows.

Conceptually we process the expanded context on each step. Serving systems generally reuse intermediate attention data through a **KV cache** instead of recomputing all earlier work. The cache was mentioned, not explained in detail.

Pre-training is itself training, rather than a separate stage before “training.” Pre-training and post-training are broad stages of learning; inference uses the resulting parameters. Training data is a selected collection, not literally everything on the internet.

## Tokenization: text to integer IDs

A particular tokenizer configuration deterministically maps the same text to the same IDs. Different encodings can produce different splits and IDs. IDs are vocabulary labels; neighboring IDs do not imply similar meanings.

Tokens may contain full words, fragments, punctuation, spaces, or byte pieces. Word count and token count tend to grow together, but there is no fixed one-to-one relationship.

These examples were actually run with tiktoken's `cl100k_base` encoding in a temporary uv environment. No tiktoken dependency was added to the project.

| Text | Token pieces | IDs |
| --- | --- | --- |
| `hello` | `"hello"` | `[15339]` |
| ` hello` | `" hello"` | `[24748]` |
| `Hello, world!` | `"Hello"`, `","`, `" world"`, `"!"` | `[9906, 11, 1917, 0]` |
| `cat` | `"cat"` | `[4719]` |
| `cats` | `"cats"` | `[38552]` |
| `unbelievable` | `"un"`, `"belie"`, `"vable"` | `[359, 32898, 24694]` |
| `Merhaba dünya!` | `"Mer"`, `"hab"`, `"a"`, `" dü"`, `"nya"`, `"!"` | `[27814, 10796, 64, 52119, 23741, 0]` |

Encoding then decoding the entire sequence restores the original text, including punctuation and spaces. Individual tokens can contain partial UTF-8 characters, so decoding each token separately as text is not always safe. The examples above were inspected using token bytes.

Source: [tiktoken documentation and code](https://github.com/openai/tiktoken).

### Does punctuation mean prompts need perfect grammar?

Spaces and punctuation affect tokenization, but different tokenization does not inherently mean worse output. Clear intent, relevant context, and explicit output constraints matter more than formal grammar. Punctuation helps disambiguate meaning. Turkish and other languages can be used; performance varies with model, language, and task. Sezai's conversational wording was understandable.

## Token IDs select embeddings (vectors)

The tokenizer produces IDs. The model owns a learned lookup table: one row per vocabulary entry, with a fixed number of values per row.

Illustrative values:

| ID | Embeddings (vectors) retrieved |
| --- | --- |
| 0 | `[0.2, -0.5, 0.8]` |
| 1 | `[0.3, -0.4, 0.7]` |
| 2 | `[-0.6, 0.1, 0.2]` |

ID `1` selects row `1`. We do not multiply the integer ID by that row. Mathematically, a one-hot row multiplied by the table selects the same row; an actual lookup avoids building that large one-hot input.

“Vector” describes the mathematical form, a list of numbers. Embeddings (vectors) describe learned numerical representations of something. Not every vector is an embedding.

These table values are **model weights**: learned numerical parameters, initialized and adjusted during training, saved in the model checkpoint, and loaded into memory for inference. Attention and feedforward matrices are other learned parameters. On GPU inference systems, relevant weights normally reside in GPU memory, though placement and offloading vary.

Two models can share a tokenizer while learning different table values. A model generally requires its matching tokenizer because its input rows and output classes were trained against those token-ID meanings. Arbitrarily swapping the tokenizer breaks that agreement.

Quantization approximates parameter values with lower precision. It can alter the table values if that table is quantized; some setups quantize other matrices and retain the table at higher precision. It does not inherently change the token IDs or tokenizer.

## Stored parameters versus contextual activations

This was the hardest distinction in the discussion. Assuming the same token ID, `bank` retrieves the same initial embeddings (vectors) in every sentence. It does not have separate stored rows for financial and geographical meanings.

But the whole sequence is different:

- “I deposited money at the bank.”
- “I walked along the river bank.”

Other positions have different initial embeddings (vectors). Attention incorporates information from accessible positions into the representation at `bank`, so later contextual vectors can differ. The stored row remains unchanged.

A causal model can use preceding positions, not future ones. Earlier examples with “The bank approved my loan” were corrected: the representation at `bank` cannot use the later word `loan`; later positions can use both.

### Numerical mixing example

All values and mixing amounts here are made up. This is an illustration, **not a full attention implementation**.

| Token | Initial embeddings (vectors) |
| --- | --- |
| `money` | `[10, 0]` |
| `river` | `[0, 10]` |
| `bank` | `[2, 2]` |

Imagine the calculation at `bank` takes half of the preceding vector and half of its own:

```text
money bank: 0.5 × [10, 0] + 0.5 × [2, 2] = [6, 1]
river bank: 0.5 × [0, 10] + 0.5 × [2, 2] = [1, 6]
```

The resulting vectors differ because one of the inputs differs. Neither output is a token ID or a final prediction. Both are temporary activations. The stored `bank` row is still `[2, 2]`. Coordinates do not necessarily have individually interpretable meanings such as “finance” or “river.”

Real attention calculates context-dependent mixing amounts and combines transformed value vectors, within a block that also includes residual connections, normalization, and a feedforward network. Those mechanics remain to be explained.

## The CNN bridge

Sezai has studied Stanford CS231N and understands learned convolution filters, activations, backpropagation, and final classification scores. Cat-versus-dog labels describe classification; detection also locates objects.

A fixed convolution filter can produce different activations for different inputs. Even if two images share the same central patch, different surrounding pixels change a filter's input when its receptive field includes those surroundings.

Likewise, `bank` starts with the same row, but attention's calculation also receives information from different surrounding token positions. If it processed only the identical bank row independently, fixed operations would give the same result.

```text
CNN: image → learned feature layers → linear output layer → class scores
LLM: token IDs → embeddings (vectors) → transformer blocks → vocabulary scores
```

Transformer blocks include attention and feedforward networks. Updated vectors flow into later blocks. At the end, a final normalization and output projection typically turn the last position's representation into next-token logits, one per vocabulary entry; softmax converts logits to probabilities.

Why multiple blocks help remains open. The earlier remark comparing token sequence length to CNN pooling was confusing and should be unpacked later. Transformer blocks generally preserve the number of token positions; they transform representations rather than acting like spatial pooling.

## Why token pieces form proper language

The output classes are tokenizer vocabulary entries, not necessarily full words. Training provides the actual next token as the target, and errors drive backpropagation. Repeated across text, this teaches patterns spanning word pieces, words, and sentences.

The hypothetical split `app` + `le` was illustrative, not a measured tiktoken split. A continuation depends on the entire accessible context, not just the final piece. “I ate an app…” and “Install the app…” suggest different continuations. Coherent output is learned, not guaranteed.

Sezai's connection: “Unbelievable. Everything comes to backpropagation.” The familiar predict → error → backpropagate → update loop applies here, with next-token targets instead of image-class targets.

## Generation stays in IDs; display decoding is separate

```text
Current IDs: [12, 31, 8]
Selected ID: 42
Next IDs:    [12, 31, 8, 42]
```

Generation appends the selected ID directly. It does not need to decode text and tokenize it again. The matching tokenizer decodes generated IDs for display, either during streaming or at the end. Streaming decoding may buffer incomplete character bytes.

Sezai demonstrated this: “if we didnt show anything while generating, model did not need to decode in between at all.”

### Highest score versus sampling — next topic

Greedy decoding picks the highest-scoring token. Sampling chooses according to a probability distribution, often modified by decoding settings. A brief illustrative distribution was discussed:

```text
“The meal was …”
delicious: 45%
excellent: 35%
tasty:     20%
```

Several continuations can be appropriate. Locally highest probability does not guarantee the best complete answer. Temperature, top-k/top-p, tradeoffs, and an actual sampling walkthrough have not been covered.

### Stopping is learned too

A period is ordinary punctuation and may be followed by another sentence. A special ending token marks a completed response in the training format. Predicting that token is learned using the same next-token objective as predicting words.

```text
Question: What is the capital of France?
Answer: Paris.
[END]
```

`[END]` is an illustrative label, not literal user-facing text. Exact end-of-sequence and end-of-turn conventions depend on the model and chat format. The serving system recognizes the selected ending ID and stops; limits, stop strings, or cancellation can also terminate generation. Endings can be predicted prematurely or too late.

Sezai explicitly confirmed understanding that ending tokens are trained and predicted too. The model does not need a separate human-like decision process for stopping.

## Understanding checklist and remaining questions

Checked items reflect Sezai's explanations or explicit confirmations, not merely topics the assistant introduced.

- [x] Word count is not a fixed token count; tokens include fragments and punctuation.
- [x] Tokenization with a fixed configuration is deterministic.
- [x] Initial embeddings (vectors) belong to the model and are learned parameters.
- [x] The same token row can lead to different contextual vectors because surrounding inputs differ.
- [x] Generation appends IDs without needing intermediate text decoding.
- [x] Ending tokens are learned and predicted too.
- [ ] Explain greedy decoding versus sampling and why either might be chosen.
- [ ] Trace attention's actual mixing calculation beyond the illustrative average.
- [ ] Explain why successive transformer blocks help, using a CNN comparison carefully.
- [ ] Predict shapes as tokens are appended; earlier shape questions were interrupted, not answered.
- [ ] Explain Q/K/V and then KV cache with a worked example.
- [ ] Implement and inspect a runnable tokenization/model example; only the assistant ran tiktoken so far.

No chapter reading, tensor practice, daily-capacity exercise, or independent benchmark has been completed or claimed here.

## Further references

- [Book and companion resources](https://sebastianraschka.com/llms-from-scratch/)
- [Author's implementation](https://github.com/rasbt/LLMs-from-scratch)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original transformer paper; detailed equations can wait until the conceptual walkthrough is clear.

These notes summarize our conversation and use explicitly illustrative examples; they are not excerpts from the book.
