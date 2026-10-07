# From tokens to next-token predictions

Conversation notes and handoff, October 3–7, 2026. The book has not arrived yet. These are preparation notes from a guided discussion, not completed book chapters or independently completed exercises. Explanations are condensed rather than a verbatim transcript; quoted observations are Sezai's words.

## Start here on the other computer

We stopped for rest after the October 7 continuation, at **forming shifted next-token training examples and batches**. The width-4 toy block was completed through feedforward and its second residual addition, then connected to vocabulary scores, ending tokens, cross-entropy, and the familiar neural-network training loop. Keep the worked example below for reference; do not restart with different numbers.

Next: resume the unanswered check in the latest training-window example: **the input contains `today`, but can the position at `bank` attend to it?** Establish causal masking during training, then show multiple windows forming a batch and how score/target shapes produce the loss. Cross-entropy and its logit-gradient direction were discussed, but Sezai said he did not fully understand the detail; do not mark a derivation or independent calculation completed. The overall training-loop connection to CNNs is understood. No experiment has been implemented or run.

Teaching preferences: concrete numerical examples, small steps, and a check before advancing. When discussing embeddings, write **embeddings (vectors)**. Clearly distinguish stored model parameters from temporary activations. Sezai already understands training versus inference from his graduation project; focus on the language-model-specific connections.

Sezai explicitly requested emojis in explanations and notes: learning is tough enough, and he enjoys using them. Use a few helpful markers such as 🧩, 🌱, and 🎉 while keeping the math readable.

Scope matters: explicitly identify this as standard scaled dot-product attention with one head. It is a real mechanism, not the exact implementation of every model. Introduce each new quantity before using it; distinguish a numerical score from the specifically named **value vector**.

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

Real attention calculates context-dependent mixing amounts and combines transformed value vectors, within a block that also includes residual connections, normalization, and a feedforward network. The first Q/K/V example below begins explaining the mixing; the other block components remain to be explained.

## The CNN bridge

Sezai has studied Stanford CS231N and understands learned convolution filters, activations, backpropagation, and final classification scores. Cat-versus-dog labels describe classification; detection also locates objects.

A fixed convolution filter can produce different activations for different inputs. Even if two images share the same central patch, different surrounding pixels change a filter's input when its receptive field includes those surroundings.

Likewise, `bank` starts with the same row, but attention's calculation also receives information from different surrounding token positions. If it processed only the identical bank row independently, fixed operations would give the same result.

```text
CNN: image → learned feature layers → linear output layer → class scores
LLM: token IDs → embeddings (vectors) → transformer blocks → vocabulary scores
```

Transformer blocks include attention and feedforward networks. Updated vectors flow into later blocks. At the end, a final normalization and output projection typically turn the last position's representation into next-token logits, one per vocabulary entry; softmax converts logits to probabilities.

In the follow-up, Sezai correctly explained that the second block receives the representations produced by the first. Successive blocks provide further learned mixing and transformation of already contextualized representations. They do not have fixed human-assigned jobs such as “grammar first, meaning second.”

The model follows its configured architecture, rather than attending until it feels ready. Each block has its own learned parameters. Even one causal attention layer can access all preceding positions; depth adds successive processing, not simply access to more distant tokens. Transformer blocks generally preserve the number of token positions and the representation width at block boundaries rather than acting like spatial pooling.

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

### Highest score versus sampling

Greedy decoding picks the highest-scoring token. Sampling chooses according to a probability distribution, often modified by decoding settings. A brief illustrative distribution was discussed:

```text
“The meal was …”
delicious: 45%
excellent: 35%
tasty:     20%
```

Several continuations can be appropriate. Locally highest probability does not guarantee the best complete answer. Sezai correctly connected the selected token to the later continuation: it becomes part of the context and can change all subsequent next-token distributions.

For a concrete sampling walkthrough, draw a number uniformly from 0 to 1:

| Token | Probability | Selection interval |
| --- | --- | --- |
| ` delicious` | 45% | 0 ≤ r < 0.45 |
| ` excellent` | 35% | 0.45 ≤ r < 0.80 |
| ` tasty` | 20% | 0.80 ≤ r < 1 |

Sezai correctly selected ` excellent` for `r = 0.78`. Selecting it is not an error just because another token has greater probability. Greedy decoding would always select ` delicious` for this distribution.

Temperature adjusts the distribution before sampling. Lower positive temperature concentrates probability toward the highest-scoring tokens; higher temperature spreads it more evenly. Temperature 1 leaves the distribution unchanged. This changes selection behavior, not the learned weights, and does not establish whether the model was trained well.

Temperature zero is normally handled specially as greedy decoding; the usual scaling calculation divides logits by temperature and cannot directly use zero. With identical inputs, fixed weights, identical settings, and deterministic computation, greedy decoding repeats the same continuation step by step. Real serving implementations can have numerical differences, so temperature zero alone is not an absolute reproducibility guarantee. Different inputs can produce different outputs.

Top-k and top-p have not been covered. Temperature was explained and discussed, but no numerical temperature exercise has been completed.

### Stopping is learned too

A period is ordinary punctuation and may be followed by another sentence. A special ending token marks a completed response in the training format. Predicting that token is learned using the same next-token objective as predicting words.

```text
Question: What is the capital of France?
Answer: Paris.
[END]
```

`[END]` is an illustrative label, not literal user-facing text. Exact end-of-sequence and end-of-turn conventions depend on the model and chat format. The serving system recognizes the selected ending ID and stops; limits, stop strings, or cancellation can also terminate generation. Endings can be predicted prematurely or too late.

Sezai explicitly confirmed understanding that ending tokens are trained and predicted too. The model does not need a separate human-like decision process for stopping.

## Width, sequence length, and model size

The October 4–5 follow-up separated three quantities: number of token positions, numbers per token vector, and number of transformer blocks.

One token represented by `[2, 1, 3]` has width 3 and can be written as a 1 × 3 row. Width 4,096 means 4,096 numbers per token vector. Ignoring the batch dimension, three token positions of width 3 form a 3 × 3 representation matrix:

```text
money  [1, 4, 2]
at     [0, 1, 1]
bank   [2, 1, 3]
```

Appending `today` adds another row, producing shape **4 × 3**, not 3 × 4. These illustrative initial embeddings (vectors) make the shape visible; subsequent contextual values can change while the width stays fixed.

Sezai initially answered that appending a token would widen the vectors. After clarification he explained: “Hence, the model itself doesn't change. The width stays, but if the input changes, there's obviously gonna be more multiplication because we have more rows to multiply.”

The same stored weight matrix is applied at each position. More tokens mean more computation and temporary memory, not more learned parameters. Attention also has more positions to consider. KV caching was mentioned as a way to reuse earlier attention work during generation, but its mechanics have not been covered.

### A square learned matrix

“Square” describes equal row and column counts, not squaring each weight. This made-up 3 × 3 matrix contains nine learned parameters:

```text
W = [1  0  2]
    [0  1  0]
    [1  0  1]

[2, 1, 3] × W = [5, 1, 7]
```

The first output coordinate is `2×1 + 1×0 + 3×1 = 5`. The coefficients in W are stored weights; `[5, 1, 7]` is a temporary activation. A 1 × 4,096 row can similarly multiply a 4,096 × 4,096 matrix to produce another 1 × 4,096 row. Other projections can be rectangular; dimensions must be compatible.

A width-3 square matrix has 9 parameters; width 6 has 36. Doubling both dimensions quadruples its parameter count. Transformer blocks contain several learned matrices, so widening representations can grow a model substantially without adding blocks. “Bigger weights” here means more learned numbers, not larger numerical values.

### Published architecture examples

Sezai correctly concluded that two models can have the same number of transformer blocks but very different parameter counts. Models can grow in depth, width, feedforward dimensions, or number of experts.

| Llama 3 model | Parameters | Transformer blocks | Representation width |
| --- | --- | --- | --- |
| 8B | 8 billion | 32 | 4,096 |
| 70B | 70 billion | 80 | 8,192 |
| 405B | 405 billion | 126 | 16,384 |

These models grow in both depth and width. More capacity can support more complex learned patterns when trained effectively; additional blocks do not guarantee a better prediction. Source: [Meta's Llama paper, Table 3](https://arxiv.org/html/2407.21783v3).

DeepSeek-V3 provides a different comparison: 61 transformer layers, 671 billion total parameters, and approximately 37 billion activated per token. Its mixture-of-experts structure includes alternative feedforward networks selected by a learned router, contributing many parameters without requiring more sequential blocks than Llama 405B. Sources: [published configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json), [technical report](https://arxiv.org/abs/2412.19437).

Expert count is an architecture choice evaluated against quality and cost; training learns routing. Experts need not correspond to clean human labels such as “math” or “coding.” Routing inspection and disabling experts can help study their behavior. This was a brief introduction, not a worked MoE lesson. Sezai asked how expert count is chosen and how specialization is identified; these questions remain for later. Parameter-count claims alone do not reveal an undisclosed proprietary model's depth, width, or expert structure.

## First Q/K/V example

This is a toy example of **standard scaled dot-product attention**, with one head and two positions. Numbers are invented; `money` and `bank` each stand for one token position. Position encoding and the rest of the transformer block are omitted. It demonstrates a real calculation rather than claiming all models implement identical attention.

Each incoming representation is projected with three learned matrices:

```text
X × WQ → queries (Q)
X × WK → keys    (K)
X × WV → values  (V)
```

WQ, WK, and WV are stored model parameters. Q, K, and V depend on the incoming representations and are temporary activations. Sezai explicitly identified these matrices as learned weights. These are three logical projections, not the total computation of a block; implementations can fuse them into one larger operation.

Queries and keys determine mixing amounts. Values supply the vectors to mix. These are learned roles, not manually assigned coordinate meanings.

### Calculate the scores at bank

Assume the projections have already produced:

| Position | Query | Key | Value |
| --- | --- | --- | --- |
| `money` | Not needed for this output | `[1, 0]` | `[10, 0]` |
| `bank` | `[2, 0]` | `[0, 1]` | `[2, 2]` |

We use the query at `bank` to score both keys. A causal position can attend to itself and preceding positions, so both are accessible here.

```text
Q_bank · K_money = 2×1 + 0×0 = 2
Q_bank · K_bank  = 2×0 + 0×1 = 0
```

Sezai correctly identified `money` as receiving more attention. A higher **query–key score** is distinct from a larger **value vector**.

Standard attention scales by the square root of the key width, here √2, and applies softmax:

```text
Raw scores:    [2, 0]
Scaled scores: [1.414…, 0]
Mixing weights after softmax: approximately [0.80, 0.20]
```

The softmax outputs sum to 1. Future positions would be masked before softmax. Calculating `money`'s output would exclude the later `bank` position. A separate softmax result is calculated for each query position.

### Why introduce V?

Q and K have answered **how much weight each position receives**. They have not yet supplied the new contextual vector. “80% of what, and 20% of what?” is the bridge to V.

V comes from the same incoming representations through its own learned projection, WV. With the assumed values above, the next calculation is:

```text
0.80 × V_money + 0.20 × V_bank
= 0.80 × [10, 0] + 0.20 × [2, 2]
≈ [8.4, 0.4]
```

This uses rounded mixing weights. It mixes both positions rather than selecting only the highest-scoring one. The output is an attention activation at `bank`, not a next-token prediction, an updated model weight, or the complete block output.

The assistant initially showed this sum before the role of V was sufficiently established. Sezai stopped the explanation, and the discussion returned to why V is needed. In the evening, the calculation was repeated from Q/K scores through scaling, softmax, and value mixing. Sezai then confirmed he could follow the calculations, while raising a different question: how Q/K/V relate to the original embeddings (vectors).

### What varies across attention implementations?

Sezai asked whether the example applies to every model. The convolution analogy was used to clarify scope: learning a 3 × 3 convolution establishes an operation without covering every kernel, stride, or grouped variant.

Multi-head attention combines several heads; grouped-query attention shares keys and values among query heads; sliding-window attention restricts accessible positions; cross-attention takes queries from one sequence and keys/values from another. Some other designs change the calculation more substantially. These variants were named, not worked through.

The example also omits attention's output projection, residual connections, normalization, and the feedforward network. Standard GPT-style attention remains the foundation for the book companion. The next sequence is: finish Q/K/V mixing, then append a token and motivate **KV caching** by identifying reusable calculations.

## Evening follow-up: connect embeddings (vectors) to attention

Sezai recognized softmax from CNNs. It is the same mathematical operation, but the interpretation differs: a classifier can use it for class probabilities; attention uses it for mixing weights over accessible positions. For the earlier two-position example, the softmax calculation was made explicit:

```text
exp(1.414…) ≈ 4.113; exp(0) = 1
money weight ≈ 4.113 / 5.113 ≈ 0.804
bank weight  ≈ 1 / 5.113 ≈ 0.196
```

The key remaining question was whether initial embeddings (vectors) are themselves queries, keys, or values. They are **incoming representations from which Q, K, and V are calculated**, not three names for the same stored embedding row. Later blocks project the previous block's updated representations using their own learned matrices. Normalization and position handling were acknowledged but not worked through.

### Training changes parameters; inference calculates activations

The phrase “embeddings change” had blurred two meanings. The clarification was explicit:

| Stored parameters learned during training | Temporary results calculated for an input |
| --- | --- |
| Embedding table | Input rows retrieved from the table |
| WQ, WK, WV | Q, K, V vectors |
| Other learned matrices | Attention outputs and contextual representations |

For the same token ID, the stored `bank` embedding row stays the same during ordinary inference. Different preceding inputs can produce different contextual representations without overwriting that row.

Training performs a forward calculation, evaluates loss against target tokens, backpropagates, and uses an optimizer to update learned parameters. Both the embedding table and WQ/WK/WV can be updated. Ordinary inference performs the forward calculation and token selection without those learning steps. Sezai said this distinction helped; an independent explanation has not yet been checked.

Appending a token adds a position rather than modifying model weights. In a causal transformer, earlier positions cannot attend to the appended future position, so their representations do not acquire that new information. The newly appended position can attend to its preceding context. This corrects the earlier vague suggestion that all existing vectors change whenever a token is appended.

### Combined example: money at bank — resume here

Assume three token IDs for `money at bank`. This is an illustrative tokenizer split, not a verified encoding result. For Llama 8B's published width, lookup retrieves three rows containing 4,096 learned numbers each: **3 × 4,096**, ignoring the batch dimension.

Shrink the width to **2** to make every number visible. Our made-up embedding table supplies:

```text
money → [10, 0]
at    → [ 0, 1]
bank  → [ 2, 1]

Input representation matrix X: 3 × 2
```

These three input rows are retrieved from stored learned parameters. Use these invented projection matrices:

```text
WQ = [1  0]    WK = [0.1  0]    WV = [1  1]
     [0  0]         [0    1]         [0  0]
```

The same matrices process every row:

| Position | Incoming x | Q = x × WQ | K = x × WK | V = x × WV |
| --- | --- | --- | --- | --- |
| `money` | `[10, 0]` | `[10, 0]` | `[1, 0]` | `[10, 10]` |
| `at` | `[0, 1]` | `[0, 0]` | `[0, 1]` | `[0, 0]` |
| `bank` | `[2, 1]` | `[2, 0]` | `[0.2, 1]` | `[2, 2]` |

This is a **new, consistent example** starting from incoming vectors and shared projection matrices. Its outputs differ from the earlier example, which supplied Q/K/V directly. Do not silently interchange the two examples.

At `bank`, use its query `[2, 0]` against the three accessible keys:

```text
money: [2, 0] · [1,   0] = 2
at:    [2, 0] · [0,   1] = 0
bank:  [2, 0] · [0.2, 1] = 0.4
```

Divide by √2 and apply softmax. In position order `[money, at, bank]`, weights are approximately `[0.639, 0.155, 0.206]`. Using the rounded amounts from the conversation:

```text
0.64 × [10, 10] + 0.15 × [0, 0] + 0.21 × [2, 2]
≈ [6.82, 6.82]
```

That is the approximate attention output at `bank`, not a replacement for its stored embedding row `[2, 1]`. It is not the full transformer-block output. The toy matrices are deliberately simple; their coordinates have no assigned semantic labels.

### Connect the toy width to Llama's blocks

Our example has width 2 and one attention head. Llama 8B has representation width 4,096 and 32 transformer blocks. Its actual attention uses multiple heads; projected Q/K/V are arranged into heads and need not have identical total widths. Attention head outputs are combined and projected back to the model width.

At block boundaries, the illustrative three-position sequence follows:

```text
3 × 4,096 incoming representations
    → block 1 → 3 × 4,096 updated representations
    → block 2 → 3 × 4,096 updated representations
    → …
    → block 32 → 3 × 4,096 updated representations
```

Each block has its own learned parameters. Final normalization and the vocabulary projection turn the last position's representation into next-token scores. The embedding table and learned matrices stay fixed during inference; the input rows, projections, and contextual outputs are temporary results. The published architecture source is [Meta's Llama paper, Table 3](https://arxiv.org/html/2407.21783v3).

This combined example was first presented before Sezai stopped for sleep on October 5. The October 6 morning follow-up below revisited its projections. KV caching, multi-head mechanics, and the remaining block components still await worked examples.

## October 6 morning: value projections and the query

We stayed with the same width-2 example rather than introducing new numbers. Sezai confirmed that each toy incoming token vector has two numbers and described V_bank as the result of multiplying the incoming vector by a weight matrix learned through training.

The value calculations were made explicit using the existing WV:

```text
WV = [1  1]
     [0  0]

V_bank  = [2, 1]  × WV = [2, 2]
V_money = [10, 0] × WV = [10, 10]
V_at    = [0, 1]  × WV = [0, 0]
```

When asked to calculate V_at, Sezai initially answered “one zero.” After showing that each output coordinate uses a column of WV, he corrected the result to `[0, 0]` and identified the mistake as matrix multiplication. Preserve the corrected result without claiming independent mastery of all projections.

We then calculated the query at bank:

```text
WQ = [1  0]
     [0  0]

Q_bank = [2, 1] × WQ
       = [2×1 + 1×0, 2×0 + 1×0]
       = [2, 0]
```

Sezai said: “I think I understand that we are using the initial embedding to multiply with every matrix we have so far.” He also identified WV/WQ/WK as learned parameters within attention. The clarification was that the transformer block contains several weight matrices and other operations; it is not itself a single weight. Q/K/V are three separate projections of the same incoming representation, not sequential transformations of one another.

The first block receives embedding-based representations; later blocks receive updated representations from the preceding block and calculate their own Q/K/V. Normalization and position handling remain omitted from the toy arithmetic. Stored embeddings and matrices remain fixed during ordinary inference; calculated vectors remain separate activations.

**Stopping point:** the V vectors are available and Q_bank has been shown. Next, calculate all three keys with the existing WK, then continue the scores and mixture. The full three-position attention calculation has not yet been independently traced by Sezai.

## October 6 continuation: attention connected, then normalization 🌱

We calculated the keys with the existing WK. Sezai initially gave K_bank as `[0.2, 0]`, then corrected it to `[0.2, 1]` after checking the second column. Q_bank = `[2, 0]` was revisited at his request. He connected per-position Q/K/V calculations to the incoming embeddings (vectors) and learned matrices; the clarification was that the matrices stay fixed during inference and are shared across positions within a layer.

The combined example then continued with scores `[2, 0, 0.4]`, scaled scores approximately `[1.414, 0, 0.283]`, and softmax weights approximately `[0.639, 0.155, 0.206]`. Using these rounded weights gives attention output `[6.802, 6.802]`. The earlier `[6.82, 6.82]` used more coarsely rounded weights; this is a rounding difference, not a different mechanism.

Sezai explained the flow in his own words and explicitly said the final sum adds value vectors multiplied by mixing amounts. Two refinements were made: attention is calculated at a token position, and scaling precedes softmax. This is demonstrated understanding of the attention calculation, not completion of a full transformer implementation. Sezai described it as a point to celebrate 🎉.

### Output projection and the first residual addition 🧩

Attention output is multiplied by another learned matrix, WO. With multiple heads, this projection transforms the combined head outputs back to the model's representation width. It is distinct from the final vocabulary projection after the entire stack.

For the toy example we assumed identity WO:

```text
WO = [1  0]
     [0  1]

[6.802, 6.802] × WO = [6.802, 6.802]
```

The residual addition then produces:

```text
Incoming bank vector + projected attention output
= [2, 1] + [6.802, 6.802]
= [8.802, 7.802]
```

Sezai remembered residual connections from ResNet and correctly identified this result under the identity-projection assumption. A learned WO generally changes the output. The residual path carries the incoming representation forward and helps gradients flow during training.

A common GPT-style pre-normalization block was drawn, marking attention as covered, output projection/residuals as introduced, and normalization/feedforward as remaining:

```text
x → normalization → attention → WO → add x → y
y → normalization → feedforward network → add y → block output
```

This is a specific common architecture, not a claim that every transformer orders operations identically. Multi-head mechanics and the feedforward network have not been worked through.

### First LayerNorm example — resume here 🌱

LayerNorm operates across coordinates of each token's vector separately; it does not mix token positions. For `bank = [2, 1]`:

```text
Mean = (2 + 1) / 2 = 1.5
Centered vector = [0.5, -0.5]
Variance = (0.5² + (-0.5)²) / 2 = 0.25
Standard deviation = √0.25 = 0.5
Standardized vector = [1, -1]
```

Actual LayerNorm divides by `sqrt(variance + epsilon)` to avoid division by zero, then applies learned per-coordinate scale and shift. With scale `[1, 1]` and shift `[0, 0]`, the result is approximately `[1, -1]`. Mean and variance are calculated from current inputs during inference; scale and shift are learned model parameters.

Normalization controls numerical scale and helps training stability. In this pre-normalization architecture it precedes Q/K/V projection. Our original toy attention deliberately omitted it: inserting LayerNorm would change Q/K/V, so the previous attention and residual numbers cannot simply be reused unchanged.

The unanswered check was: **for “money at bank,” does LayerNorm normalize each token vector separately, or use one mean across all three tokens?** Continue here. Do not mark LayerNorm understood merely because it was explained.

### How preparation relates to the book 📖

Sezai asked whether we will revisit this when the book arrives. Yes: preparation supplies a mental map; book reading and runnable PyTorch implementations remain the main path. Chapter 2 covers text data and embeddings, Chapter 3 attention, Chapter 4 the GPT architecture, and Chapter 5 training and decoding topics. See the [author's chapter outline and code](https://github.com/rasbt/LLMs-from-scratch).

We will connect hand calculations to runnable tensors and inspect shapes, revisiting unclear ideas rather than treating preparation as completed chapter work. Llama size comparisons and experts are supporting tangents. Understanding the full transformer is not a prerequisite to starting the book.

## October 7: rebuild the block with width 4 🧩

Sezai asked how LayerNorm's learned scale and shift work after standardization. An illustrative standardized `[1, -1]`, scale `[2, 0.5]`, and shift `[0.1, 0.3]` produce `[2.1, -0.2]` coordinate-wise. Scale and shift are learned by backpropagation and optimizer updates, commonly initialized to ones and zeros; calculated mean/variance are input-dependent statistics. The same scale/shift vectors are shared across positions within one module. The final output need not retain zero mean and unit variance.

Sezai could follow the arithmetic but remained uncertain why we normalize and then rescale. Record this motivation as open. He correctly summarized that embedding rows, WQ/WK/WV, and normalization scale/shift are learned parameters; WO was included in the recap.

We introduced the feedforward pattern without doing its arithmetic: expand → nonlinear activation → shrink. The book uses GELU, while ReLU was mentioned as a familiar comparison. Residual additions were linked to ResNet. A residual vector addition requires far fewer operations than a large matrix projection; this is a workload-size comparison, not a claim that an individual GPU multiplication is dramatically slower than addition. YOLO is a detection system and ResNet an architecture that can serve as a backbone, so they are not direct alternatives.

Sezai correctly explained that the first residual path bypasses normalization and attention, carrying the block's incoming representation. Later blocks carry their own incoming representations rather than repeatedly adding original embeddings. The second residual carries the result of the first addition. Both LayerNorm modules use the same recipe but have separate learned parameters. Architecture can change normalization type, placement, and count; Gemma 2's four RMSNorm modules were cited as an example, not explained in detail. Source: [Gemma 2 report, architecture section](https://arxiv.org/html/2408.00118v3).

### Why the width-2 example was misleading

The equal-coordinate attention update in the earlier example only changed the shared offset, which LayerNorm removes. Sezai noticed this. A separate width-4 example illustrated a pattern change that survives normalization, but he reasonably called it cherry-picked and asked to start from token lookup instead. LayerNorm does remove certain differences; it does not preserve all information. The residual path retains the unnormalized representation.

The following is the complete fresh example used after that request. It replaces the earlier supplied width-4 intermediate values as the current learning example. It is simplified single-head, causal, pre-normalization attention, omitting positional information and dropout. All IDs and weights are illustrative, not measured tokenizer or trained-model outputs.

### 1. Token lookup: three positions, width 4

| Token | Illustrative ID | Stored embedding row |
| --- | --- | --- |
| `money` | 0 | `[3, 1, 3, 1]` |
| `at` | 1 | `[1, 1, 3, 3]` |
| `bank` | 2 | `[1, 3, 1, 3]` |

The input shape is **3 × 4**, ignoring batch. Each ID retrieves a learned row without updating it during inference. Sezai confirmed this starting point was clear.

### 2. First LayerNorm

Each row has mean 2 and variance 1. Assume this module's learned scale is `[1, 1, 1, 1]` and shift `[0, 0, 0, 0]`. Ignoring tiny epsilon effects:

```text
money → [ 1, -1,  1, -1]
at    → [-1, -1,  1,  1]
bank  → [-1,  1, -1,  1]
```

These normalized rows feed the projections. The original incoming rows remain on the first residual path.

### 3. Q/K/V projections

Choose WQ and WK as 4 × 4 identity matrices, and WV as a diagonal matrix with diagonal `[2, 1, 1, 1]`. This deliberately simplifies the arithmetic: real WQ/WK are independently learned and generally different. Sezai questioned the identity choice; its pedagogical purpose was stated explicitly.

| Position | Q | K | V |
| --- | --- | --- | --- |
| `money` | `[1, -1, 1, -1]` | `[1, -1, 1, -1]` | `[2, -1, 1, -1]` |
| `at` | `[-1, -1, 1, 1]` | `[-1, -1, 1, 1]` | `[-2, -1, 1, 1]` |
| `bank` | `[-1, 1, -1, 1]` | `[-1, 1, -1, 1]` | `[-2, 1, -1, 1]` |

### 4. Scores and mixing amounts at bank

Use Q_bank against all three accessible keys:

```text
money: (-1×1) + (1×-1) + (-1×1) + (1×-1) = -4
at:    (-1×-1) + (1×-1) + (-1×1) + (1×1) = 0
bank:  (-1×-1) + (1×1) + (-1×-1) + (1×1) = 4
```

Sezai independently answered **4** for Q_bank · K_bank. Negative raw scores are allowed. With key width 4, divide by √4 = 2, giving `[-2, 0, 2]`. Softmax gives approximately `[0.016, 0.117, 0.867]` in `[money, at, bank]` order. Here bank attends mostly to itself; attention need not favor another position.

### 5. Weighted values and output projection

Using those rounded weights:

```text
money: 0.016 × [2, -1, 1, -1] = [ 0.032, -0.016,  0.016, -0.016]
at:    0.117 × [-2, -1, 1, 1] = [-0.234, -0.117,  0.117,  0.117]
bank:  0.867 × [-2, 1, -1, 1] = [-1.734,  0.867, -0.867,  0.867]

Attention output ≈ [-1.936, 0.734, -0.734, 0.968]
```

Choose identity WO explicitly, so the projected output is unchanged. Values here and below use rounded mixing amounts consistently, rather than claiming exact softmax precision.

### 6. First residual addition

Add the **original pre-normalization** bank input, not its standardized vector:

```text
 [ 1,     3,      1,     3    ]
+[-1.936, 0.734, -0.734, 0.968]
= [-0.936, 3.734,  0.266, 3.968] = y
```

Sezai explicitly confirmed understanding of which vector the residual addition uses.

### 7. Second LayerNorm — latest stopping point 🌱

Normalize y with this separate module. Assume its own learned scale is `[1, 1, 1, 1]` and shift `[0, 0, 0, 0]`:

```text
y = [-0.936, 3.734, 0.266, 3.968]
Mean = 1.758
Centered = [-2.694, 1.976, -1.492, 2.210]
Variance = 4.568094
Standard deviation ≈ 2.137

LayerNorm₂(y) ≈ [-1.260, 0.925, -0.698, 1.034]
```

Tiny epsilon effects are omitted in these rounded values. Keep the two paths distinct:

| Path | Current vector |
| --- | --- |
| Feedforward input | `[-1.260, 0.925, -0.698, 1.034]` |
| Second residual shortcut | y = `[-0.936, 3.734, 0.266, 3.968]` |

**Next:** choose small, explicitly illustrative feedforward matrices, calculate expand → activation → shrink, then `block output = y + Feedforward(LayerNorm₂(y))`. No such matrices or outputs have been calculated yet. Sezai requested this exact example as the handoff and wants to continue from here.

## October 7 continuation: finish the block and connect training 🔗

### Feedforward expansion, GELU, and shrinking

We resumed with normalized bank input z ≈ `[-1.260, 0.925, -0.698, 1.034]` and residual y = `[-0.936, 3.734, 0.266, 3.968]`. A diagram clarified that y is copied into two paths, not divided into coordinate groups: one passes through LayerNorm₂ and feedforward; the other preserves y for addition. Sezai confirmed this connection.

Both LayerNorm modules used scales of ones and shifts of zeros in this example. These are equal illustrative values in **separate parameter sets**, not shared weights. Attention gathered information from preceding positions; feedforward processes each position separately with shared weights across positions.

Use zero biases and this invented 4 × 6 expansion matrix:

```text
W_expand = [1  0  0  0  1  0]
           [0  1  0  0  1  0]
           [0  0  1  0  0  1]
           [0  0  0  1  0  1]

z × W_expand ≈ [-1.260, 0.925, -0.698, 1.034, -0.335, 0.336]
```

Sezai asked what fifth/sixth coordinates mean: they are the fifth/sixth numbers in this **one row**, each produced by the corresponding matrix column, not new tokens. The fifth column combines z₁ + z₂; the sixth combines z₃ + z₄. He confirmed how the 1 × 6 result is obtained.

GELU is a fixed nonlinear activation applied coordinate-wise; it has no learned parameters here. Unlike ReLU's sharp zeroing of negative inputs, GELU smoothly adjusts values and can retain small negatives. Using rounded values:

```text
GELU output h ≈ [-0.131, 0.761, -0.169, 0.878, -0.124, 0.212]
```

Expansion → activation → shrinking all belong to feedforward. The surrounding LayerNorm and residual addition are outside it. Without a nonlinear activation, the two linear projections could be combined into one. Sezai understood the shape return but questioned the usefulness: our sparse copying matrix demonstrates mechanics, not useful trained behavior. Training learns intermediate combinations and their output combination to reduce loss; this toy network was not trained and cannot be claimed to improve predictions. A separate ReLU illustration `x → [x, -x] → [max(0,x), max(0,-x)] → |x|` showed that returning to the same width does not undo a nonlinear transformation.

Choose this invented 6 × 4 shrinking matrix:

```text
W_shrink = [1  0  0  0]
           [0  1  0  0]
           [0  0  1  0]
           [0  0  0  1]
           [1  0  1  0]
           [0  1  0  1]

h × W_shrink ≈ [-0.255, 0.973, -0.293, 1.090]

Block output at bank = y + feedforward update
≈ [-1.191, 4.707, -0.027, 5.058]
```

This uses the conversation's rounded intermediates. Sezai correctly explained that this is passed to the **next transformer block** at the bank position, rather than retrieving the original embedding again. The whole sequence's updated rows pass forward. Blocks use their own learned transformations; intermediate outputs do not overwrite stored embedding rows. Sezai explicitly distinguished learned initial embeddings and transformations from calculated representations, and asked not to repeat the training/inference distinction unnecessarily.

### Architecture shape trace, inspired by Sezai's CNN notes

Sezai supplied his [Tiny YOLOv3 layer-shape gist](https://gist.github.com/kantarcise/6a2d4993ba1df697f75efa861a69238f) and requested a similar operation-by-operation trace. The response showed two toy transformer blocks explicitly, rather than silently jumping from first to last. Both have the following shape flow for three positions, width 4, one width-4 head, and feedforward width 6, ignoring batch:

| Operation in each block | Input → output |
| --- | --- |
| First LayerNorm | 3 × 4 → 3 × 4 |
| Q/K/V projections | 3 × 4 → three 3 × 4 matrices |
| Q × K transposed | (3 × 4) × (4 × 3) → 3 × 3 |
| Scaling, causal masking, softmax | 3 × 3 → 3 × 3 |
| Mixing V | (3 × 3) × (3 × 4) → 3 × 4 |
| WO and first residual addition | 3 × 4 → 3 × 4 |
| Second LayerNorm | 3 × 4 → 3 × 4 |
| Expansion | 3 × 4 → 3 × 6 |
| GELU | 3 × 6 → 3 × 6 |
| Shrinking and second residual addition | 3 × 6 → 3 × 4 |

Block 1 output feeds Block 2 input; shape repetition is not value repetition. **Only bank's Block 1 numerical output was calculated.** Block 2's numerical results were not calculated or claimed. Position handling and dropout remain omitted.

Final LayerNorm preserves shape 3 × 4. For next-token generation, take the last position's row (1 × 4), then multiply by a learned width × vocabulary matrix. With five vocabulary entries, (1 × 4) × (4 × 5) produces one row of five logits. WO within attention and this final vocabulary projection are different operations.

### Vocabulary, next-token selection, and stopping

The vocabulary is the tokenizer's complete set of token entries, not the tokens appearing in one prompt. A model scores all entries so it can continue with tokens absent from the input. Token combinations can express words not represented by a single entry; a fixed vocabulary does not make sequences predictable.

Sezai connected this to CNN output classes. COCO's 80 detection categories, rather than standard ImageNet's 1,000 classification classes, matched his YOLO reference. Unlike those object categories, tokenizer entries include pieces, punctuation, and special tokens.

An illustrative five-token vocabulary used IDs 0–4 for `money`, ` at`, ` bank`, ` today`, and `.`. Supplied logits `[-1, 0, -2, 2, 1]` give probabilities approximately `[0.032, 0.086, 0.012, 0.636, 0.234]`. These were not computed from Block 2. Greedy selects ID 3; sampling uses the distribution. Append the ID to `[0, 1, 2]` to get `[0, 1, 2, 3]`: length grows, width and vocabulary do not. Ending IDs are vocabulary entries scored the same way; the serving system recognizes a selected ending ID and stops. Adding `[END]` to the toy vocabulary makes six entries for the later training example.

### Long prompts, caching, and chat context — brief tangents

Sezai asked whether all input positions need attention outputs. During prompt processing, yes: earlier updated representations contribute to later blocks. A standard causal head has N(N+1)/2 accessible query–key pairs. GPU operations parallelize the work, but attention pair counts grow quadratically with length. KV caching reuses earlier keys/values during generation; its actual calculations remain to be worked through.

He also asked whether Codex chat history keeps growing. Messages extend conversation history; caching can reuse processing of an unchanged prefix without shortening logical context. Compaction reduces active context while preserving state needed to continue; visible history and active model context need not be identical. Exact cache reuse in this desktop session was not inspected. He deferred compaction as a separate topic. References: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [Codex prompting guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide). Persisted handoff notes preserve exact examples across context changes and computers.

### Training targets: the text supplies the labels

Sezai connected the full forward/loss/backpropagation/optimizer loop to detector training. For next-token pretraining, target IDs come from text shifted one token ahead, rather than manual boxes/classes. Causal attention prevents the model from seeing its future target. Training typically calculates vocabulary logits and losses at every input position, not just the last generation position.

For one unweighted target position, cross-entropy is `-ln(probability of the actual next token)`. Target probability 0.10 gives loss about 2.303; 0.80 gives about 0.223. PyTorch takes raw logits and target IDs, using a numerically stable log-softmax calculation. Source: [CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

With probabilities `[0.10, 0.10, 0.10, 0.20, 0.50]` and target ` today` (ID 3), loss is about 1.609. The per-logit gradient is predicted probability minus the one-hot target: `[0.10, 0.10, 0.10, -0.80, 0.50]`. Subtracting a negative gradient would increase the target logit if logits were directly adjustable. Actual backpropagation carries gradients to parameters through the network. Sezai said the direction made sense but he did not fully understand the detail; no derivation or independent calculation was checked.

From-scratch training normally includes embeddings, attention/WO, feedforward matrices/biases, normalization parameters, final projection, and learned position embeddings if used. The optimizer updates trainable parameters supplied to it; not every number necessarily changes each step. Lookup gradients for untied input embeddings involve used rows; freezing, optimizer state, and weight decay affect updates. Ordinary training changes parameter values within a fixed architecture. Sezai correctly summarized the shared CNN/LLM training framework.

Backpropagation was described as the dominant practical gradient-calculation method for the neural networks discussed, separate from the optimizer. [Evolution strategies](https://arxiv.org/abs/1703.03864) and [Forward-Forward](https://arxiv.org/abs/2212.13345) were brief research tangents, not mainstream alternatives adopted for this plan. Sezai wanted the standard loop kept central.

### Dataset and hardware discussion — no implementation yet

A trainable bigram model uses only the immediately preceding token, with a vocabulary × vocabulary score table, and can teach the loss/update loop without attention. A proposed tiny character transformer had one block, width 32, one head, feedforward width 64, and context length 32. These are proposals, not built models or measured results.

Sezai reports an Apple M5 computer and access to an RTX 3090. A tiny model can start on CPU; MPS availability would need local verification before Apple GPU use. No hardware benchmark was run. Source: [PyTorch MPS documentation](https://docs.pytorch.org/docs/2.14/notes/mps.html).

**Reserve The Verdict for following the book.** Sezai finds Tiny Shakespeare overused and asked for something fresher. Alternatives discussed were a narrow subset of synthetic educational text from [Cosmopedia v2 in SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), or explicitly fictional synthetic data-platform event logs. Neither was selected, downloaded, or generated. A small model may learn formatting/local patterns without producing useful explanations. Character targets are the actual next characters; use separate held-out text to check generalization. A short sequence can demonstrate memorization without establishing general language ability.

Sezai asked about the six-hour weekly budget. October 3–7 spans five days; the assistant's rough 3–5 active-study-hour estimate was **not measured**, excludes idle/build waiting, and must not be treated as a study log. Preparation counts toward the budget; book reading, coding, and writing will share it.

### Latest stopping point: training windows and batches 🌱

Use the illustrative six-entry vocabulary:

```text
money at bank today . [END]
  0    1   2    3   4   5

Training sequence length: 4
Input:  [0, 1, 2, 3]
Target: [1, 2, 3, 4]
```

| Input position | Accessible context | Target |
| --- | --- | --- |
| 0 | `money` | ` at` |
| 1 | `money at` | ` bank` |
| 2 | `money at bank` | ` today` |
| 3 | `money at bank today` | `.` |

One example produces 4 × 6 vocabulary scores and four target IDs. Loss is calculated at all four positions. Several examples form a batch; average loss, backpropagate, then take an optimizer step. Batch shapes and window construction have only been introduced, not worked through.

**Unanswered check:** the input contains `today`, but can the position at `bank` attend to it during training? Sezai stopped for rest before answering. Resume here, then build a tiny batch. Avoid jumping to implementation or claiming that the detailed cross-entropy gradient is mastered.

## Understanding checklist and remaining questions

Checked items reflect Sezai's explanations or explicit confirmations, not merely topics the assistant introduced.

- [x] Word count is not a fixed token count; tokens include fragments and punctuation.
- [x] Tokenization with a fixed configuration is deterministic.
- [x] Initial embeddings (vectors) belong to the model and are learned parameters.
- [x] The same token row can lead to different contextual vectors because surrounding inputs differ.
- [x] Generation appends IDs without needing intermediate text decoding.
- [x] Ending tokens are learned and predicted too.
- [x] Explain that sampling can choose a lower-probability valid token and that the choice affects later context; select a token using the interval example.
- [x] Identify previous-block representations as the input to the next block; recognize fixed architecture depth during prediction.
- [x] Distinguish width from sequence length after correcting the initial shape answer; explain that additional token rows increase work without changing model parameters.
- [x] Recognize WQ/WK/WV as learned weights and identify the higher query–key score in the toy example.
- [ ] Independently explain temperature and why one decoding strategy might be chosen; top-k/top-p remain uncovered.
- [x] Follow the supplied two-position query–key scores, scaling, softmax, and weighted-value arithmetic; explicitly confirmed in the evening.
- [x] Explain that the toy width is two and that incoming vectors multiply learned projection matrices; correct V_at to `[0, 0]` after the column-wise calculation was shown.
- [x] Explain the combined attention flow in his own words and identify the final result as a weighted sum of value vectors; matrix arithmetic was guided and corrections remain recorded.
- [ ] Explain stored embeddings and projection matrices versus temporary Q/K/V and contextual outputs, and distinguish training updates from inference calculations; clarification helped, but independent understanding remains to be checked.
- [x] Recognize the ResNet residual connection and the toy residual result under identity WO.
- [x] Identify the pre-normalization incoming representation used by the residual path and independently calculate the width-4 bank query–key self-score of 4.
- [ ] Explain LayerNorm per token, including calculated statistics and learned scale/shift; the first example was supplied, not yet checked.
- [x] Connect the completed toy block output to the next block's incoming representation and distinguish temporary outputs from learned embeddings/transformations.
- [x] Understand how the expansion produces a 1 × 6 row, and connect next-token vocabulary scores to the CNN class-score analogy.
- [x] Explain the shared forward → loss → backpropagation → optimizer training framework for CNNs and language models.
- [ ] Revisit the usefulness of feedforward transformations beyond the illustrative sparse matrices; work through multi-head attention and a non-identity WO.
- [ ] Independently explain cross-entropy and derive/check its logit gradient; direction was discussed, detailed understanding remains open.
- [ ] Answer the causal-mask training-window check and work through batched input/target/score shapes.
- [ ] Explain KV cache with a worked appended-token example.
- [ ] Return to expert count, routing, and specialization after the attention fundamentals.
- [ ] Implement and inspect a runnable tokenization/model example; only the assistant ran tiktoken so far.

No chapter reading, tensor practice, daily-capacity exercise, or independent benchmark has been completed or claimed here.

## Further references

- [Book and companion resources](https://sebastianraschka.com/llms-from-scratch/)
- [Author's implementation](https://github.com/rasbt/LLMs-from-scratch)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original transformer paper; detailed equations can wait until the conceptual walkthrough is clear.

These notes summarize our conversation and use explicitly illustrative examples; they are not excerpts from the book.
