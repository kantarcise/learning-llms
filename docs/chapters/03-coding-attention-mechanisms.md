# Chapter 3 — Coding Attention Mechanisms

Status: reading started. These notes cover the chapter introduction and sections 3.1–3.2, and the context-vector goal, score calculation, and first normalization in 3.3.1. The remaining attention calculations and sections 3.4–3.6 are still ahead.

[Chapter 2](02-working-with-text-data.md) prepared the inputs: text became token IDs, training windows, and vectors carrying token and position information. Now we move into the model architecture itself and study **attention**: how a token's representation gathers information from other accessible positions.

These notes follow Sebastian Raschka's [Build a Large Language Model (From Scratch)](https://sebastianraschka.com/llms-from-scratch/). The [author's Chapter 3 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03) accompanies the implementation.

Section names and numbering follow [Manning’s published contents](https://www.manning.com/preview/build-a-large-language-model-from-scratch/chapter-3). Numbered headings are book sections; unnumbered headings contain our companion notes. Unwritten sections are placeholders, not completed reading.

### Where we are in the book's roadmap

We are at **step 2, “Attention mechanism,” within Stage 1** of the author's roadmap. This is not Stage 2, which concerns pretraining and the foundation model. Data preparation gave us the inputs; attention is the next architectural component to unpack.

<figure markdown="1">

[![The author's roadmap from data preparation and attention to a foundation model and fine-tuning. Chapter 3 focuses on step 2, Attention mechanism, in Stage 1.](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)

<figcaption markdown="span">Sebastian Raschka's [book roadmap](https://sebastianraschka.com/llms-from-scratch/), embedded from his website. We are focusing on step 2, “Attention mechanism.” Open the image for a full-size view.</figcaption>
</figure>

### Four implementations, one gradual progression

The chapter introduces four variations. We will follow the progression rather than unpacking the most advanced version immediately.

<figure markdown="1">

[![Simplified self-attention leads to trainable projections, then causal masking, then multiple attention heads.](../assets/chapter-03/attention-progression.svg)](../assets/chapter-03/attention-progression.svg)

<figcaption markdown="span">Original companion figure explaining the same progression as Figure 3.2 in Sebastian Raschka's book, using our own layout and wording. These are cumulative teaching steps, not mutually exclusive categories: the final GPT-style multi-head attention also has trainable weights and a causal mask.</figcaption>
</figure>

A **head** is one attention calculation; multiple heads perform several such calculations in parallel and combine their outputs. Heads are not successive transformer blocks. That is enough terminology for the introduction—we will work through the details in section 3.6.

Our preparation already touched queries, keys, values, and causal masking. The book now gives us an ordered path to implement them. Sections 3.1 and 3.2 first motivate attention before the simplified calculation in 3.3.

## 3.1 The problem with modeling long sequences

Starting with the problem makes the architecture easier to motivate. Translation cannot generally be handled by replacing each source word with its dictionary equivalent and keeping the same order. The target sentence needs the appropriate meaning and grammar, and the relevant source information may occur earlier or later than the word currently being generated.

### Translation is not a row of independent substitutions

Consider our own German-to-English example:

```text
German:                    Ich habe das Buch gelesen.
Literal, source-order:     I have the book read.
Natural English:           I have read the book.
```

The final German word, `gelesen`, contributes `read`, which appears before `the book` in the English translation. This short example isolates reordering; longer sentences also require resolving meaning and relationships across much greater distances. Not every translation needs reordering, but a general translation system must support it.

<figure markdown="1">

[![Word substitution preserves the German order and produces I have the book read; aligning the same words in English order produces I have read the book.](../assets/chapter-03/translation-order.svg)](../assets/chapter-03/translation-order.svg)

<figcaption markdown="span">Original companion illustration inspired by the problem in the book's Figure 3.3, using a different sentence and our own layout. Arrows show illustrative word correspondences, not measured attention scores. Words are used for readability; actual models operate on tokens.</figcaption>
</figure>

Having access to a later word in the **source sentence** is not the same as peeking at a future **target answer**. A translation encoder can read the complete source before decoding begins, while the decoder generates the target from its available prefix. This connects to the [encoder–decoder distinction in Chapter 1](01-understanding-large-language-models.md#the-original-translation-model-two-streams).

### The basic recurrent encoder–decoder

Before Transformers, recurrent neural networks (RNNs) were used for neural translation. In the basic design discussed here, an **encoder** processes the source sequence and a **decoder** generates the target sequence. These are roles in the model, not the tokenizer's `encode` and `decode` methods.

The encoder reads one token representation at a time and updates its **hidden state**: a vector of computed values carrying information from the sequence processed so far. The state is an activation, not a new set of learned weights. The same recurrent computation and learned weights are reused at each step.

```text
Current token vector + previous encoder state → updated encoder state
```

After the final source token, the last encoder state supplies the source summary used to initialize the decoder in this schematic. The decoder then updates its own state and predicts target tokens one at a time, using the previous target token as input (a start marker is used first).

<figure markdown="1">

[![The encoder updates hidden states h1 through h5; only h5 initializes the decoder, which updates states s1 through s5 and generates the translated tokens.](../assets/chapter-03/rnn-context-bottleneck.svg)](../assets/chapter-03/rnn-context-bottleneck.svg)

<figcaption markdown="span">Original companion illustration of the basic recurrent encoder–decoder bottleneck discussed in Figure 3.4. Encoder and decoder boxes represent steps through time, not separate layers with independent weights. This simplified design has no attention; outputs are illustrative, not measured results. Recurrent variants can carry additional state, such as an LSTM's cell state.</figcaption>
</figure>

### Why a fixed-size summary becomes a bottleneck

My initial interpretation was that we are trying to keep the whole context in a very small thing. More precisely, **the state has a fixed width while the input can keep getting longer**. It might contain hundreds or thousands of numbers; the issue is that more source information must be carried through the same-sized representation.

For illustration, a 256-coordinate encoder state still has 256 coordinates after reading either 5 or 100 tokens. Each update must preserve information that will be useful later. Earlier details can become difficult to retain and recover, particularly when the translation depends on distant parts of a long sentence. This is a learned compression bottleneck, not a literal rule that each coordinate stores one word or that a particular length always fails.

In this **basic model without attention**, the decoder has no direct route back to individual earlier encoder states. It does not repeatedly read an unchanging decoder state: its own state evolves as it generates. But the source information available through that state must have survived the encoder's final summary and subsequent decoder updates.

This limitation is not true of every RNN. Attention was introduced into recurrent encoder–decoder systems before the Transformer: the [Bahdanau, Cho, and Bengio paper](https://arxiv.org/abs/1409.0473) motivates attention as a way to consult relevant source representations instead of relying exclusively on one fixed-length source summary.

That gives us the question for attention: **when producing a particular output token, could the decoder consult the source positions it needs?** Section 3.2 develops that idea. We do not need to implement an RNN first to follow the motivation.

## 3.2 Capturing data dependencies with attention mechanisms

The fixed-summary encoder–decoder can work well on short sequences, but retaining and recovering relevant details becomes harder as sequences grow. This is not a sharp length cutoff or a statement that every RNN fails on long text. The limitation here is the basic decoder's lack of direct access to the earlier source representations.

Attention adds that access. The encoder still builds representations for the source positions, but the decoder can consult them again while producing each target token. It does not need every source detail to survive exclusively in the final encoder state.

### Consult different source positions at each output step

Return to our example, `Ich habe das Buch gelesen` → `I have read the book`. Suppose the generated target prefix is `I have`, and we are about to generate `read`.

The decoder can compute mixing amounts over the source representations, combine those representations into a **context vector**, and use that vector to help produce the next output. In our illustration, the representation associated with `gelesen` receives the largest weight. These numbers are invented to explain the mechanism, not extracted from a model.

<figure markdown="1">

[![All five encoder states contribute to a weighted context vector used to generate read after I have; the illustrative weight for gelesen is largest.](../assets/chapter-03/rnn-with-attention.svg)](../assets/chapter-03/rnn-with-attention.svg)

<figcaption markdown="span">Original companion illustration inspired by Figure 3.5, extending our previous recurrent encoder–decoder drawing. Thicker arrows indicate larger illustrative mixing weights. The decoder computes a new mixture at each output step. This is a conceptual sketch, not the exact Bahdanau implementation; start/end markers, token feedback wiring, and some internal operations are omitted.</figcaption>
</figure>

The context vector still has a fixed width. The improvement is that **its contents can be recomputed from all available encoder states for each output step**, rather than requiring one source summary to serve every prediction. The model retains multiple source representations, so this extra access also requires storage and computation.

“Access to words” is shorthand: the attention calculation works with their **encoder representations**, not raw text or a dictionary. Each encoder representation can already contain contextual information. The mixture is usually soft: several positions can contribute, rather than the model selecting exactly one word.

The mixing weights depend on the current computation. They are not permanent importance scores attached to source words, and a large weight is not a complete explanation of why a model chose an output. We will calculate attention weights explicitly in the following sections.

### From recurrent attention to the Transformer

The [Bahdanau, Cho, and Bengio paper](https://arxiv.org/abs/1409.0473) was first posted in **2014** and presented at ICLR 2015. It introduced an influential attention-based neural translation approach that lets a recurrent decoder use relevant source representations. This is the historical example here, not a claim that every idea called attention originated in that paper.

In **2017**, [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduced the Transformer, showing that strong translation models could be built without recurrent layers. Attention handled interactions between positions alongside feedforward layers and other components. “Without recurrence” does not mean “without sequence order”: positional information and causal masking still matter.

The progression is:

```text
Basic recurrent encoder–decoder
    → one final source summary

Recurrent encoder–decoder with attention
    → consult source representations for each output step

Transformer
    → use attention-based blocks without recurrent layers
```

The translation example has a decoder attending to an encoder's representations. That is attention **between two sequences**. The self-attention calculation we implement next mixes positions **within one sequence**. The shared idea is a context-dependent weighted combination; the architectures and exact calculations differ.

!!! tip "What self-attention adds"

    **Self-attention builds context-aware representations by letting each position in a sequence weigh and combine information from positions in that same sequence.** Without a mask, all positions can interact; with a causal mask, each position can use only itself and earlier positions.

    Here, “better representations” means that a token's vector can reflect its surrounding context. It does not mean that attention necessarily takes less computation.

Before calculating attention, locate it in the complete model. Chapter 2 prepared input vectors from text. This chapter develops the attention mechanism that mixes information between positions. Chapter 4 assembles the full GPT-like architecture around it.

<figure markdown="1">

[![Text becomes input vectors, then enters a GPT-like transformer with self-attention and other components; vocabulary scores lead to token selection and decoded text.](../assets/chapter-03/attention-in-model.svg)](../assets/chapter-03/attention-in-model.svg)

<figcaption markdown="span">Original companion illustration inspired by Figure 3.6 in [Sebastian Raschka's book](https://sebastianraschka.com/llms-from-scratch/), using our own layout and example. This is a learning map, not an exact execution graph: normalization, residual connections, and feedforward operations surround attention within each block. The input phrase is illustrative; no tokenization or prediction result is claimed.</figcaption>
</figure>

**Attention receives vectors and produces vectors.** It does not directly produce words or token IDs. The complete model eventually projects its representations into vocabulary scores; selecting token IDs and decoding them gives us generated text.

## 3.3 Attending to different parts of the input with self-attention

!!! note "What does the self in self-attention mean?"

    **Self means that the positions being related belong to the same sequence.** It does not mean a token attends only to itself. A token can incorporate information from other positions as well as its own.

    In the translation example from section 3.2, a decoder attends to representations from a separate encoder sequence. That is **cross-attention**. Both are attention mechanisms; the distinction is where their inputs come from. Image models can likewise use self-attention between representations of image patches.

### 3.3.1 A simple self-attention mechanism without trainable weights

The goal is to turn each input vector into a **context vector**: a representation that incorporates information from the other available positions, including the position itself. Calling it an “enriched embedding” is a useful intuition. In this simplified calculation, it is a weighted combination of the input vectors, not an extra row permanently written into the token embedding table.

Keep these symbols separate:

| Symbol | Meaning | Small example |
| --- | --- | --- |
| `T` | Number of token positions in this sequence | 4 positions |
| `d` | Number of coordinates in each input vector | 3 numbers per vector |
| x<sup>(2)</sup> | Input vector at the second position; the number is a position, not an exponent | `[0, 2, 1]` |
| z<sup>(2)</sup> | Context vector computed for the second position | Also 3 numbers here |
| α<sub>23</sub> | Mixing weight for input position 3 when computing the output for position 2 | One scalar, not a vector |

The book uses six word-labeled positions for “Your journey starts with one step.” Its input tensor therefore has six rows, each containing three numbers. In general, **the input shape is `[T, d]`: positions × coordinates per vector**. This is one sequence, with no batch axis in this example. Word labels help us read the example; a real tokenizer may split words into multiple tokens.

For our companion drawings, use four word-labeled positions and deliberately simple, invented vectors. These are teaching inputs, not representations from a trained model:

```python
import torch

# One sequence: T=4 positions, d=3 coordinates per vector.
# These are floating-point vectors already, not integer token IDs.
inputs = torch.tensor([
    [1.0, 0.0, 1.0],  # Small   — position 1
    [0.0, 2.0, 1.0],  # models  — position 2
    [1.0, 1.0, 0.0],  # learn   — position 3
    [0.0, 1.0, 2.0],  # patterns — position 4
])
```

<figure markdown="1">

[![Four three-coordinate vectors are weighted separately and summed to form the context vector for the second position, models.](../assets/chapter-03/self-attention-context.svg)](../assets/chapter-03/self-attention-context.svg)

<figcaption markdown="span">Original companion illustration inspired by Figure 3.7 in [Raschka's book](https://sebastianraschka.com/llms-from-scratch/), using a different sentence and invented vectors. The weights are symbolic: this drawing shows the goal before we calculate the mixing amounts. All four positions are accessible because this example has no causal mask.</figcaption>
</figure>

We focus on z<sup>(2)</sup> first, but the eventual goal is **one context vector per input position**. There is no concatenation here: multiplying each three-coordinate vector by a scalar and summing the results still produces three coordinates. Four input positions will ultimately produce four context vectors.

**First calculate scores, not mixing amounts.** Use the second input vector as the query—the position for which we want context—and take its dot product with each input vector. In this simplified version, “query” describes the role of an input vector; we have not introduced a learned query projection yet.

```python
# Python index 1 selects position 2: [0, 2, 1], shape [3].
query = inputs[1]

# For each row, multiply matching coordinates and add them.
# inputs has shape [4, 3]; query has shape [3]; scores has shape [4].
scores = inputs @ query
print(scores)  # tensor([1., 5., 2., 4.])
```

<figure markdown="1">

[![The query vector zero two one is compared with each input by dot product, producing scores one five two four.](../assets/chapter-03/self-attention-scores.svg)](../assets/chapter-03/self-attention-scores.svg)

<figcaption markdown="span">Original companion illustration inspired by Figure 3.8, using the same invented inputs as our preceding figure. The displayed scores are exact for these inputs. A score is an intermediate comparison value; it is not yet a normalized attention weight.</figcaption>
</figure>

For example, the score comparing position 2 with position 3 is `0×1 + 2×1 + 1×0 = 2`. These dot products depend on the vectors and their magnitudes; with invented inputs, we should not read semantic understanding into the resulting rankings.

!!! tip "Attention weights are not the same as trainable model weights"

    This section has no learned query/key/value matrices. It still computes attention scores and, subsequently, mixing weights from the inputs. Those mixing weights are computed activations, not stored parameters adjusted directly by the optimizer. In a full model, the input embeddings can themselves be learned even when a particular attention operation has no trainable projections.

The book's figures shorten the displayed decimal values to save space. Two rows can therefore look identical in a drawing while differing in the actual tensor. Use the full values from the code for calculations; do not expect dot products of truncated display values to reproduce the original scores exactly.

#### Reading the book's notation

The raised parenthesized number is a **position label, not a power**: x<sup>(2)</sup> means the input vector at position 2. More generally, x<sup>(T)</sup> is the last input vector when the sequence has T positions. The corresponding context vector is z<sup>(2)</sup>.

- **ω (omega)** denotes an attention score: ω<sub>23</sub> = x<sup>(2)</sup> · x<sup>(3)</sup>.
- **α (alpha)** denotes a normalized attention weight: α<sub>23</sub> is how much position 3 contributes to the context vector for position 2.
- The first subscript identifies the query position; the second identifies the contributing position. These figures use omega and alpha, not theta.

A dot product multiplies matching coordinates and sums the products. It measures alignment **and depends on vector magnitudes**. If we double a vector without changing its direction, its dot product with the query doubles. So a higher score is not automatically evidence of a closer semantic relationship. In this example, larger positive scores will receive larger mixing weights after normalization.

#### From scores to weights that sum to one

For our query at position 2, the scores are `[1, 5, 2, 4]`. A first, simple normalization divides each score by their sum, `12`:

```python
# scores: shape [4], one scalar per input position for this query.
# All four scores in this toy example are positive.
attention_weights = scores / scores.sum()
print(attention_weights)  # tensor([0.0833, 0.4167, 0.1667, 0.3333])
print(attention_weights.sum())  # tensor(1.)
```

<figure markdown="1">

[![Scores omega 21 through omega 24 are divided by their sum 12 to obtain weights alpha 21 through alpha 24, summing to one.](../assets/chapter-03/self-attention-normalization.svg)](../assets/chapter-03/self-attention-normalization.svg)

<figcaption markdown="span">Original companion illustration inspired by Figure 3.9 in [Raschka's book](https://sebastianraschka.com/llms-from-scratch/). We retain our four-position example and show exact fractions to avoid rounding ambiguity. This figure demonstrates simple division by the score sum, not softmax.</figcaption>
</figure>

The weights describe how to distribute the mixture across the input vectors: position 2 receives `5/12` of the weight, position 4 receives `4/12`, and so on. This makes the combination interpretable as a weighted average and prevents its scale from growing merely because we add more equally weighted contributors. **This normalization is across positions for one query; it is not LayerNorm across a token vector's coordinates.**

!!! warning "This first normalization is only a teaching step"

    Arbitrary dot products can be negative, and their sum can be zero or very close to zero. Dividing by that sum is therefore not a general solution for attention. Softmax will give us nonnegative weights summing to one even when scores are negative. Normalization alone does not guarantee stable training.

We now have mixing weights. Using them to calculate the context vector is the next part of the walkthrough.

## 3.4 Implementing self-attention with trainable weights

Notes to come as we read this section.

## 3.5 Hiding future words with causal attention

Notes to come as we read this section.

## 3.6 Extending single-head attention to multi-head attention

Notes to come as we read this section.

## Questions to bring to the chapter

- How do multiple heads differ from multiple transformer blocks in the code? Revisit this in section 3.6, after the single-head calculation is clear.

## End of chapter

<!-- Summarize what became clear, link exercises, and record unresolved questions. -->
