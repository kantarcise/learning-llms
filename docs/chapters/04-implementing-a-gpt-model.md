# Chapter 4 — Implementing a GPT Model from Scratch to Generate Text

[Chapter 2](02-working-with-text-data.md) prepared text as token IDs and embedding vectors. [Chapter 3](03-coding-attention-mechanisms.md) built masked multi-head attention. Now we assemble these pieces into a **GPT-like language model**: a model that produces scores for the next token at every input position.

These notes follow Sebastian Raschka's [Build a Large Language Model (From Scratch)](https://sebastianraschka.com/llms-from-scratch/) and the [author's Chapter 4 implementation](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04). The chapter develops normalization, feedforward networks, shortcut connections, and transformer blocks, then puts them together and examines parameter counts, storage requirements, and text generation. Unwritten sections below remain placeholders.

### Where we are in the book's roadmap

We are at **step 3, “LLM architecture,” within Stage 1**. We already built the attention component; this chapter builds the model around it. Chapter 5 adds the training loop and pretraining.

<figure markdown="1">

[![The author's roadmap: data preparation, attention, and LLM architecture lead to pretraining and then fine-tuning. Chapter 4 focuses on step 3, LLM architecture.](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)](https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg)

<figcaption markdown="span">Sebastian Raschka's [book roadmap](https://sebastianraschka.com/llms-from-scratch/), embedded from his website. As in Figure 4.1, our focus is step 3 of Stage 1: assembling the architecture. Open the image for a full-size view.</figcaption>
</figure>

## 4.1 Coding an LLM architecture

**GPT** stands for **Generative Pre-trained Transformer**. During generation, it takes the available token sequence, computes scores for possible next tokens, and uses a selection rule to choose one. We append that token and repeat. The model's `forward` calculation produces scores; the surrounding generation code handles selection and repetition.

### From an attention module to a whole model

The architecture is easier to follow when we separate its nested parts:

- **The GPT model** contains embedding layers, a stack of transformer blocks, final normalization, and a vocabulary output projection.
- **Each transformer block** contains masked multi-head attention and a feedforward network, with normalization and residual (shortcut) connections around these computations.
- **Multi-head attention** is the component we implemented in Chapter 3. It is inside a transformer block, not the entire block.

<figure markdown="1">

[![GPT data flow from text through token IDs, token plus position embeddings, repeated transformer blocks, final normalization, and vocabulary logits. Generation selects a token from the last position's scores and appends it to the input.](../assets/chapter-04/gpt-architecture-overview.svg)](../assets/chapter-04/gpt-architecture-overview.svg)

<figcaption markdown="span">Original companion illustration of the architecture introduced in the book's Figure 4.2, with our own layout and an explicit separation between the model and generation code. This is an overview; the following sections implement the internals. Example text is illustrative, not output from a trained model. Open the image for a full-size view.</figcaption>
</figure>

Many parts repeat, which makes a large model's structure manageable. **Repeated structure does not mean shared weights:** successive transformer blocks normally have separate learned parameters. Each block receives the updated vectors from the previous block.

This is the same overall organization we explored in [Character Logs](../preparation/training-comparison.md#model-architecture-drawn-from-the-code) and [Cosmopedia](../preparation/cosmopedia.md). Character Logs has one transformer block with one attention head; Cosmopedia has four transformer blocks with four heads per block. The book now walks through the building blocks systematically, using GPT-2 as its reference architecture.

### Scaling up the example: GPT-2 small

Earlier calculations used short vectors so that we could follow every number. The book now targets the smallest GPT-2 variant, commonly called **GPT-2 small / 124M**. Its vector width is **768**, and it has **12 transformer blocks**, each with **12 attention heads**. These are different architectural choices: 12 heads are inside each of the 12 blocks.

The GPT-2 tokenizer vocabulary contains **50,257 tokens**, not necessarily whole words. The token embedding table therefore has 50,257 rows and 768 coordinates per row. **Vocabulary size tells us how many rows; width tells us how many coordinates per row.**

!!! note "GPT-2 parameter-count names"

    The original GPT-2 report used counts such as 117M and 345M; OpenAI later [acknowledged a counting error](https://github.com/openai/gpt-2#readme). The [released checkpoints](https://github.com/openai/gpt-2/blob/master/DEVELOPERS.md) are named **124M, 355M, 774M, and 1558M**. Older counts such as 345M, 762M, and 1542M can still appear in explanations. We will use the released checkpoint names when discussing weight loading.

    “124M” names the reference model, not a guarantee that every teaching implementation has exactly that count. Bias choices and whether input embeddings and the output projection share parameters affect the total; we will count our implementation once it is assembled.

### What exactly counts as a parameter?

This is the same idea as in a convolutional neural network: **parameters are the stored, trainable numbers** that the optimizer updates. Embedding-table entries, attention projection weights, feedforward weights and biases, and normalization's learned scale and shift are all examples.

A weight matrix with 2,048 rows and 2,048 columns contains:

```text
2,048 × 2,048 = 4,194,304 learned numbers
```

That counts the **matrix alone**. A linear layer with an additional bias of length 2,048 would have 4,196,352 parameters in total. The input vectors, computed queries/keys/values, attention weights, and output logits are activations calculated using those parameters; they do not each add new trainable parameters.

!!! tip "Why learn with GPT-2 rather than GPT-3?"

    GPT-2 has [publicly released pretrained weights and reference code](https://github.com/openai/gpt-2), so we can build the architecture and later load real learned parameters. Its smallest variant is also much more manageable for local experiments than the largest GPT-3 model (175 billion parameters).

    Both use the decoder-only transformer approach, so the building blocks we learn transfer. That does not make every architectural detail identical, or make training a GPT-2 model to its original quality a quick laptop task. Implementing the model, running inference, and fully pretraining it have very different compute requirements.

For the later weight-loading step, follow [Section 5.5](05-pretraining-on-unlabeled-data.md#55-loading-pretrained-weights-from-openai) and the [author's Chapter 5 code](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05). The introductory passage in this PDF points to Chapter 6, but the companion's section map places weight loading in 5.5; Chapter 6 covers classification fine-tuning.

## 4.2 Normalizing activations with layer normalization

Notes to come as we read this section.

## 4.3 Implementing a feed forward network with GELU activations

Notes to come as we read this section.

## 4.4 Adding shortcut connections

Notes to come as we read this section.

## 4.5 Connecting attention and linear layers in a transformer block

Notes to come as we read this section.

## 4.6 Coding the GPT model

Notes to come as we read this section.

## 4.7 Generating text

Notes to come as we read this section.

## Questions to bring to the chapter

<!-- Add actual questions during reading. -->

## End of chapter

<!-- Summarize what became clear, link exercises, and record unresolved questions. -->
