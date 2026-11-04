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

### The configuration: seven choices, seven different meanings

The book collects the model settings in one dictionary. These are **hyperparameters we choose**, not learned weight values. The following examples adapt Listing 4.1 and the surrounding examples from the [author's Chapter 4 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb), adding type hints, comments, and explicit shapes.

```python title="Configuration for the GPT-2-small teaching architecture"
from typing import TypedDict


class GPTConfig(TypedDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool


GPT_CONFIG_124M: GPTConfig = {
    "vocab_size": 50_257,
    "context_length": 1_024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

`TypedDict` describes the expected keys and value types for readers and type checkers. The actual object is still an ordinary Python dictionary; this annotation does not validate it at runtime.

| Setting | Meaning in this model | Concrete consequence |
| --- | --- | --- |
| `vocab_size = 50_257` | Number of token IDs in the GPT-2 vocabulary | Token IDs range from 0 to 50,256. The embedding table has 50,257 rows; the output has 50,257 scores per position. |
| `context_length = 1_024` | Maximum number of positions the model supports | The learned position table has 1,024 rows. A four-token input uses only rows 0–3. This is not the batch size or vector width. |
| `emb_dim = 768` | Coordinates in each token representation | Both embedding tables have 768 columns. Transformer blocks preserve this width. |
| `n_heads = 12` | Attention heads **inside each** transformer block | The real attention module splits width 768 into 12 heads of width 64 each. The placeholder does not perform this operation yet. |
| `n_layers = 12` | Number of successive transformer blocks | The model sends its vectors through 12 blocks in sequence. This does not mean 12 attention heads total. |
| `drop_rate = 0.1` | Dropout probability during training | Each affected element has a 10% chance of being zeroed; survivors are scaled by `1 / 0.9`. This is not a guarantee that exactly 10% vanish in each call. `model.eval()` disables dropout. |
| `qkv_bias = False` | Whether the query, key, and value projections have additive bias vectors | We still learn all three weight matrices. This setting does not disable biases throughout the model. It will matter when we implement the real blocks and later load pretrained weights. |

The name `GPT_CONFIG_124M` describes the architecture we are working toward. **The dummy model is not yet the complete 124M-parameter model:** its placeholder blocks have no attention or feedforward parameters.

### First build the outline, then fill in its components

<figure markdown="1">

[![Implementation order: GPT placeholder backbone, normalization, GELU, feedforward network, shortcuts, transformer block, and complete GPT architecture. Multi-head attention from Chapter 3 joins the transformer block.](../assets/chapter-04/implementation-order.svg)](../assets/chapter-04/implementation-order.svg)

<figcaption markdown="span">Original companion illustration of the development order in Figure 4.3. These numbers describe the order we implement components, not the order that tensors pass through layers. GELU will be inside the feedforward network; shortcuts will wrap sublayers.</figcaption>
</figure>

`DummyGPTModel` **inherits from** `nn.Module`. This gives PyTorch a way to register its child modules and parameters, move them between devices, and propagate training/evaluation mode. It is an ordinary subclass, not an abstract class. Calling `model(batch)` uses PyTorch's module machinery to invoke `forward(batch)`.

We define the placeholder classes first so this code can be read and run from top to bottom. For all three classes below:

- `B` = batch size: how many examples.
- `T` = sequence length: how many token positions per example.
- `D` = embedding width: how many coordinates per token vector.
- `V` = vocabulary size: how many candidate next tokens.

```python title="Placeholder components and DummyGPTModel — adapted from Listing 4.1"
import torch
from torch import nn


class DummyTransformerBlock(nn.Module):
    """Pass floating-point vectors [B,T,D] through unchanged.

    B = examples, T = token positions, D = vector width.
    No attention, normalization, feedforward network, or shortcuts yet.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        # Accept the future block's configuration, but do not use it yet.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the same [B,T,D] tensor: examples, positions, coordinates."""
        return x


class DummyLayerNorm(nn.Module):
    """Placeholder for normalization; it does not normalize anything yet."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        # Arguments mirror the interface we will need; neither is used yet.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return floating-point [B,T,D] unchanged: examples, positions, width."""
        return x


class DummyGPTModel(nn.Module):
    """Assemble the model outline with identity placeholders inside it."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        # Independent learned tables: token identity versus sequence position.
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Construct 12 distinct block objects; Sequential calls them in order.
        # The * unpacks the list into individual constructor arguments.
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # At every position, map D coordinates to V vocabulary scores.
        # This output "head" is not an attention head.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Map torch.long token IDs [B,T] to floating-point logits [B,T,V].

        B = examples, T = positions, V = vocabulary size, D = vector width.
        IDs must be in [0,V); 1 <= T <= the configured context length.
        Inputs and model must be on the same device. These are caller
        requirements, not checks performed by the type annotations.
        This computes activations; it does not update learned parameters.
        """
        _batch_size, seq_len = in_idx.shape
        # Each ID retrieves a row: [B,T] -> [B,T,D].
        tok_embeds = self.tok_emb(in_idx)

        # arange makes integer positions [0, ..., T-1], not embeddings.
        positions = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(positions)  # [T] -> [T,D]

        # Broadcast the same position rows across B examples; add, not concat.
        x = tok_embeds + pos_embeds  # [B,T,D]
        x = self.drop_emb(x)  # Random dropout in train mode; identity in eval.
        x = self.trf_blocks(x)  # Currently unchanged by all dummy blocks.
        x = self.final_norm(x)  # Currently unchanged: no normalization yet.
        logits = self.out_head(x)  # [B,T,D] -> [B,T,V]
        return logits
```

`nn.Sequential` does not itself invent attention or a feedforward network. It just calls its contained modules in order. Since every dummy block returns `x`, even twelve of them leave `x` unchanged. The embedding tables and output projection are real trainable layers; the dropout layer is real too.

!!! warning "Runnable does not mean a working GPT yet"

    This outline can return correctly shaped logits, but it has **no cross-token attention**. Each position's scores depend on its own token and position embeddings (and training-mode dropout), not the preceding tokens. The dummy normalization also leaves values untouched. We are checking the wiring before replacing these placeholders with real computations.

### A batch of two texts: watch the shapes

Use the same two short inputs as the book so the token IDs are easy to compare. Run these snippets in order, in an environment with the repository's `experiments` and `book` dependency groups (PyTorch and tiktoken).

```python
import tiktoken

# Tokenization happens outside the model; embedding lookup happens inside it.
tokenizer = tiktoken.get_encoding("gpt2")
texts = ["Every effort moves you", "Every day holds a"]
encoded = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in texts]
batch = torch.stack(encoded, dim=0)
print(batch)
print(batch.shape)
```

```text
tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])
torch.Size([2, 4])
```

Each text happens to encode to four tokens. `torch.stack` adds a batch axis and requires equal-sized tensors; it does not pad unequal sequences automatically. Both rows begin with ID 6109, so they retrieve the same initial token vector at position zero.

<figure markdown="1">

[![Two texts become IDs of shape 2 by 4. Inside DummyGPTModel, token vectors of shape 2 by 4 by 768 are added to position vectors of shape 4 by 768. After dropout and identity placeholders, the output projection produces scores of shape 2 by 4 by 50257.](../assets/chapter-04/dummy-gpt-data-flow.svg)](../assets/chapter-04/dummy-gpt-data-flow.svg)

<figcaption markdown="span">Original companion figure expanding the input/output overview in Figure 4.4. Here we explicitly distinguish 768-coordinate hidden representations from 50,257-coordinate vocabulary logits and show the actual dummy behavior. Batch examples remain separate.</figcaption>
</figure>

There is an important distinction in Figure 4.4's terminology: the hidden vectors flowing through the transformer retain width **768**, but the complete `DummyGPTModel.forward` returns **50,257 scores per position**, after `out_head`. Those scores are logits, not 768-coordinate context vectors, token IDs, or probabilities.

```python
# Fix initialization randomness, then create the model described above.
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
```

```text
Output shape: torch.Size([2, 4, 50257])
```

For example, `logits[0, 3]` contains 50,257 scores at the fourth position of the first input. In the completed causal GPT, those scores will predict what comes **after** `"Every effort moves you"`, using that whole available prefix. Our dummy model cannot yet use the whole prefix because its attention is missing.

The model does not automatically return the word `"forward"` shown in the book's illustration. That is an intended continuation, not evidence that these randomly initialized placeholders can generate meaningful text. We will add the real components before training or loading pretrained weights.

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
