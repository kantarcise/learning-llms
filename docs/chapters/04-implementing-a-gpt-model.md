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

The difficulty of training deep networks is familiar from CNNs: gradients can become very small or very large as they propagate through many layers, making useful parameter updates harder. The forward-pass values also need care. **Activations** are the values computed as data passes through the network; they are distinct from the learned parameters used to compute them.

Layer normalization helps control the scale of those activations. It often makes optimization more stable and can help training converge. It is one part of the design, alongside choices such as residual connections, initialization, and the optimizer; it does not guarantee stable training or solve every gradient problem by itself.

### What does “mean zero, variance one” mean?

Take one token's vector. Normalization calculates statistics **across the coordinates of that vector**, subtracts their mean, and divides by their standard deviation (with a small numerical-stability adjustment in the implementation).

- **Mean zero:** the coordinates are centered around zero; this does not mean each coordinate becomes zero.
- **Variance one**, or **unit variance:** their spread has a standardized scale; this does not mean every coordinate becomes one.
- **Same shape:** a vector with 768 coordinates still has 768 coordinates afterward.

<figure markdown="1">

[![A layer produces the activation vector 2, 2, 6, 6. Its mean is 4 and variance is 4. Subtracting 4 and dividing by standard deviation 2 gives minus 1, minus 1, 1, 1, with mean zero and variance one.](../assets/chapter-04/layer-normalization-intuition.svg)](../assets/chapter-04/layer-normalization-intuition.svg)

<figcaption markdown="span">Original companion illustration of the normalization idea in Figure 4.5, using our own four-coordinate example. Variance here divides by the number of coordinates. The drawing isolates standardization before learned scale and shift, and omits the small epsilon term for clean arithmetic. These are illustrative values, not recorded model activations.</figcaption>
</figure>

Our example starts with `[2, 2, 6, 6]`. Its mean is 4. Subtracting 4 gives `[-2, -2, 2, 2]`; the original vector's standard deviation is 2, so dividing by 2 gives `[-1, -1, 1, 1]`. The values still differ, but their center and spread have changed. We have transformed the representation, not erased it.

For a GPT tensor shaped `[B, T, D]`, **B is the number of examples, T is the number of token positions, and D is vector width**. Layer normalization in this model works along the last axis, D. Each example's token position gets its own mean and variance. It does not pool statistics across tokens or across the batch, as one might initially imagine from familiarity with batch normalization in CNNs.

!!! note "Standardization is the first part of LayerNorm"

    The full layer also applies a learned scale and shift for each coordinate, which we will implement as we continue Section 4.2. Consequently, its final output does not have to retain mean zero and variance one. The small epsilon term used for numerical stability also makes the standardized variance approximately one rather than exactly one; a constant vector is a special case with zero variance.

    The mean and variance are calculated from the current input vector. The scale and shift are stored model parameters learned during training. These are different kinds of numbers.

### Where it sits in our GPT architecture

“Before and after attention” is a useful first orientation, but the residual additions matter. The GPT-2-style block we are building follows this sequence:

```text
Block input
    │
    ├── save as the first residual shortcut
    ↓
LayerNorm → masked multi-head attention → dropout
    ↓
Add the first shortcut
    │
    ├── save as the second residual shortcut
    ↓
LayerNorm → feedforward network → dropout
    ↓
Add the second shortcut → block output
```

The second LayerNorm acts on the result **after the first residual addition** and prepares it for the feedforward network. These are two separate LayerNorm instances with their own learned scale and shift parameters. After the whole stack of transformer blocks, one more LayerNorm sits before the vocabulary output projection—the position currently occupied by `DummyLayerNorm` in our outline.

This specifies the architecture we are implementing. Other transformer designs can change the normalization type or its placement. For now, the next step is to calculate the statistics on a small example and turn the identity placeholder into a real operation.

### Inspecting a layer's outputs before normalization

The book now uses a small standalone network to make the statistics visible. This is a demonstration, not a change to GPT's embedding width. We have **two examples with five input features each**, and a linear layer produces **six output features per example**. ReLU then replaces negative values with zero; its outputs are **nonnegative**, including zero.

```python
import torch
from torch import nn

torch.manual_seed(123)
batch_example = torch.randn(2, 5)  # [examples, input features]
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)  # [2,5] -> [2,6] -> [2,6]
print(out)
```

```text
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
       grad_fn=<ReluBackward0>)
```

These are the outputs from running the book's example locally. Each row belongs to one example. `nn.Linear(5, 6)` has learned weights and, by default, a bias; ReLU has no learned parameters. Nothing has been trained here: the layer starts with random parameters, and the seed makes this run reproducible in the same environment. The printed `grad_fn` indicates that PyTorch tracked the operation for backpropagation; it does not mean a backward pass already occurred.

We now calculate statistics from **`out`, the six output activations in each row**, rather than from the five original input features:

```python
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
```

```text
Mean:
 tensor([[0.1324],
        [0.2170]], grad_fn=<MeanBackward1>)
Variance:
 tensor([[0.0231],
        [0.0398]], grad_fn=<VarBackward0>)
```

The first mean and variance describe the first row; the second pair describes the second row. Calculating these statistics does not yet normalize or change `out`.

!!! note "Why these variances differ from our earlier example"

    This snippet follows the book and uses `torch.var`'s default correction: with six values, it divides the sum of squared deviations by five. Our earlier illustration divided by the number of coordinates. The final LayerNorm implementation will also divide by the number of coordinates, using `unbiased=False` (equivalently, `correction=0`). With that setting, these rows have variances approximately 0.0192 and 0.0332. Keep the convention consistent when comparing results; the book discusses the choice later in this section.

### `dim` chooses which axis to reduce

For this two-dimensional tensor, axis 0 indexes examples (rows), and axis 1 indexes features (columns). **A reduction collapses the chosen axis** by combining its values:

| Operation on `out`, shape `[2,6]` | Which values are combined? | Result with `keepdim=True` |
| --- | --- | --- |
| `out.mean(dim=1, keepdim=True)` | The six features within each row | `[2,1]`: one mean per example |
| `out.mean(dim=-1, keepdim=True)` | The last axis, also the six features here | `[2,1]`: the same result |
| `out.mean(dim=0, keepdim=True)` | The two examples within each column | `[1,6]`: one mean per feature across examples |

<figure markdown="1">

[![A two-row, three-column example compares reduction axes. Across each row, dim 1 or minus 1 gives means 3 and 7. Down each column, dim 0 gives means 2, 5, and 8.](../assets/chapter-04/mean-reduction-axes.svg)](../assets/chapter-04/mean-reduction-axes.svg)

<figcaption markdown="span">Original companion illustration of the axis distinction in Figure 4.6, using our own small integer matrix. Both panels start with the same values. Results retain the reduced axis with `keepdim=True`; arrows show which values are averaged.</figcaption>
</figure>

For GPT's `[B,T,D]` tensors, the last axis is axis 2: **B = batch examples, T = token positions, D = vector width**. `dim=-1` continues to mean “across each token vector's coordinates,” whereas `dim=1` would now combine token positions. This is why the negative index is useful beyond this two-dimensional demonstration.

### `keepdim=True` keeps the shape convenient for subtraction

PyTorch spells the argument **`keepdim`**, singular. It retains the reduced axis with length one; it does not preserve its original length or prevent the reduction.

```text
Original out:          shape [2,6]

keepdim=False:         shape [2]       [0.1324, 0.2170]
keepdim=True:          shape [2,1]     [[0.1324],
                                      [0.2170]]
```

Both results contain the same two means. A tensor of shape `[2]` is **one-dimensional with two elements**; a tensor of shape `[2,1]` is **two-dimensional with two rows and one column**. “Two numbers” and “two axes” are different ideas.

The practical benefit appears when we subtract: `[2,6] - [2,1]` broadcasts each row's mean across that row's six features. `[2,6] - [2]` fails because broadcasting aligns dimensions from the right, and lengths six and two do not match. Keeping the axis lets the forthcoming normalization formula express exactly which mean belongs to which example.

### Applying the normalization and checking the result

Continue with `out`, `mean`, and `var` from the preceding example. The square root of variance is standard deviation:

```python
out_norm = (out - mean) / torch.sqrt(var)
print(out_norm)
print(out_norm.mean(dim=-1, keepdim=True))
print(out_norm.var(dim=-1, keepdim=True))
```

```text
tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
        [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
       grad_fn=<DivBackward0>)
tensor([[-5.9605e-08],
        [ 1.9868e-08]], grad_fn=<MeanBackward1>)
tensor([[1.0000],
        [1.0000]], grad_fn=<VarBackward0>)
```

These outputs were verified locally. Negative values are expected: a coordinate below its row's mean becomes negative when we subtract that mean, even though ReLU's original outputs were nonnegative. The variance check uses the same default correction as the variance used in the denominator.

`-5.9605e-08` means approximately **-0.000000059605**. Floating-point arithmetic has finite precision, so calculations that give exactly zero on paper can leave a tiny remainder. This is why a numerical check normally allows a small tolerance instead of requiring exact equality with zero.

PyTorch also lets us change how tensors are displayed:

```python
# Formatting only: the stored tensor values do not change.
torch.set_printoptions(precision=4, sci_mode=False)
print(out_norm.mean(dim=-1, keepdim=True))
```

```text
tensor([[-0.0000],
        [ 0.0000]], grad_fn=<MeanBackward1>)
```

The displayed `-0.0000` is the same tiny negative value rounded for printing. Turning scientific notation off does not improve numerical accuracy or make the stored mean exactly zero. The print setting also affects subsequent tensor displays in this Python session; `torch.set_printoptions(profile="default")` restores the defaults.

### Replacing the placeholder with a real `LayerNorm`

This class adapts the book's Listing 4.2 with type hints and comments. It uses **population variance**, dividing by the number of coordinates, as discussed above. That differs from the introductory calculation's default sample correction.

```python title="LayerNorm — standardization followed by learned scale and shift"
class LayerNorm(nn.Module):
    """Normalize each vector's last axis, then apply learned scale and shift.

    For floating-point input [B,T,D], B = examples, T = token positions,
    and D = embedding width. Also accepts [B,D]. The output shape is
    unchanged, and the last axis must have length emb_dim.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5  # Fixed stability constant, not a learned parameter.
        # One learned scale and one learned shift per coordinate.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [B,T,D] to [B,T,D], or [B,D] to [B,D], without mixing rows.

        B = examples, T = positions, D = coordinates. Statistics are
        recalculated from x; scale and shift are stored learned parameters.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Epsilon goes inside the square root, added to the variance.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # [D] broadcasts across every example and token position.
        return self.scale * norm_x + self.shift
```

Epsilon prevents division by zero for a constant vector and limits amplification when variance is very small. With epsilon included, standardized variance is generally slightly below one; a constant vector standardizes to zeros.

### Why learn scale and shift after normalizing?

Normalization gives the next computation a controlled starting scale. The learned transformation lets training adjust **how each coordinate should be presented to the next layer**. We preserve a predictable normalization rule while allowing a useful coordinate-wise adjustment.

For a width of four, imagine this example **after standardization**:

```text
Standardized vector:   [-1,  -1,   1,   1]
Learned scale:         [ 2,   1, 0.5,   1]
Learned shift:         [ 0,   1,   0,  -1]
                       ------------------
Output:                [-2,   0, 0.5,   0]
```

Each coordinate gets its own multiplication and addition. For example, the second coordinate becomes `1 × (-1) + 1 = 0`. These scale/shift values are illustrative, not a recorded training result. At initialization, our class uses all-one scales and all-zero shifts, so it initially returns the standardized values without this extra adjustment.

| Quantity | Where does it come from? | When does it change? |
| --- | --- | --- |
| Mean and variance | Calculated from the current vector | Every forward pass, according to the input |
| Scale and shift | Stored `nn.Parameter` vectors | Updated by the optimizer during training |
| Epsilon | A chosen small constant | Stays fixed in this implementation |

`nn.Parameter` registers scale and shift as trainable members of the module. When the module belongs to the full GPT model, they are included in `model.parameters()`. Backpropagation computes their gradients from the model's loss, and the optimizer uses those gradients to update them, just as it updates attention weights or embeddings. Calling `forward` alone does not update them, and gradient updates do not guarantee that every individual training step improves the loss.

A LayerNorm with width 768 therefore contributes **768 scales + 768 shifts = 1,536 parameters**. Its mean and variance add no stored trainable parameters. The same scale/shift pair is used at every token position and for every example in that layer; a different LayerNorm instance has its own pair.

!!! tip "The bridge to CNN normalization"

    Many batch-normalization layers used in CNNs also have learned scale and shift. The idea may already have been present in the framework's layer even if we did not implement those parameters ourselves. What differs is which values supply the normalization statistics. Our LayerNorm computes them independently for each token vector and uses the same rule during training and evaluation; it does not maintain running batch statistics.

The learned scale and shift can move the final mean away from zero and the final variance away from one. This is intentional. They also cannot generally undo normalization for every input: they are shared learned values, while each input vector has its own calculated mean and variance.

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
