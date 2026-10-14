"""A small next-character model: architecture map and reading guide.

Start reading at TinyTransformer, which assembles the smaller components.
The class order below defines the inner components first; it is not execution
order. Each class inherits nn.Module, PyTorch's base class for model components.
Calling model(token_ids) runs its forward method through PyTorch's call machinery.

Containment (our default model has ONE block and ONE attention head):
    TinyTransformer
        token_embedding       25 vocabulary rows x 32 coordinates
        position_embedding    24 position rows x 32 coordinates
        block: TransformerBlock
            norm1: LayerNorm
            attention: CausalAttention (learned Q, K, V, output projections)
            norm2: LayerNorm
            feedforward: Linear(32,64), GELU, Linear(64,32)
        final_norm: LayerNorm
        vocabulary_projection: Linear(32,25)

Execution for the inspection batch:
    IDs [2,8]
    -> token lookup [2,8,32] + position lookup [8,32]
       (the same position vectors are added to each example)
    -> x [2,8,32]
    -> y = x + attention(norm1(x))          first residual addition
    -> z = y + feedforward(norm2(y))       second residual addition
    -> final_norm(z)
    -> vocabulary_projection -> scores [2,8,25]

Shape letters: B = batch examples, T = input positions, D = vector width,
V = vocabulary entries. Defaults: D=32, V=25 for our corpus, maximum T=24;
inspection uses B=2 and T=8. Vocabulary size sets embedding table ROWS;
width sets COORDINATES PER ROW. The constructor accepts other sizes.

Parameters are stored trainable numbers. Activations are results computed from
an input. Embedding tables start randomly; LayerNorm scale/shift start at ones
and zeros. forward computes results without updating parameters. The training
loop in train.py calls backward() for gradients and optimizer.step() for updates.

Terminology and small examples: docs/preparation/glossary.md (repository root).
"""

import math

import torch
from torch import nn


class CausalAttention(nn.Module):
    """One attention head: mix each position's accessible value vectors.

    Args:
        width: Number D of coordinates in each token representation.
        context_length: Maximum sequence length, used to allocate a causal mask.

    Query/key/value/output projections are learned. The mask is fixed. Batch
    examples remain independent; only positions within an example are mixed.
    """

    future_mask: torch.Tensor

    def __init__(self, width: int, context_length: int) -> None:
        super().__init__()
        # Learned matrices: shared across all token positions and batch examples.
        # nn.Linear stores W as [out, in] and computes x @ W.T.
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        # A fixed rule, not a learned weight: True marks a future position.
        # A buffer moves with the model and is saved with its state.
        # triu(..., diagonal=1) keeps entries strictly ABOVE the diagonal:
        # column j > row i. The diagonal itself (attending to self) stays allowed.
        self.register_buffer(
            "future_mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map floating-point representations [B,T,D] to updates [B,T,D].

        T must fit the configured context. Scores have shape [B,T,T]; their
        final axis indexes keys. Future keys receive zero weight after softmax.
        """
        # Axis names: B = batch examples, T = token positions, D = vector width.
        # Our inspection uses B=2, T=8, D=32. Here x has already passed norm1.
        # Each learned projection produces a new [B,T,D] tensor. These Q/K/V
        # activations are not the stored weight matrices W_Q/W_K/W_V.
        # Worked example: follow 'o' at position 1 in the short input "job".
        # Use illustrative width-2 PROJECTED vectors (not actual embeddings):
        # q_o=[2,0]; keys for [j,o,b]=[[1,0],[0,1],[3,0]];
        # values for [j,o,b]=[[1,2],[3,0],[9,9]]. The real model uses width 32.
        q, k, v = self.query(x), self.key(x), self.value(x)

        # Every query needs a dot product with every key in the SAME example.
        # k initially stores one key per ROW: [B,T,D]. transpose(-2,-1) swaps
        # the last two axes (positions and coordinates), giving [B,D,T].
        # Thus [B,T,D] @ [B,D,T] -> [B,T,T]. In the inspection: 8x32 @ 32x8.
        # Entry [b,i,j] sums q[b,i,c]*k[b,j,c] over all D coordinates c.
        # The batch axis is untouched: examples never attend to each other.
        # Divide by sqrt(D): dot-product scale tends to grow with key width;
        # scaling helps avoid excessively sharp softmax from that growth alone.
        # D equals the full width only because this model has ONE head.
        # Example row for 'o': [2*1+0*0, 2*0+0*1, 2*3+0*0] = [2,0,6].
        # Scaling by sqrt(2) gives approximately [1.4142,0,4.2426].
        # Transposition places keys in columns; it does not change key values.
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        length = x.shape[1]  # Axis 1 is positions; -1 above means the last axis.

        # Mask row i, column j whenever j>i (the key lies in the future).
        # Slice the preallocated mask to the actual input length T.
        # -infinity becomes zero after softmax, unlike a raw score of zero.
        # Example row becomes [1.4142,0,-inf]: 'o' must predict 'b' using "jo",
        # so it cannot read the future 'b'. The target ID still goes to the loss.
        scores = scores.masked_fill(self.future_mask[:length, :length], -torch.inf)

        # dim=-1 means normalize across key positions, separately for each query.
        # Each row's mixing amounts sum to 1; future positions have amount 0.
        # Example exp scores: [4.1133,1,0]; normalized: [0.8044,0.1956,0].
        # A zero score still receives positive weight; a masked score gets zero.
        mixing_amounts = torch.softmax(scores, dim=-1)

        # [B,T,T] @ [B,T,D] -> [B,T,D]: for query i, sum all value vectors
        # weighted by that query's mixing amounts. W_O then projects the result.
        # No residual addition here: TransformerBlock.forward adds the shortcut.
        # Example: 0.8044*[1,2] + 0.1956*[3,0] + 0*[9,9] gives roughly
        # [1.3911,1.6089] using unrounded amounts. W_O transforms this vector;
        # only an illustrative identity W_O would leave it unchanged. Our W_O
        # is learned. This returns an attention update, not vocabulary logits.
        return self.output(mixing_amounts @ v)


class TransformerBlock(nn.Module):
    """Pre-normalization attention and feedforward updates with two shortcuts.

    width is the persistent representation size D; hidden_width is the temporary
    feedforward expansion size. Both residual additions preserve [B,T,D]. Each
    LayerNorm has its own learned parameters and normalizes a token across D.
    """

    def __init__(self, width: int, hidden_width: int, context_length: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalAttention(width, context_length)
        # A separate LayerNorm with its own learned scale and shift.
        self.norm2 = nn.LayerNorm(width)
        # Per token: D -> hidden_width -> D. GELU adds nonlinearity.
        # These layers share weights across positions but do not mix positions.
        self.feedforward = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return updated [B,T,D] vectors from incoming [B,T,D] vectors.

        y = x + attention(norm1(x)); output = y + feedforward(norm2(y)).
        The feedforward network processes positions independently with shared
        weights. Its output is an update to y, not a replacement for y.
        """
        # Shortcut 1 carries x unchanged; attention supplies an update.
        # LayerNorm normalizes each token across its D coordinates.
        y = x + self.attention(self.norm1(x))
        # Shortcut 2 carries y (not the original embedding) unchanged.
        return y + self.feedforward(self.norm2(y))


class TinyTransformer(nn.Module):
    """A randomly initialized next-character model, with one transformer block.

    Args:
        vocabulary_size: Number V of possible character IDs.
        context_length: Maximum T of input positions (default 24).
        width: Token representation size D (default 32).
        hidden_width: Feedforward expansion size (default 64).

    Learned token and position vectors are added before the block. Final
    normalization and a separate vocabulary projection produce raw scores.
    There is no dropout, pretrained checkpoint, or KV cache in this example.
    """

    def __init__(
        self,
        vocabulary_size: int,
        context_length: int = 24,
        width: int = 32,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocabulary_size, width)
        # Unlike the hand calculation, this model includes learned position vectors.
        self.position_embedding = nn.Embedding(context_length, width)
        self.block = TransformerBlock(width, hidden_width, context_length)
        self.final_norm = nn.LayerNorm(width)
        self.vocabulary_projection = nn.Linear(width, vocabulary_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute one vocabulary-score vector for every input position.

        Args:
            token_ids: An integer tensor [B,T]. B is the number of examples and
                T is the number of positions per example. torch.long means
                64-bit integers. Our inspection supplies [2,8]. Each ID must
                be from 0 through vocabulary_size-1 (0 through 24 here).
                Input length T must be from 1 through context_length (24 here).

        Returns:
            Floating-point logits [B,T,V], where V is vocabulary size: [2,8,25]
            for our inspection. A logit is a raw prediction score: it may be
            negative and the scores need not sum to 1. Applying softmax across
            V turns a position's scores into probabilities. No softmax or token
            selection happens in this method.

        At position 3 of "job=orde" (the '='), scores predict the next character,
        'o', using only positions 0 through 3. All positions are computed in
        parallel, with causal attention preventing access to future positions.
        The outputs and intermediate vectors are activations: temporary computed
        values, not new stored parameters. This call does not update weights.
        """
        length = token_ids.shape[1]
        if not 0 < length <= self.context_length:
            raise ValueError("Input length must fit the model's context window.")
        positions = torch.arange(length, device=token_ids.device)
        # IDs [B,T] -> embeddings [B,T,D]. Position vectors [T,D]
        # are added to every example so the model can distinguish positions.
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.block(x)
        # [B,T,D] -> [B,T,V]: raw vocabulary scores at EVERY position.
        # Training uses all positions; generation selects the last position.
        return self.vocabulary_projection(self.final_norm(x))
