"""One-head, one-block, pre-normalization character transformer."""

import math

import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, width, context_length):
        super().__init__()
        # Learned matrices: shared across all token positions and batch examples.
        # nn.Linear stores W as [out, in] and computes x @ W.T.
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        # A fixed rule, not a learned weight: True marks a future position.
        # A buffer moves with the model and is saved with its state.
        self.register_buffer(
            "future_mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), 1),
        )

    def forward(self, x):
        # x: [batch, positions, width]. Matmul keeps examples separate.
        # Each projection preserves [B, T, D]; these activations are temporary.
        q, k, v = self.query(x), self.key(x), self.value(x)
        # [B,T,D] @ [B,D,T] -> [B,T,T]: one score per query/key pair.
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        length = x.shape[1]
        scores = scores.masked_fill(self.future_mask[:length, :length], -torch.inf)
        # Softmax runs over keys. Masked scores become zero mixing amounts.
        mixing_amounts = torch.softmax(scores, dim=-1)
        # [B,T,T] @ [B,T,D] -> [B,T,D], followed by learned W_O.
        return self.output(mixing_amounts @ v)


class TransformerBlock(nn.Module):
    def __init__(self, width, hidden_width, context_length):
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

    def forward(self, x):
        # Shortcut 1 carries x unchanged; attention supplies an update.
        # LayerNorm normalizes each token across its D coordinates.
        y = x + self.attention(self.norm1(x))
        # Shortcut 2 carries y (not the original embedding) unchanged.
        return y + self.feedforward(self.norm2(y))


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary_size, context_length=32, width=32, hidden_width=64):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocabulary_size, width)
        # Unlike the hand calculation, this model includes learned position vectors.
        self.position_embedding = nn.Embedding(context_length, width)
        self.block = TransformerBlock(width, hidden_width, context_length)
        self.final_norm = nn.LayerNorm(width)
        self.vocabulary_projection = nn.Linear(width, vocabulary_size, bias=False)

    def forward(self, token_ids):
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
