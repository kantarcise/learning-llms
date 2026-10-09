"""One-head, one-block, pre-normalization character transformer."""

import math

import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, width, context_length):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.register_buffer(
            "future_mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), 1),
        )

    def forward(self, x):
        # x: [batch, positions, width]. Matmul keeps examples separate.
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(x.shape[-1])
        length = x.shape[1]
        scores = scores.masked_fill(self.future_mask[:length, :length], -torch.inf)
        mixing_amounts = torch.softmax(scores, dim=-1)
        return self.output(mixing_amounts @ v)


class TransformerBlock(nn.Module):
    def __init__(self, width, hidden_width, context_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalAttention(width, context_length)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, x):
        y = x + self.attention(self.norm1(x))
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
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.block(x)
        return self.vocabulary_projection(self.final_norm(x))
