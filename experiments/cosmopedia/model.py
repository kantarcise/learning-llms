"""A small causal language model, trained from random weights.

IDs [B,T] -> token + position vectors [B,T,D] -> four blocks -> final norm
-> vocabulary logits [B,T,V]. B=batch examples, T=positions, D=vector width,
V=tokenizer vocabulary size. Defaults: D=256, T<=256, four attention heads.
Each block applies norm/attention/residual, then norm/feedforward/residual.

We reuse a TOKENIZER, not a pretrained model. All learned parameters start here.
The input embedding and output vocabulary projection SHARE one weight table:
this is weight tying. It avoids storing a second large V-by-D matrix. Both input
lookup and output scoring contribute gradients to that same learned parameter.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    """Architecture choices; vocabulary size must match the tokenizer."""

    vocabulary_size: int
    context_length: int = 256
    width: int = 256
    heads: int = 4
    layers: int = 4
    hidden_width: int = 1024


class Attention(nn.Module):
    """Causal multi-head attention: heads mix positions in separate feature spaces."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Vector width must be divisible by the head count")
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map floating vectors [B,T,D] to [B,T,D] without seeing future positions.

        B=batch examples; T=positions; D=width (256); H=heads (4); K=D/H (64).
        Each token still has 256 coordinates. We arrange Q/K/V as four groups
        of 64 coordinates, rather than increasing the total representation width.
        """
        batch, length, width = x.shape
        # One Linear computes three independent projections in one operation.
        # Split [B,T,3*D] into Q, K, V, each [B,T,D].
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        # [B,T,D] -> [B,T,H,K] -> [B,H,T,K]. Each head has its own scores.
        query = query.reshape(batch, length, self.heads, self.head_width).transpose(
            1, 2
        )
        key = key.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, length, self.heads, self.head_width).transpose(
            1, 2
        )
        # PyTorch implements softmax(Q @ K.T / sqrt(K) + causal mask) @ V.
        # is_causal=True hides future positions; batch examples never mix.
        mixed = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        # Concatenate the four heads back into 256 numbers per position, then
        # learn an output projection that combines information across heads.
        mixed = mixed.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(mixed)


class Block(nn.Module):
    """Pre-normalized attention and feedforward, each with a residual shortcut."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.hidden_width),
            nn.GELU(),
            nn.Linear(config.hidden_width, config.width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preserve [B,T,D]: batch examples, token positions, vector coordinates."""
        x = x + self.attention(self.norm1(x))
        return x + self.feedforward(self.norm2(x))


class LanguageModel(nn.Module):
    """Compose embeddings, blocks, final normalization, and tied output scores."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.final_norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        """Small random weights; Linear biases start at zero, norms at defaults."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [B,T,V] for int64 IDs [B,T], with 1 <= T <= context.

        B=batch examples, T=input positions, V=vocabulary entries. Each position
        predicts the NEXT token. A forward call computes activations, not updates.
        """
        if (
            token_ids.ndim != 2
            or not 1 <= token_ids.shape[1] <= self.config.context_length
        ):
            raise ValueError("Expected [batch, positions] within the context limit")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.final_norm(self.blocks(x))
        # Same V-by-D table as input lookup: [B,T,D] @ [D,V] -> [B,T,V].
        return F.linear(x, self.token_embedding.weight)

    def configuration(self) -> dict[str, int]:
        """Return plain architecture settings for reproducible checkpoints."""
        return asdict(self.config)
