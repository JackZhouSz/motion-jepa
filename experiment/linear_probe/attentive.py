"""A single-query attentive classifier for frozen temporal JEPA tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AttentiveProbe(nn.Module):
    """Masked cross-attention followed by a residual MLP and classification."""

    def __init__(self, input_dim: int, num_frames: int, num_classes: int,
                 num_heads: int = 6, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if min(input_dim, num_frames, num_classes, num_heads) <= 0 or input_dim % num_heads:
            raise ValueError("Positive dimensions and input_dim divisible by num_heads required")
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.empty(1, 1, input_dim))
        self.token_norm = nn.LayerNorm(input_dim)
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.kv_proj = nn.Linear(input_dim, 2 * input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.mlp_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, int(input_dim * mlp_ratio)),
                                 nn.GELU(), nn.Linear(int(input_dim * mlp_ratio), input_dim))
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor,
                valid_frames: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.num_frames, self.input_dim):
            raise ValueError(f"Expected [B,{self.num_frames},{self.input_dim}] tokens")
        active = (torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
                  if valid_frames is None else valid_frames.to(tokens.device, dtype=torch.bool))
        if active.shape != tokens.shape[:2] or not active.any(dim=1).all():
            raise ValueError("Each sample must have at least one valid token and a matching mask")
        # Zero padding before normalization/projection, including arbitrary NaN padding.
        tokens = self.token_norm(tokens.masked_fill(~active.unsqueeze(-1), 0))
        batch, length, dim = tokens.shape
        query = self.query.expand(batch, -1, -1)
        q = self.q_proj(query).reshape(batch, 1, self.num_heads, dim // self.num_heads).transpose(1, 2)
        kv = self.kv_proj(tokens).reshape(batch, length, 2, self.num_heads, dim // self.num_heads)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=active[:, None, None, :])
        x = query + self.out_proj(attended.transpose(1, 2).reshape(batch, 1, dim))
        x = x + self.mlp(self.mlp_norm(x))
        return self.head(self.norm(x[:, 0]))
