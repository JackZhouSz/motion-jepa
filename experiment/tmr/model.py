"""Trainable text and motion readouts over externally frozen token features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from model.modules import TransformerBlock1D, initialize_transformer
from model.pos_embs import ContinuousSinCosPosEmbed1D


@dataclass(frozen=True)
class AlignmentConfig:
    text_dim: int
    motion_dim: int
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("text_dim", "motion_dim", "embed_dim", "depth", "num_heads", "ff_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embed_dim % 2:
            raise ValueError("embed_dim must be even for sinusoidal positions")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _SequenceReadout(nn.Module):
    """Project token sequences and pool them with a learnable CLS token."""

    def __init__(self, input_dim: int, config: AlignmentConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.positions = ContinuousSinCosPosEmbed1D(config.embed_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock1D(
                config.embed_dim,
                config.num_heads,
                mlp_ratio=config.ff_dim / config.embed_dim,
                drop=config.dropout,
                attn_drop=config.dropout,
            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.apply(initialize_transformer)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected tokens [B,L,{self.input_dim}], got {tuple(tokens.shape)}")
        if tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("Each sequence must contain at least one valid token")
        if not tokens.is_floating_point():
            raise ValueError("Token features must have a floating-point dtype")
        active = torch.as_tensor(valid_mask, device=tokens.device, dtype=torch.bool)
        if active.shape != tokens.shape[:2]:
            raise ValueError(f"valid_mask must have shape {tuple(tokens.shape[:2])}")
        if not active.any(dim=1).all():
            raise ValueError("Each sequence must contain at least one valid token")

        # Ignore even NaN/Inf cache padding before it enters the projection.
        cleaned = tokens.masked_fill(~active.unsqueeze(-1), 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid token features must be finite")
        projected = self.input_projection(cleaned.to(self.input_projection.weight.dtype))
        cls = self.cls_token.to(projected.dtype).expand(len(tokens), -1, -1)
        x = torch.cat((cls, projected), dim=1)
        positions = self.positions(torch.arange(x.shape[1], device=x.device))
        x = x + positions.to(dtype=x.dtype).unsqueeze(0)
        active = torch.cat((torch.ones_like(active[:, :1]), active), dim=1)
        x = x.masked_fill(~active.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, active)
        return F.normalize(self.norm(x[:, 0]).float(), dim=-1)


class TextMotionAlignment(nn.Module):
    """Independent CLS Transformers aligned by a contrastive objective.

    Backbones are intentionally external: this module consumes raw motion or
    frozen JEPA motion tokens, and frozen text-backbone token sequences.
    """

    def __init__(self, config: AlignmentConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = _SequenceReadout(config.text_dim, config)
        self.motion_encoder = _SequenceReadout(config.motion_dim, config)

    def encode_text(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(tokens, valid_mask)

    def encode_motion(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(tokens, valid_mask)

    def forward(
        self,
        motion_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if motion_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError("Text and motion batch sizes must match")
        return self.encode_motion(motion_tokens, motion_mask), self.encode_text(text_tokens, text_mask)


__all__ = ["AlignmentConfig", "TextMotionAlignment"]
