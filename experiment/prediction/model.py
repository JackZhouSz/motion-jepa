"""Deterministic frame-query decoding of frozen MotionJEPA tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.modules import initialize_transformer
from model.pos_embs import ContinuousSinCosPosEmbed1D
from model.token_layout import TokenLayout


@dataclass(frozen=True)
class DecoderConfig:
    hidden_dim: int = 384
    depth: int = 4
    num_heads: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "depth", "num_heads", "ffn_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even for sinusoidal positions")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class MotionDecoder(nn.Module):
    """Decode full raw-frame sequences without pooling JEPA spatial tokens.

    Tokens must already use the functional LayerNorm applied to JEPA teacher
    targets. Output coordinates are the normalized motion features used by the
    encoder. Incomplete final temporal patches remain absent from the memory;
    their valid raw frames are still queried and reconstructed.
    """

    def __init__(
        self,
        feature_dim: int,
        token_layout: TokenLayout,
        motion_dim: int = 366,
        config: DecoderConfig | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("motion_dim", motion_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.token_layout = token_layout
        self.config = config or DecoderConfig()
        hidden = self.config.hidden_dim
        self.input_projection = nn.Linear(feature_dim, hidden)
        self.positions = ContinuousSinCosPosEmbed1D(hidden, theta=100.0)
        self.frame_query = nn.Parameter(torch.zeros(1, 1, hidden))
        spatial_count = int(token_layout.token_num_joints or 1)
        self.spatial_position = (
            nn.Parameter(torch.zeros(1, 1, spatial_count, hidden))
            if token_layout.kind == "2d" else None
        )
        self.register_buffer(
            "frame_indices",
            torch.arange(token_layout.raw_num_frames, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "patch_centers",
            torch.arange(token_layout.token_num_frames, dtype=torch.float32)
            * token_layout.temporal_patch_size
            + (token_layout.temporal_patch_size - 1) / 2.0,
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=hidden,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.ffn_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                layer_norm_eps=1.0e-6,
            )
            for _ in range(self.config.depth)
        )
        self.norm = nn.LayerNorm(hidden, eps=1.0e-6)
        self.output_projection = nn.Linear(hidden, motion_dim)
        self.apply(initialize_transformer)
        nn.init.trunc_normal_(self.frame_query, std=0.02)
        if self.spatial_position is not None:
            nn.init.trunc_normal_(self.spatial_position, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        fps: torch.Tensor,
        valid_frames: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self.token_layout
        expected_tail = (layout.token_num_frames, self.feature_dim)
        if layout.kind == "2d":
            expected_tail = (
                layout.token_num_frames,
                int(layout.token_num_joints),
                self.feature_dim,
            )
        if tokens.ndim != len(expected_tail) + 1 or tuple(tokens.shape[1:]) != expected_tail:
            raise ValueError(f"Expected tokens [B,{','.join(map(str, expected_tail))}], got {tuple(tokens.shape)}")
        if not tokens.is_floating_point() or len(tokens) == 0:
            raise ValueError("Token features must be a nonempty floating-point batch")
        active_frames = torch.as_tensor(valid_frames, device=tokens.device, dtype=torch.bool)
        expected_frames = (len(tokens), layout.raw_num_frames)
        if tuple(active_frames.shape) != expected_frames:
            raise ValueError(f"valid_frames must have shape {expected_frames}")
        rates = torch.as_tensor(fps, device=tokens.device, dtype=torch.float32)
        if rates.shape != (len(tokens),) or not torch.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("fps must contain one positive finite frame rate per sample")

        memory_active = layout.valid_token_mask(active_frames)
        if layout.kind == "2d":
            memory_active = memory_active.unsqueeze(-1).expand(-1, -1, int(layout.token_num_joints))
        if token_mask is not None:
            supplied_mask = torch.as_tensor(token_mask, device=tokens.device, dtype=torch.bool)
            if layout.kind == "2d" and supplied_mask.shape == tokens.shape[:2]:
                supplied_mask = supplied_mask.unsqueeze(-1).expand_as(memory_active)
            if supplied_mask.shape != memory_active.shape:
                raise ValueError(f"token_mask must match temporal tokens or the token grid {tuple(memory_active.shape)}")
            memory_active = memory_active & supplied_mask
        if not memory_active.flatten(1).any(dim=1).all():
            raise ValueError("Each sample must have at least one valid JEPA memory token")

        cleaned = tokens.masked_fill(~memory_active.unsqueeze(-1), 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid token features must be finite")
        memory = self.input_projection(cleaned.to(self.input_projection.weight.dtype))
        memory_position = self.positions(self.patch_centers[None] / rates[:, None]).to(memory.dtype)
        if layout.kind == "2d":
            memory = memory + memory_position.unsqueeze(2) + self.spatial_position.to(memory.dtype)
        else:
            memory = memory + memory_position
        memory = memory.masked_fill(~memory_active.unsqueeze(-1), 0)
        memory = memory.reshape(len(tokens), -1, self.config.hidden_dim)
        memory_padding = ~memory_active.reshape(len(tokens), -1)

        query = self.frame_query.to(memory.dtype) + self.positions(
            self.frame_indices[None] / rates[:, None]
        ).to(memory.dtype)
        query = query.masked_fill(~active_frames.unsqueeze(-1), 0)
        for block in self.blocks:
            query = block(
                query,
                memory,
                tgt_key_padding_mask=~active_frames,
                memory_key_padding_mask=memory_padding,
            )
            query = query.masked_fill(~active_frames.unsqueeze(-1), 0)
        output = self.output_projection(self.norm(query))
        return output.masked_fill(~active_frames.unsqueeze(-1), 0)


__all__ = ["DecoderConfig", "MotionDecoder"]
