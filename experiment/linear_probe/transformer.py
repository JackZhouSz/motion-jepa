"""Frame-token Transformer classifier for raw motion or JEPA features."""

from __future__ import annotations

import torch
from torch import nn

from model.modules import TransformerBlock1D, initialize_transformer


class MotionTransformerClassifier(nn.Module):
    """Pre-norm temporal Transformer classified by a learnable CLS token."""

    def __init__(
        self,
        motion_dim: int | None = None,
        num_frames: int = 90,
        num_classes: int = 100,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        *,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if motion_dim is not None and input_dim is not None and motion_dim != input_dim:
            raise ValueError("motion_dim and input_dim must match when both are provided")
        self.input_dim = int(input_dim if input_dim is not None else motion_dim or 366)
        self.motion_dim = self.input_dim  # Backward-compatible attribute.
        self.num_frames = int(num_frames)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.input_projection = nn.Linear(self.input_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_frames + 1, self.embed_dim)
        )
        drop_paths = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock1D(
                    self.embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                    attn_drop=dropout,
                    drop_path=drop_paths[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        self.apply(initialize_transformer)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        motion: torch.Tensor,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.num_frames, self.input_dim)
        if motion.ndim != 3 or motion.shape[1:] != expected:
            raise ValueError(
                f"Expected input [B,{self.num_frames},{self.input_dim}], "
                f"got {tuple(motion.shape)}"
            )
        if valid_frames is None:
            active = torch.ones(
                motion.shape[:2], device=motion.device, dtype=torch.bool
            )
        else:
            active = valid_frames.to(device=motion.device, dtype=torch.bool)
            if active.shape != motion.shape[:2]:
                raise ValueError(
                    f"valid_frames must have shape {tuple(motion.shape[:2])}, "
                    f"got {tuple(active.shape)}"
                )
        x = self.input_projection(motion)
        frame_positions = self.position_embedding[:, 1:].to(
            device=x.device, dtype=x.dtype
        )
        x = x + frame_positions
        x = x * active.unsqueeze(-1).to(dtype=x.dtype)

        cls = self.cls_token.to(device=x.device, dtype=x.dtype).expand(
            motion.shape[0], -1, -1
        )
        cls = cls + self.position_embedding[:, :1].to(
            device=x.device, dtype=x.dtype
        )
        x = torch.cat((cls, x), dim=1)
        cls_active = torch.ones(
            (motion.shape[0], 1), device=motion.device, dtype=torch.bool
        )
        token_active = torch.cat((cls_active, active), dim=1)
        for block in self.blocks:
            x = block(x, token_active)
        cls_output = self.norm(x)[:, 0]
        return self.head(self.dropout(cls_output))


__all__ = ["MotionTransformerClassifier"]
