"""Temporal convolutional classifier for raw motion or JEPA frame tokens."""

from __future__ import annotations

import torch
from torch import nn


def _valid_mask(
    batch: int,
    frames: int,
    device: torch.device,
    valid_frames: torch.Tensor | None,
) -> torch.Tensor:
    if valid_frames is None:
        return torch.ones(batch, frames, device=device, dtype=torch.bool)
    active = valid_frames.to(device=device, dtype=torch.bool)
    if active.shape != (batch, frames):
        raise ValueError(
            f"valid_frames must have shape {(batch, frames)}, got {tuple(active.shape)}"
        )
    return active


class TemporalResidualBlock(nn.Module):
    """Two temporal convolutions with an optional strided projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.stride = int(stride)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.projection = (
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                ),
                nn.GroupNorm(32, out_channels),
            )
            if self.stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.projection(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.activation(x + residual)
        if self.stride > 1:
            active = active[:, :: self.stride]
        x = x * active.unsqueeze(1).to(dtype=x.dtype)
        return x, active


class MotionCNNClassifier(nn.Module):
    """Group-normalized temporal ResNet over generic `[B,T,D]` input."""

    def __init__(
        self,
        motion_dim: int | None = None,
        num_classes: int = 100,
        widths: tuple[int, int, int] = (128, 192, 256),
        blocks_per_stage: int = 1,
        dropout: float = 0.1,
        *,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if motion_dim is not None and input_dim is not None and motion_dim != input_dim:
            raise ValueError("motion_dim and input_dim must match when both are provided")
        if blocks_per_stage <= 0:
            raise ValueError("blocks_per_stage must be positive")
        self.input_dim = int(input_dim if input_dim is not None else motion_dim or 366)
        self.motion_dim = self.input_dim  # Backward-compatible attribute.
        self.num_classes = int(num_classes)
        self.widths = tuple(int(width) for width in widths)
        stem_width = self.widths[0]
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                stem_width,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(32, stem_width),
            nn.GELU(),
        )
        blocks = []
        in_channels = stem_width
        for stage_index, out_channels in enumerate(self.widths):
            for block_index in range(blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(TemporalResidualBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.widths[-1], self.num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        motion: torch.Tensor,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input [B,T,{self.input_dim}], got {tuple(motion.shape)}"
            )
        active = _valid_mask(
            len(motion), motion.shape[1], motion.device, valid_frames
        )
        x = motion * active.unsqueeze(-1).to(dtype=motion.dtype)
        x = self.stem(x.transpose(1, 2))
        x = x * active.unsqueeze(1).to(dtype=x.dtype)
        for block in self.blocks:
            x, active = block(x, active)
        weights = active.unsqueeze(1).to(dtype=x.dtype)
        pooled = (x * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return self.head(self.dropout(pooled))


__all__ = ["MotionCNNClassifier", "TemporalResidualBlock"]
