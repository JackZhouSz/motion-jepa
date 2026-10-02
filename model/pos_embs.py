"""Absolute and rotary temporal position embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def normalize_position_encoding(
    position_encoding: str = "absolute",
    rope_theta: float = 100.0,
    rope_time_scale: float = 1.0,
) -> dict:
    """Canonical positional settings, including parameter-free model semantics."""
    if position_encoding not in {"absolute", "rope"}:
        raise ValueError("position_encoding must be 'absolute' or 'rope'")
    if position_encoding == "absolute":
        return {"temporal": "absolute"}
    theta, time_scale = float(rope_theta), float(rope_time_scale)
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("rope_theta must be finite and positive")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("rope_time_scale must be finite and positive")
    return {"temporal": "rope", "rope_theta": theta, "rope_time_scale": time_scale}


def temporal_token_positions(
    num_frames: int,
    fps: torch.Tensor,
    temporal_patch_size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return original frame/patch-center times in seconds as FP32 ``[B,T]``.

    Masking must gather these coordinates with the same indices as the tokens;
    compact sequence offsets do not represent elapsed motion time.
    """
    if temporal_patch_size <= 0 or num_frames < temporal_patch_size:
        raise ValueError("temporal_patch_size must be in [1, num_frames]")
    rates = torch.as_tensor(fps, device=device, dtype=torch.float32)
    if rates.ndim != 1:
        raise ValueError(f"fps must have shape [B], got {tuple(rates.shape)}")
    centers = (
        torch.arange(num_frames // temporal_patch_size, device=rates.device, dtype=torch.float32)
        * temporal_patch_size + (temporal_patch_size - 1) / 2.0
    )
    return centers.unsqueeze(0) / rates.clamp_min(1.0).unsqueeze(1)


class RotaryPosEmbed1D(nn.Module):
    """Full-head RoPE factors for explicit continuous positions.

    Factors are computed once per model forward and shared by its blocks. The
    frequency calculation stays in FP32 even when the model is cast to BF16/FP16.
    ``time_scale`` is a fixed multiplier of physical seconds, not a per-sample FPS.
    """

    def __init__(self, head_dim: int, theta: float = 100.0, time_scale: float = 1.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError(f"RoPE head_dim must be positive and even, got {head_dim}")
        settings = normalize_position_encoding("rope", theta, time_scale)
        self.head_dim = int(head_dim)
        self.theta = settings["rope_theta"]
        self.time_scale = settings["rope_time_scale"]

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ``[B,T]`` timestamps to FP32 ``[B,1,T,D/2]`` cosine/sine."""
        if positions.ndim != 2:
            raise ValueError(f"RoPE positions must have shape [B,T], got {tuple(positions.shape)}")
        frequency = self.theta ** (
            -torch.arange(0, self.head_dim, 2, device=positions.device, dtype=torch.float32)
            / self.head_dim
        )
        angles = (positions.float() * self.time_scale)[:, None, :, None] * frequency
        return angles.cos(), angles.sin()


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate projected ``[B,H,T,D]`` queries/keys, pairing adjacent channels."""
    cosine, sine = rotary
    expected = (query.shape[0], 1, query.shape[2], query.shape[3] // 2) if query.ndim == 4 else None
    if (
        query.ndim != 4 or key.shape != query.shape or query.shape[-1] % 2
        or cosine.shape != expected or sine.shape != expected
    ):
        raise ValueError("RoPE requires matching [B,H,T,D] Q/K and [B,1,T,D/2] factors")
    cosine = cosine.to(device=query.device, dtype=query.dtype)
    sine = sine.to(device=query.device, dtype=query.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    return rotate(query), rotate(key)


class ContinuousSinCosPosEmbed1D(nn.Module):
    """
    Continuous 1D sin/cos positional embedding as an nn.Module.
    Input positions can be float tensors, e.g. [N] or [B, N].
    """
    def __init__(self, embed_dim, theta=10000.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.embed_dim = embed_dim

        half_dim = embed_dim // 2
        omega = torch.arange(half_dim, dtype=torch.float32)
        omega = 1.0 / (theta ** (omega / float(half_dim)))  # [D/2]
        self.register_buffer("omega", omega, persistent=False)


    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: float tensor with shape (...,)
        returns: tensor with shape (..., embed_dim)
        """
        pos = positions.to(dtype=self.omega.dtype)
        out = pos.unsqueeze(-1) * self.omega  # (..., D/2)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)  # (..., D)
        return emb
