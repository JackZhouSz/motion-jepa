"""Deterministic sample seeds and masked FP32 Euler integration with CFG."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from typing import Sequence

import torch


def seed_for_sample(seed: int, sample_id: str, draw_index: int = 0, stream: str = "noise") -> int:
    """Stable 63-bit seeds independent of batch boundaries and Python hashing."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    if not isinstance(draw_index, int) or isinstance(draw_index, bool) or draw_index < 0:
        raise ValueError("draw_index must be a nonnegative integer")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty string")
    payload = json.dumps([seed, sample_id, draw_index, stream], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def seeded_noise(
    sample_ids: Sequence[str],
    seed: int,
    draw_index: int,
    shape: Sequence[int],
    device: torch.device | str,
    stream: str = "noise",
) -> torch.Tensor:
    """Draw one CPU-generated FP32 noise tensor per ID, then move the batch."""
    if not sample_ids:
        raise ValueError("sample_ids must contain at least one sample")
    dimensions = tuple(shape)
    if not dimensions or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
        raise ValueError("Noise shape must contain positive integer dimensions")
    noise = []
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu").manual_seed(seed_for_sample(seed, sample_id, draw_index, stream))
        noise.append(torch.randn(dimensions, generator=generator, dtype=torch.float32))
    return torch.stack(noise).to(device=device)


def seeded_times(
    sample_ids: Sequence[str],
    seed: int,
    device: torch.device | str,
    stream: str = "validation_time",
) -> torch.Tensor:
    """Draw reproducible per-ID uniform flow times without changing global RNG."""
    if not sample_ids:
        raise ValueError("sample_ids must contain at least one sample")
    times = []
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu").manual_seed(seed_for_sample(seed, sample_id, 0, stream))
        times.append(torch.rand((), generator=generator, dtype=torch.float32))
    return torch.stack(times).to(device=device)


@torch.no_grad()
def sample_motion(
    model,
    tokens: torch.Tensor,
    fps: torch.Tensor,
    valid_frames: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    num_samples: int = 1,
    steps: int = 32,
    guidance_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """Integrate noise to normalized motion, returning [B,K,T,C].

    Initial noise accepts [B,K,T,C], or [B,T,C] when K is one. CFG branches are
    evaluated sequentially to avoid doubling the peak inference batch size.
    The caller's model train/eval mode is restored even if integration fails.
    """
    for name, value in (("num_samples", num_samples), ("steps", steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    scale = float(guidance_scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    if tokens.ndim not in (3, 4) or len(tokens) == 0 or not tokens.is_floating_point():
        raise ValueError("Tokens must be a nonempty floating-point temporal sequence or spatial grid")
    batch = len(tokens)
    frames, channels = int(model.token_layout.raw_num_frames), int(model.motion_dim)
    active = torch.as_tensor(valid_frames, device=tokens.device, dtype=torch.bool)
    if active.shape != (batch, frames) or not active.any(dim=1).all():
        raise ValueError(f"valid_frames must have shape {(batch, frames)} with a valid frame per sample")
    rates = torch.as_tensor(fps, device=tokens.device, dtype=torch.float32)
    if rates.shape != (batch,) or not torch.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("fps must contain one positive finite frame rate per sample")
    expected_shape = (batch, num_samples, frames, channels)
    if initial_noise is None:
        states = torch.randn(expected_shape, device=tokens.device, dtype=torch.float32)
    else:
        if not isinstance(initial_noise, torch.Tensor) or not initial_noise.is_floating_point():
            raise ValueError("Initial noise must be a floating-point tensor")
        states = initial_noise.detach().to(device=tokens.device, dtype=torch.float32).clone()
        if num_samples == 1 and states.shape == (batch, frames, channels):
            states = states[:, None]
        if states.shape != expected_shape:
            raise ValueError(f"Initial noise must have shape {expected_shape}")
    states = states.masked_fill(~active[:, None, :, None], 0)
    if not torch.isfinite(states).all():
        raise ValueError("Initial noise must be finite on valid frames")
    if use_bfloat16 and tokens.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 sampling is not supported by this CUDA device")

    states = states.reshape(batch * num_samples, frames, channels)
    repeated_tokens = tokens.repeat_interleave(num_samples, dim=0)
    repeated_rates = rates.repeat_interleave(num_samples, dim=0)
    repeated_active = active.repeat_interleave(num_samples, dim=0)
    repeated_mask = None
    if token_mask is not None:
        mask = torch.as_tensor(token_mask, device=tokens.device, dtype=torch.bool)
        if mask.ndim not in (2, 3) or mask.shape[0] != batch:
            raise ValueError("token_mask must be a batch-aligned temporal mask or spatial grid")
        repeated_mask = mask.repeat_interleave(num_samples, dim=0)
    conditioned = torch.zeros(batch * num_samples, device=tokens.device, dtype=torch.bool)
    unconditioned = torch.ones_like(conditioned)
    was_training = model.training
    model.eval()
    try:
        for step in range(steps):
            time = torch.full((len(states),), step / steps, device=tokens.device, dtype=torch.float32)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_bfloat16 and tokens.device.type == "cuda" else nullcontext()
            )
            with context:
                conditional_velocity = None
                null_velocity = None
                if scale != 0.0:
                    conditional_velocity = model(
                        states, time, repeated_tokens, repeated_rates, repeated_active,
                        token_mask=repeated_mask, condition_drop=conditioned,
                    )
                if scale != 1.0:
                    null_velocity = model(
                        states, time, repeated_tokens, repeated_rates, repeated_active,
                        token_mask=repeated_mask, condition_drop=unconditioned,
                    )
            if scale == 0.0:
                velocity = null_velocity.float()
            elif scale == 1.0:
                velocity = conditional_velocity.float()
            else:
                velocity = null_velocity.float() + scale * (conditional_velocity.float() - null_velocity.float())
            if velocity.shape != states.shape:
                raise ValueError("Flow velocity must have the same shape as the motion state")
            velocity = velocity.masked_fill(~repeated_active[..., None], 0)
            if not torch.isfinite(velocity).all():
                raise FloatingPointError("Sampling velocity became non-finite on valid frames")
            states = (states + velocity / steps).masked_fill(~repeated_active[..., None], 0)
            if not torch.isfinite(states).all():
                raise FloatingPointError("Sampling state became non-finite on valid frames")
    finally:
        model.train(was_training)
    return states.reshape(expected_shape)


__all__ = ["sample_motion", "seed_for_sample", "seeded_noise", "seeded_times"]
