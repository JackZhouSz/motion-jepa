"""Raw-motion rectified-flow velocity conditioned on frozen JEPA tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.modules import initialize_transformer
from model.pos_embs import ContinuousSinCosPosEmbed1D
from model.token_layout import TokenLayout


@dataclass(frozen=True)
class FlowConfig:
    hidden_dim: int = 384
    depth: int = 4
    num_heads: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.0

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


class MotionFlow(nn.Module):
    """Predict velocity in normalized raw-motion space with optional null CFG.

    JEPA tokens already have the functional LayerNorm used for teacher targets.
    ``time`` is flow time in [0,1]; frame and patch positions separately use
    seconds derived from ``fps``. Condition dropout removes a whole sample's
    JEPA memory and leaves one learned null token available for attention.
    """

    def __init__(
        self,
        feature_dim: int,
        token_layout: TokenLayout,
        motion_dim: int = 366,
        config: FlowConfig | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("motion_dim", motion_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.token_layout = token_layout
        self.config = config or FlowConfig()
        hidden = self.config.hidden_dim
        self.feature_projection = nn.Linear(feature_dim, hidden)
        self.motion_projection = nn.Linear(motion_dim, hidden)
        self.physical_positions = ContinuousSinCosPosEmbed1D(hidden, theta=100.0)
        self.flow_positions = ContinuousSinCosPosEmbed1D(hidden, theta=10000.0)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden)
        )
        self.null_memory = nn.Parameter(torch.zeros(1, 1, hidden))
        self.spatial_position = (
            nn.Parameter(torch.zeros(1, 1, int(token_layout.token_num_joints), hidden))
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
        nn.init.trunc_normal_(self.null_memory, std=0.02)
        if self.spatial_position is not None:
            nn.init.trunc_normal_(self.spatial_position, std=0.02)

    def forward(
        self,
        noisy_motion: torch.Tensor,
        time: torch.Tensor,
        tokens: torch.Tensor,
        fps: torch.Tensor,
        valid_frames: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        condition_drop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self.token_layout
        expected_tokens = (layout.token_num_frames, self.feature_dim)
        if layout.kind == "2d":
            expected_tokens = (
                layout.token_num_frames, int(layout.token_num_joints), self.feature_dim
            )
        if tokens.ndim != len(expected_tokens) + 1 or tuple(tokens.shape[1:]) != expected_tokens:
            raise ValueError(f"Expected tokens [B,{','.join(map(str, expected_tokens))}], got {tuple(tokens.shape)}")
        if not tokens.is_floating_point() or len(tokens) == 0:
            raise ValueError("JEPA tokens must be a nonempty floating-point batch")
        batch = len(tokens)
        expected_motion = (batch, layout.raw_num_frames, self.motion_dim)
        if tuple(noisy_motion.shape) != expected_motion or not noisy_motion.is_floating_point():
            raise ValueError(f"Expected floating-point noisy_motion {expected_motion}")
        if noisy_motion.device != tokens.device:
            raise ValueError("Noisy motion and JEPA tokens must be on the same device")
        active_frames = torch.as_tensor(valid_frames, device=tokens.device, dtype=torch.bool)
        if active_frames.shape != noisy_motion.shape[:2]:
            raise ValueError(f"valid_frames must have shape {tuple(noisy_motion.shape[:2])}")
        if not active_frames.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid raw frame")
        rates = torch.as_tensor(fps, device=tokens.device, dtype=torch.float32)
        if rates.shape != (batch,) or not torch.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("fps must contain one positive finite frame rate per sample")
        flow_time = torch.as_tensor(time, device=tokens.device, dtype=torch.float32)
        if (flow_time.shape != (batch,) or not torch.isfinite(flow_time).all()
                or ((flow_time < 0) | (flow_time > 1)).any()):
            raise ValueError("time must contain one finite flow time in [0,1] per sample")
        dropped = (
            torch.zeros(batch, device=tokens.device, dtype=torch.bool)
            if condition_drop is None
            else torch.as_tensor(condition_drop, device=tokens.device, dtype=torch.bool)
        )
        if dropped.shape != (batch,):
            raise ValueError("condition_drop must have shape [B]")

        memory_active = layout.valid_token_mask(active_frames)
        if layout.kind == "2d":
            memory_active = memory_active.unsqueeze(-1).expand(-1, -1, int(layout.token_num_joints))
        if token_mask is not None:
            supplied = torch.as_tensor(token_mask, device=tokens.device, dtype=torch.bool)
            if layout.kind == "2d" and supplied.shape == tokens.shape[:2]:
                supplied = supplied.unsqueeze(-1).expand_as(memory_active)
            if supplied.shape != memory_active.shape:
                raise ValueError(f"token_mask must match temporal tokens or the token grid {tuple(memory_active.shape)}")
            memory_active = memory_active & supplied
        memory_active = memory_active & ~dropped.reshape(batch, *([1] * (memory_active.ndim - 1)))
        if (~memory_active.flatten(1).any(dim=1) & ~dropped).any():
            raise ValueError("Each conditioned sample must have at least one valid JEPA memory token")

        cleaned_tokens = tokens.masked_fill(~memory_active.unsqueeze(-1), 0)
        cleaned_motion = noisy_motion.masked_fill(~active_frames.unsqueeze(-1), 0)
        if not torch.isfinite(cleaned_tokens).all() or not torch.isfinite(cleaned_motion).all():
            raise ValueError("Valid noisy motion and active JEPA tokens must be finite")
        memory = self.feature_projection(cleaned_tokens.to(self.feature_projection.weight.dtype))
        temporal = self.physical_positions(self.patch_centers[None] / rates[:, None]).to(memory.dtype)
        if layout.kind == "2d":
            memory = memory + temporal.unsqueeze(2) + self.spatial_position.to(memory.dtype)
        else:
            memory = memory + temporal
        memory = memory.masked_fill(~memory_active.unsqueeze(-1), 0)
        memory = memory.reshape(batch, -1, self.config.hidden_dim)
        memory = torch.cat((memory, self.null_memory.to(memory.dtype).expand(batch, -1, -1)), dim=1)
        memory_active = torch.cat((memory_active.reshape(batch, -1), dropped[:, None]), dim=1)

        query = self.motion_projection(cleaned_motion.to(self.motion_projection.weight.dtype))
        query = query + self.physical_positions(self.frame_indices[None] / rates[:, None]).to(query.dtype)
        embedded_time = self.flow_positions(flow_time * 1000.0)
        query = query + self.time_mlp(embedded_time.to(self.time_mlp[0].weight.dtype)).to(query.dtype)[:, None]
        query = query.masked_fill(~active_frames.unsqueeze(-1), 0)
        for block in self.blocks:
            query = block(
                query, memory,
                tgt_key_padding_mask=~active_frames,
                memory_key_padding_mask=~memory_active,
            )
            query = query.masked_fill(~active_frames.unsqueeze(-1), 0)
        velocity = self.output_projection(self.norm(query))
        return velocity.masked_fill(~active_frames.unsqueeze(-1), 0)

    def sample(
        self,
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
        from .sampling import sample_motion
        return sample_motion(
            self, tokens, fps, valid_frames, token_mask=token_mask,
            num_samples=num_samples, steps=steps, guidance_scale=guidance_scale,
            initial_noise=initial_noise, use_bfloat16=use_bfloat16,
        )


__all__ = ["FlowConfig", "MotionFlow"]
