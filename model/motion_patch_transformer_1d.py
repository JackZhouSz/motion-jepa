"""Non-overlapping temporal-patch MotionJEPA encoder and predictor."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from mask.utils import apply_index_masks, repeat_mask_blocks

from .modules import TransformerBlock1D, _make_safe_attention_mask, initialize_transformer
from .pos_embs import ContinuousSinCosPosEmbed1D
from .specs import MODEL_SPECS, PREDICTOR_SPECS
from .token_layout import TokenLayout


class _TemporalPatchPositions(nn.Module):
    def __init__(
        self,
        raw_num_frames: int,
        temporal_patch_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        token_num_frames = raw_num_frames // temporal_patch_size
        centers = (
            torch.arange(token_num_frames, dtype=torch.float32) * temporal_patch_size
            + (temporal_patch_size - 1) / 2.0
        )
        self.register_buffer("frame_center", centers, persistent=False)
        self.embedding = ContinuousSinCosPosEmbed1D(embed_dim, theta=100.0)

    def forward(self, fps: torch.Tensor) -> torch.Tensor:
        fps = torch.as_tensor(fps, device=self.frame_center.device, dtype=torch.float32)
        if fps.ndim != 1:
            raise ValueError(f"fps must have shape [B], got {tuple(fps.shape)}")
        return self.embedding(
            self.frame_center.unsqueeze(0) / fps.clamp_min(1.0).unsqueeze(1)
        )


class MotionPatchTransformer1D(nn.Module):
    """Encode non-overlapping temporal patches as transformer tokens."""

    def __init__(
        self,
        in_chans: int,
        num_frames: int = 300,
        temporal_patch_size: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1.0e-6),
    ) -> None:
        super().__init__()
        self.in_chans = int(in_chans)
        self.num_frames = int(num_frames)
        self.temporal_patch_size = int(temporal_patch_size)
        if self.temporal_patch_size <= 0 or self.num_frames < self.temporal_patch_size:
            raise ValueError("temporal_patch_size must be in [1, num_frames]")
        self.token_num_frames = self.num_frames // self.temporal_patch_size
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.token_layout = TokenLayout(
            kind="1d",
            patchified=True,
            raw_num_frames=self.num_frames,
            token_num_frames=self.token_num_frames,
            temporal_patch_size=self.temporal_patch_size,
        )
        self.patch_embed = nn.Conv1d(
            self.in_chans,
            self.embed_dim,
            kernel_size=self.temporal_patch_size,
            stride=self.temporal_patch_size,
        )
        self.positions = _TemporalPatchPositions(
            self.num_frames, self.temporal_patch_size, self.embed_dim
        )
        drop_paths = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock1D(
                    self.embed_dim,
                    self.num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_paths[index],
                    norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)
        self.apply(initialize_transformer)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def forward(
        self,
        motion: torch.Tensor,
        fps: torch.Tensor,
        masks: list[torch.Tensor] | None = None,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[1:] != (self.num_frames, self.in_chans):
            raise ValueError(
                f"Expected motion [B,{self.num_frames},{self.in_chans}], got {tuple(motion.shape)}"
            )
        x = self.patch_embed(motion.transpose(1, 2)).transpose(1, 2)
        x = x + self.positions(fps).to(device=x.device, dtype=x.dtype)
        active = None
        if valid_frames is not None:
            active = self.token_layout.valid_token_mask(
                valid_frames.to(device=x.device, dtype=torch.bool)
            )
        if masks is not None:
            if not masks:
                raise ValueError("At least one context mask is required")
            if active is not None:
                for mask in masks:
                    index = mask.to(device=x.device, dtype=torch.long)
                    if (
                        index.shape[0] != len(motion)
                        or (index < 0).any()
                        or (index >= self.token_num_frames).any()
                        or not torch.gather(active, 1, index).all()
                    ):
                        raise ValueError("Context masks select an invalid or padded patch")
            x = apply_index_masks(x, masks)
            active = None
        elif active is not None:
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        attention_active = _make_safe_attention_mask(active) if active is not None else None
        for block in self.blocks:
            x = block(x, attention_active)
        x = self.norm(x)
        if active is not None:
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        return x


class MotionPatchTransformerPredictor1D(nn.Module):
    """Predict target temporal-patch embeddings from context patches."""

    def __init__(
        self,
        num_frames: int = 300,
        temporal_patch_size: int = 3,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1.0e-6),
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.temporal_patch_size = int(temporal_patch_size)
        if self.temporal_patch_size <= 0 or self.num_frames < self.temporal_patch_size:
            raise ValueError("temporal_patch_size must be in [1, num_frames]")
        self.token_num_frames = self.num_frames // self.temporal_patch_size
        self.embed_dim = int(embed_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.token_layout = TokenLayout(
            kind="1d",
            patchified=True,
            raw_num_frames=self.num_frames,
            token_num_frames=self.token_num_frames,
            temporal_patch_size=self.temporal_patch_size,
        )
        self.input_proj = nn.Linear(self.embed_dim, self.predictor_embed_dim)
        self.positions = _TemporalPatchPositions(
            self.num_frames, self.temporal_patch_size, self.predictor_embed_dim
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.predictor_embed_dim))
        drop_paths = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock1D(
                    self.predictor_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_paths[index],
                    norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(self.predictor_embed_dim)
        self.output_proj = nn.Linear(self.predictor_embed_dim, self.embed_dim)
        self.apply(initialize_transformer)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context: torch.Tensor,
        fps: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> torch.Tensor:
        if not masks_enc or not masks_pred:
            raise ValueError("Predictor requires encoder and target masks")
        batch_size = len(fps)
        if len(context) != batch_size * len(masks_enc):
            raise ValueError("Context batch does not match encoder mask count")
        x = self.input_proj(context)
        position = self.positions(fps).to(device=x.device, dtype=x.dtype)
        x = x + apply_index_masks(position, masks_enc)
        context_tokens = context.shape[1]
        target_position = apply_index_masks(position, masks_pred)
        target_position = repeat_mask_blocks(
            target_position, batch_size, len(masks_enc)
        )
        target = self.mask_token.to(dtype=x.dtype) + target_position
        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, target], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, context_tokens:])
        return self.output_proj(x)


def _encoder(size: str, **kwargs) -> MotionPatchTransformer1D:
    return MotionPatchTransformer1D(**MODEL_SPECS[size], **kwargs)


def _predictor(size: str, **kwargs) -> MotionPatchTransformerPredictor1D:
    return MotionPatchTransformerPredictor1D(**PREDICTOR_SPECS[size], **kwargs)


def mot_patch_tiny_1d(**kwargs):
    return _encoder("tiny", **kwargs)


def mot_patch_small_1d(**kwargs):
    return _encoder("small", **kwargs)


def mot_patch_base_1d(**kwargs):
    return _encoder("base", **kwargs)


def mot_patch_large_1d(**kwargs):
    return _encoder("large", **kwargs)


def mot_patch_huge_1d(**kwargs):
    return _encoder("huge", **kwargs)


def mot_patch_giant_1d(**kwargs):
    return _encoder("giant", **kwargs)


def mot_predictor_patch_tiny_1d(**kwargs):
    return _predictor("tiny", **kwargs)


def mot_predictor_patch_small_1d(**kwargs):
    return _predictor("small", **kwargs)


def mot_predictor_patch_base_1d(**kwargs):
    return _predictor("base", **kwargs)


def mot_predictor_patch_large_1d(**kwargs):
    return _predictor("large", **kwargs)


def mot_predictor_patch_huge_1d(**kwargs):
    return _predictor("huge", **kwargs)


def mot_predictor_patch_giant_1d(**kwargs):
    return _predictor("giant", **kwargs)


__all__ = [
    "MotionPatchTransformer1D",
    "MotionPatchTransformerPredictor1D",
    "mot_patch_tiny_1d",
    "mot_patch_small_1d",
    "mot_patch_base_1d",
    "mot_patch_large_1d",
    "mot_patch_huge_1d",
    "mot_patch_giant_1d",
    "mot_predictor_patch_tiny_1d",
    "mot_predictor_patch_small_1d",
    "mot_predictor_patch_base_1d",
    "mot_predictor_patch_large_1d",
    "mot_predictor_patch_huge_1d",
    "mot_predictor_patch_giant_1d",
]
