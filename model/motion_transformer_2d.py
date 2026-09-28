"""Skeletal-temporal MotionJEPA encoder and predictor."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from mask.utils import gather_grid
from skeleton import SOMASkeleton30

from .modules import AxialBlock2D, PredictorAxialBlock2D, initialize_transformer
from .pos_embs import ContinuousSinCosPosEmbed1D
from .specs import MODEL_SPECS, PREDICTOR_SPECS
from .token_layout import TokenLayout


class MotionFeatureTokenizer2D(nn.Module):
    """Losslessly route ``motion_jepa_366_v1`` fields to 30 semantic joints."""

    FEATURE_DIM = 366
    NUM_JOINTS = 30
    ROOT_POSITION = slice(0, 3)
    ROOT_HEADING = slice(3, 5)
    LOCAL_POSITIONS = slice(5, 92)
    GLOBAL_ROTATIONS = slice(92, 272)
    VELOCITIES = slice(272, 362)
    FOOT_CONTACTS = slice(362, 366)

    def __init__(self, embed_dim: int):
        super().__init__()
        skeleton = SOMASkeleton30(load=False)
        self.contact_by_joint = {
            int(joint): contact
            for contact, joint in enumerate(skeleton.foot_joint_idx)
        }
        input_dims = []
        for joint in range(self.NUM_JOINTS):
            size = 14 if joint == 0 else 12
            if joint in self.contact_by_joint:
                size += 1
            input_dims.append(size)
        self.input_dims = tuple(input_dims)
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, embed_dim) for input_dim in self.input_dims]
        )

    def split_features(self, motion: torch.Tensor) -> list[torch.Tensor]:
        """Return the raw, semantically routed tensor for each joint."""
        if motion.ndim != 3 or motion.shape[-1] != self.FEATURE_DIM:
            raise ValueError(
                f"2D tokenization requires motion [B,T,{self.FEATURE_DIM}], got {tuple(motion.shape)}"
            )
        routed = []
        for joint in range(self.NUM_JOINTS):
            rotation = motion[..., 92 + 6 * joint : 92 + 6 * (joint + 1)]
            velocity = motion[..., 272 + 3 * joint : 272 + 3 * (joint + 1)]
            if joint == 0:
                fields = [
                    motion[..., self.ROOT_POSITION],
                    motion[..., self.ROOT_HEADING],
                    rotation,
                    velocity,
                ]
            else:
                local_index = joint - 1
                local_position = motion[
                    ..., 5 + 3 * local_index : 5 + 3 * (local_index + 1)
                ]
                fields = [local_position, rotation, velocity]
            contact = self.contact_by_joint.get(joint)
            if contact is not None:
                fields.append(motion[..., 362 + contact : 363 + contact])
            routed.append(torch.cat(fields, dim=-1))
        return routed

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [projection(features) for projection, features in zip(self.projections, self.split_features(motion))],
            dim=2,
        )


class _GridPositions(nn.Module):
    def __init__(self, num_frames: int, num_joints: int, embed_dim: int):
        super().__init__()
        self.register_buffer(
            "frame_index", torch.arange(num_frames, dtype=torch.float32), persistent=False
        )
        self.temporal = ContinuousSinCosPosEmbed1D(embed_dim, theta=100.0)
        self.joint = nn.Parameter(torch.zeros(1, 1, num_joints, embed_dim))
        nn.init.trunc_normal_(self.joint, std=0.02)

    def forward(self, fps: torch.Tensor) -> torch.Tensor:
        fps = torch.as_tensor(fps, device=self.frame_index.device, dtype=torch.float32)
        if fps.ndim != 1:
            raise ValueError(f"fps must have shape [B], got {tuple(fps.shape)}")
        temporal = self.temporal(
            self.frame_index.unsqueeze(0) / fps.clamp_min(1.0).unsqueeze(1)
        )
        return temporal.unsqueeze(2) + self.joint


class MotionTransformer2D(nn.Module):
    """Axial encoder over a dense frame-by-SOMA30 joint grid."""

    def __init__(
        self,
        in_chans: int = 366,
        num_frames: int = 300,
        num_joints: int = 30,
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
        if int(in_chans) != 366 or int(num_joints) != 30:
            raise ValueError("MotionTransformer2D requires motion_jepa_366_v1 on SOMA30")
        self.in_chans = 366
        self.num_frames = int(num_frames)
        self.num_joints = 30
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.token_layout = TokenLayout(
            kind="2d",
            patchified=False,
            raw_num_frames=self.num_frames,
            token_num_frames=self.num_frames,
            raw_num_joints=self.num_joints,
            token_num_joints=self.num_joints,
        )
        self.tokenizer = MotionFeatureTokenizer2D(self.embed_dim)
        self.positions = _GridPositions(self.num_frames, self.num_joints, self.embed_dim)
        drop_paths = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                AxialBlock2D(
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
        x = self.tokenizer(motion)
        position = self.positions(fps).to(device=x.device, dtype=x.dtype)
        x = x + position
        frame_active = None
        if valid_frames is not None:
            frame_active = valid_frames.to(device=x.device, dtype=torch.bool)
            if frame_active.shape != motion.shape[:2]:
                raise ValueError(
                    f"valid_frames must have shape {tuple(motion.shape[:2])}, "
                    f"got {tuple(frame_active.shape)}"
                )
        if masks is None:
            if frame_active is None:
                active = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
            else:
                active = frame_active.unsqueeze(-1).expand(-1, -1, self.num_joints)
                x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        else:
            if not masks:
                raise ValueError("At least one context mask is required")
            x = torch.cat([x] * len(masks), dim=0)
            prepared_masks = [
                mask.to(device=x.device, dtype=torch.bool) for mask in masks
            ]
            if frame_active is not None:
                valid_grid = frame_active.unsqueeze(-1)
                if any((mask & ~valid_grid).any() for mask in prepared_masks):
                    raise ValueError("Context masks select an invalid or padded frame")
            active = torch.cat(prepared_masks, dim=0)
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, active)
        return self.norm(x) * active.unsqueeze(-1).to(dtype=x.dtype)


class MotionTransformerPredictor2D(nn.Module):
    """Predict a target query grid from a separately masked context grid."""

    def __init__(
        self,
        num_frames: int = 300,
        num_joints: int = 30,
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
        if int(num_joints) != 30:
            raise ValueError("MotionTransformerPredictor2D requires SOMA30")
        self.num_frames = int(num_frames)
        self.num_joints = 30
        self.embed_dim = int(embed_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.token_layout = TokenLayout(
            kind="2d",
            patchified=False,
            raw_num_frames=self.num_frames,
            token_num_frames=self.num_frames,
            raw_num_joints=self.num_joints,
            token_num_joints=self.num_joints,
        )
        self.input_proj = nn.Linear(self.embed_dim, self.predictor_embed_dim)
        self.positions = _GridPositions(
            self.num_frames, self.num_joints, self.predictor_embed_dim
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, self.predictor_embed_dim)
        )
        self.stream_embed = nn.Parameter(
            torch.zeros(1, 2, 1, 1, self.predictor_embed_dim)
        )
        drop_paths = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                PredictorAxialBlock2D(
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
        nn.init.trunc_normal_(self.stream_embed, std=0.02)

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
        base_position = self.positions(fps).to(device=context.device, dtype=context.dtype)
        context_groups = []
        position_groups = []
        context_masks = []
        target_masks = []
        for target_mask in masks_pred:
            for enc_index, context_mask in enumerate(masks_enc):
                context_groups.append(
                    context[enc_index * batch_size : (enc_index + 1) * batch_size]
                )
                position_groups.append(base_position)
                context_masks.append(context_mask)
                target_masks.append(target_mask)
        context_grid = torch.cat(context_groups, dim=0)
        position = torch.cat(position_groups, dim=0)
        context_active = torch.cat(context_masks, dim=0).to(
            device=context.device, dtype=torch.bool
        )
        target_active = torch.cat(target_masks, dim=0).to(
            device=context.device, dtype=torch.bool
        )
        context_stream = (
            self.input_proj(context_grid) + position + self.stream_embed[:, 0]
        )
        target_stream = self.mask_token + position + self.stream_embed[:, 1]
        context_stream = context_stream * context_active.unsqueeze(-1).to(context_stream.dtype)
        target_stream = target_stream * target_active.unsqueeze(-1).to(target_stream.dtype)
        x = torch.stack([context_stream, target_stream], dim=1)
        active = torch.stack([context_active, target_active], dim=1)
        for block in self.blocks:
            x = block(x, active)
        target = self.output_proj(self.norm(x[:, 1]))
        return gather_grid(target, target_active)


def _encoder(size: str, **kwargs) -> MotionTransformer2D:
    return MotionTransformer2D(**MODEL_SPECS[size], **kwargs)


def mot_tiny_2d(**kwargs):
    return _encoder("tiny", **kwargs)


def mot_small_2d(**kwargs):
    return _encoder("small", **kwargs)


def mot_base_2d(**kwargs):
    return _encoder("base", **kwargs)


def mot_large_2d(**kwargs):
    return _encoder("large", **kwargs)


def mot_huge_2d(**kwargs):
    return _encoder("huge", **kwargs)


def mot_giant_2d(**kwargs):
    return _encoder("giant", **kwargs)


def _predictor(size: str, **kwargs) -> MotionTransformerPredictor2D:
    return MotionTransformerPredictor2D(**PREDICTOR_SPECS[size], **kwargs)


def mot_predictor_tiny_2d(**kwargs):
    return _predictor("tiny", **kwargs)


def mot_predictor_small_2d(**kwargs):
    return _predictor("small", **kwargs)


def mot_predictor_base_2d(**kwargs):
    return _predictor("base", **kwargs)


def mot_predictor_large_2d(**kwargs):
    return _predictor("large", **kwargs)


def mot_predictor_huge_2d(**kwargs):
    return _predictor("huge", **kwargs)


def mot_predictor_giant_2d(**kwargs):
    return _predictor("giant", **kwargs)


__all__ = [
    "MotionFeatureTokenizer2D",
    "MotionTransformer2D",
    "MotionTransformerPredictor2D",

    "mot_tiny_2d",
    "mot_small_2d",
    "mot_base_2d",
    "mot_large_2d",
    "mot_huge_2d",
    "mot_giant_2d",

    "mot_predictor_tiny_2d",
    "mot_predictor_small_2d",
    "mot_predictor_base_2d",
    "mot_predictor_large_2d",
    "mot_predictor_giant_2d",
    "mot_predictor_huge_2d",

]
