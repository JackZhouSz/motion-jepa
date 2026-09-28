"""Temporal and anatomical patchification for skeletal MotionJEPA."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mask.utils import gather_grid
from skeleton import SOMASkeleton30

from .modules import (
    AxialBlock2D,
    PredictorAxialBlock2D,
    initialize_transformer,
    prepare_packed_axial_layout,
)
from .motion_transformer_2d import MotionFeatureTokenizer2D
from .pos_embs import ContinuousSinCosPosEmbed1D
from .specs import MODEL_SPECS, PREDICTOR_SPECS
from .token_layout import TokenLayout


SPATIAL_POOLING = "graph_mean"
SPATIAL_GRAPH_VERSION = 1
TRAJECTORY_TOKEN_INDEX = 0
BODY_TOKEN_OFFSET = 1
TRAJECTORY_FIELDS = ("root_x", "root_z", "heading_cos", "heading_sin")
_PACKED_ENCODER_ATTENTION = True
_PACKED_PREDICTOR_ATTENTION = True

SPATIAL_GROUPINGS: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {
    # Preserve every SOMA joint as its own spatial token. This isolates
    # temporal patchification from lossy anatomical pooling.
    "joint30": tuple(
        (f"joint_{joint:02d}", (joint,))
        for joint in range(MotionFeatureTokenizer2D.NUM_JOINTS)
    ),
    "fine11": (
        ("pelvis", (0,)),
        ("torso", (1, 2, 3)),
        ("head", (4, 5, 6, 7, 8, 9)),
        ("left_upper_arm", (10, 11)),
        ("left_lower_arm_hand", (12, 13, 14, 15)),
        ("right_upper_arm", (16, 17)),
        ("right_lower_arm_hand", (18, 19, 20, 21)),
        ("left_leg", (22, 23)),
        ("left_foot", (24, 25)),
        ("right_leg", (26, 27)),
        ("right_foot", (28, 29)),
    ),
    "coarse7": (
        ("pelvis", (0,)),
        ("torso", (1, 2, 3)),
        ("head", (4, 5, 6, 7, 8, 9)),
        ("left_arm_hand", (10, 11, 12, 13, 14, 15)),
        ("right_arm_hand", (16, 17, 18, 19, 20, 21)),
        ("left_leg_foot", (22, 23, 24, 25)),
        ("right_leg_foot", (26, 27, 28, 29)),
    ),
}


def get_spatial_grouping(name: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return and validate a fixed SOMA30 anatomical grouping."""
    try:
        groups = SPATIAL_GROUPINGS[str(name)]
    except KeyError as error:
        choices = ", ".join(sorted(SPATIAL_GROUPINGS))
        raise ValueError(f"Unknown spatial_grouping {name!r}; choose one of: {choices}") from error
    flattened = [joint for _, joints in groups for joint in joints]
    if sorted(flattened) != list(range(MotionFeatureTokenizer2D.NUM_JOINTS)):
        raise ValueError(f"Spatial grouping {name!r} must assign every SOMA30 joint once")
    return groups


def _normalized_group_adjacency(
    groups: tuple[tuple[str, tuple[int, ...]], ...],
) -> torch.Tensor:
    skeleton = SOMASkeleton30(load=False)
    adjacency = torch.eye(skeleton.nbjoints, dtype=torch.float32)
    group_by_joint = {
        joint: group_index
        for group_index, (_, joints) in enumerate(groups)
        for joint in joints
    }
    for child, parent in enumerate(skeleton.parents.tolist()):
        if parent >= 0 and group_by_joint[child] == group_by_joint[parent]:
            adjacency[child, parent] = 1.0
            adjacency[parent, child] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def spatial_patch_signature(name: str) -> dict:
    groups = get_spatial_grouping(name)
    skeleton = SOMASkeleton30(load=False)
    adjacency = _normalized_group_adjacency(groups)
    edges = [
        [row, column]
        for row in range(len(adjacency))
        for column in range(row + 1, len(adjacency))
        if float(adjacency[row, column]) != 0.0
    ]
    return {
        "spatial_grouping": str(name),
        "spatial_pooling": SPATIAL_POOLING,
        "spatial_graph_version": SPATIAL_GRAPH_VERSION,
        "group_names": [group_name for group_name, _ in groups],
        "spatial_token_names": [
            "trajectory", *[group_name for group_name, _ in groups]
        ],
        "trajectory_token_index": TRAJECTORY_TOKEN_INDEX,
        "body_token_offset": BODY_TOKEN_OFFSET,
        "trajectory_fields": list(TRAJECTORY_FIELDS),
        "joint_groups": [list(joints) for _, joints in groups],
        "joint_names": list(skeleton.names),
        "graph_edges": edges,
    }


def _supports_persistent_packing(block: nn.Module, x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    compute_dtype = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda") else x.dtype
    )
    return compute_dtype in (torch.bfloat16, torch.float16) and all(
        attention.attn_drop == 0.0
        for attention in (block.temporal_attn, block.spatial_attn)
    )


class _PatchGridPositions(nn.Module):
    def __init__(
        self,
        raw_num_frames: int,
        temporal_patch_size: int,
        num_groups: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        token_num_frames = raw_num_frames // temporal_patch_size
        centers = (
            torch.arange(token_num_frames, dtype=torch.float32) * temporal_patch_size
            + (temporal_patch_size - 1) / 2.0
        )
        self.register_buffer("frame_center", centers, persistent=False)
        self.temporal = ContinuousSinCosPosEmbed1D(embed_dim, theta=100.0)
        self.group = nn.Parameter(torch.zeros(1, 1, num_groups, embed_dim))
        nn.init.trunc_normal_(self.group, std=0.02)

    def forward(self, fps: torch.Tensor) -> torch.Tensor:
        fps = torch.as_tensor(fps, device=self.frame_center.device, dtype=torch.float32)
        if fps.ndim != 1:
            raise ValueError(f"fps must have shape [B], got {tuple(fps.shape)}")
        temporal = self.temporal(
            self.frame_center.unsqueeze(0) / fps.clamp_min(1.0).unsqueeze(1)
        )
        return temporal.unsqueeze(2) + self.group


class _TemporalGraphMeanStem(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        temporal_patch_size: int,
        spatial_grouping_name: str,
    ) -> None:
        super().__init__()
        self.groups = get_spatial_grouping(spatial_grouping_name)
        tokenizer = MotionFeatureTokenizer2D(embed_dim=embed_dim)
        self.num_joints = tokenizer.NUM_JOINTS
        self.trajectory_input_dim = len(TRAJECTORY_FIELDS)
        self.embed_dim = int(embed_dim)

        # Grouped convolution requires an equal channel count per joint. Route
        # each semantic field into a 14-channel joint slot and point unused
        # slots at one appended zero channel. Groups keep every joint's weights
        # independent while issuing one convolution instead of 30 small ones.
        padding_index = tokenizer.FEATURE_DIM
        self.register_buffer(
            "trajectory_feature_indices",
            torch.tensor((0, 2, 3, 4), dtype=torch.long),
            persistent=False,
        )
        feature_indices = []
        body_input_dims = []
        for joint, input_dim in enumerate(tokenizer.input_dims):
            if joint == 0:
                indices = [1]
            else:
                local_start = 5 + 3 * (joint - 1)
                indices = list(range(local_start, local_start + 3))
            indices.extend(range(92 + 6 * joint, 92 + 6 * (joint + 1)))
            indices.extend(range(272 + 3 * joint, 272 + 3 * (joint + 1)))
            contact = tokenizer.contact_by_joint.get(joint)
            if contact is not None:
                indices.append(362 + contact)
            body_input_dims.append(len(indices))
            feature_indices.append(indices)
        self.body_input_dims = tuple(body_input_dims)
        self.joint_input_dim = max(self.body_input_dims)
        feature_indices = [
            indices + [padding_index] * (self.joint_input_dim - len(indices))
            for indices in feature_indices
        ]
        self.register_buffer(
            "joint_feature_indices",
            torch.tensor(feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.temporal_conv = nn.Conv1d(
            self.num_joints * self.joint_input_dim,
            self.num_joints * self.embed_dim,
            kernel_size=temporal_patch_size,
            stride=temporal_patch_size,
            groups=self.num_joints,
        )
        self.trajectory_conv = nn.Conv1d(
            self.trajectory_input_dim,
            self.embed_dim,
            kernel_size=temporal_patch_size,
            stride=temporal_patch_size,
        )
        self.graph_projection = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.GELU()
        self.register_buffer(
            "adjacency", _normalized_group_adjacency(self.groups), persistent=False
        )
        group_pool = torch.zeros(len(self.groups), self.num_joints)
        for group_index, (_, joints) in enumerate(self.groups):
            group_pool[group_index, list(joints)] = 1.0 / len(joints)
        self.register_buffer("group_pool", group_pool, persistent=False)
        nn.init.trunc_normal_(self.temporal_conv.weight, std=0.02)
        if self.temporal_conv.bias is not None:
            nn.init.zeros_(self.temporal_conv.bias)
        nn.init.trunc_normal_(self.trajectory_conv.weight, std=0.02)
        if self.trajectory_conv.bias is not None:
            nn.init.zeros_(self.trajectory_conv.bias)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[-1] != MotionFeatureTokenizer2D.FEATURE_DIM:
            raise ValueError(
                f"2D tokenization requires motion [B,T,{MotionFeatureTokenizer2D.FEATURE_DIM}], "
                f"got {tuple(motion.shape)}"
            )
        trajectory = motion[..., self.trajectory_feature_indices].transpose(1, 2)
        trajectory = self.trajectory_conv(trajectory).transpose(1, 2).unsqueeze(2)
        padded_motion = F.pad(motion, (0, 1))
        routed = padded_motion[..., self.joint_feature_indices]
        routed = routed.permute(0, 2, 3, 1).reshape(
            motion.shape[0], self.num_joints * self.joint_input_dim, motion.shape[1]
        )
        projected = self.temporal_conv(routed)
        joints = projected.reshape(
            motion.shape[0], self.num_joints, self.embed_dim, projected.shape[-1]
        ).permute(0, 3, 1, 2)
        mixed = torch.einsum("ij,btjd->btid", self.adjacency, joints)
        joints = self.activation(joints + self.graph_projection(mixed))
        body = torch.matmul(self.group_pool, joints)
        return torch.cat([trajectory, body], dim=2)


class MotionPatchTransformer2D(nn.Module):
    """Axial encoder over temporal patches and pooled anatomical groups."""

    def __init__(
        self,
        in_chans: int = 366,
        num_frames: int = 300,
        num_joints: int = 30,
        temporal_patch_size: int = 3,
        spatial_grouping: str = "fine11",
        spatial_pooling: str = SPATIAL_POOLING,
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
            raise ValueError("MotionPatchTransformer2D requires motion_jepa_366_v1 on SOMA30")
        if spatial_pooling != SPATIAL_POOLING:
            raise ValueError(f"Only spatial_pooling={SPATIAL_POOLING!r} is supported")
        self.in_chans = 366
        self.num_frames = int(num_frames)
        self.num_joints = 30
        self.temporal_patch_size = int(temporal_patch_size)
        if self.temporal_patch_size <= 0 or self.num_frames < self.temporal_patch_size:
            raise ValueError("temporal_patch_size must be in [1, num_frames]")
        self.token_num_frames = self.num_frames // self.temporal_patch_size
        self.spatial_grouping = str(spatial_grouping)
        self.spatial_pooling = str(spatial_pooling)
        self.groups = get_spatial_grouping(self.spatial_grouping)
        self.spatial_token_names = (
            "trajectory", *[group_name for group_name, _ in self.groups]
        )
        self.token_num_joints = len(self.spatial_token_names)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.token_layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=self.num_frames,
            token_num_frames=self.token_num_frames,
            temporal_patch_size=self.temporal_patch_size,
            raw_num_joints=self.num_joints,
            token_num_joints=self.token_num_joints,
            trajectory_token_index=TRAJECTORY_TOKEN_INDEX,
            body_token_offset=BODY_TOKEN_OFFSET,
            spatial_token_names=self.spatial_token_names,
            trajectory_fields=TRAJECTORY_FIELDS,
        )
        self.patch_embed = _TemporalGraphMeanStem(
            self.embed_dim, self.temporal_patch_size, self.spatial_grouping
        )
        self.positions = _PatchGridPositions(
            self.num_frames,
            self.temporal_patch_size,
            self.token_num_joints,
            self.embed_dim,
        )
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

    def spatial_patch_signature(self) -> dict:
        return spatial_patch_signature(self.spatial_grouping)

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
        x = self.patch_embed(motion)
        x = x + self.positions(fps).to(device=x.device, dtype=x.dtype)
        frame_active = None
        if valid_frames is not None:
            frame_active = self.token_layout.valid_token_mask(
                valid_frames.to(device=x.device, dtype=torch.bool)
            )
        if masks is None:
            if frame_active is None:
                active = torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
            else:
                active = frame_active.unsqueeze(-1).expand(-1, -1, self.token_num_joints)
                x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        else:
            if not masks:
                raise ValueError("At least one context mask is required")
            prepared_masks = [mask.to(device=x.device, dtype=torch.bool) for mask in masks]
            expected = (len(motion), self.token_num_frames, self.token_num_joints)
            if any(tuple(mask.shape) != expected for mask in prepared_masks):
                raise ValueError(f"Context masks must have shape {expected}")
            if frame_active is not None:
                valid_grid = frame_active.unsqueeze(-1)
                if any((mask & ~valid_grid).any() for mask in prepared_masks):
                    raise ValueError("Context masks select an invalid or padded patch")
            active = torch.cat(prepared_masks, dim=0)
        use_packed = (
            masks is not None
            and _PACKED_ENCODER_ATTENTION
            and bool(self.blocks)
            and _supports_persistent_packing(self.blocks[0], x)
        )
        if use_packed:
            layout = prepare_packed_axial_layout(
                active, spatial_padded_seqlen=self.token_num_joints
            )
            cells = self.token_num_frames * self.token_num_joints
            expanded_batch = layout.dense_indices // cells
            source_indices = (
                (expanded_batch % len(motion)) * cells
                + layout.dense_indices % cells
            )
            packed = x.reshape(-1, x.shape[-1]).index_select(
                0, source_indices
            ).index_select(0, layout.temporal_order)
            for block in self.blocks:
                packed = block.forward_packed(packed, layout)
            canonical = self.norm(
                packed.index_select(0, layout.canonical_from_temporal)
            )
            return canonical.new_zeros(
                *active.shape, canonical.shape[-1]
            ).view(-1, canonical.shape[-1]).index_copy(
                0, layout.dense_indices, canonical
            ).view(*active.shape, canonical.shape[-1])
        if masks is not None:
            x = torch.cat([x] * len(masks), dim=0)
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        attention_masks = (
            self.blocks[0].prepare_attention_masks(active) if self.blocks else None
        )
        mlp_active_indices = None
        if (
            masks is not None
            and self.blocks
            and self.blocks[0].mlp.drop1.p == 0.0
            and self.blocks[0].mlp.drop2.p == 0.0
        ):
            mlp_active_indices = active.flatten().nonzero(as_tuple=False).flatten()
        for block in self.blocks:
            x = block(x, active, attention_masks, mlp_active_indices)
        return self.norm(x) * active.unsqueeze(-1).to(dtype=x.dtype)


class MotionPatchTransformerPredictor2D(nn.Module):
    """Predict target anatomical patch embeddings from a context grid."""

    def __init__(
        self,
        num_frames: int = 300,
        num_joints: int = 30,
        temporal_patch_size: int = 3,
        spatial_grouping: str = "fine11",
        spatial_pooling: str = SPATIAL_POOLING,
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
            raise ValueError("MotionPatchTransformerPredictor2D requires SOMA30")
        if spatial_pooling != SPATIAL_POOLING:
            raise ValueError(f"Only spatial_pooling={SPATIAL_POOLING!r} is supported")
        self.num_frames = int(num_frames)
        self.num_joints = 30
        self.temporal_patch_size = int(temporal_patch_size)
        if self.temporal_patch_size <= 0 or self.num_frames < self.temporal_patch_size:
            raise ValueError("temporal_patch_size must be in [1, num_frames]")
        self.token_num_frames = self.num_frames // self.temporal_patch_size
        self.spatial_grouping = str(spatial_grouping)
        self.spatial_pooling = str(spatial_pooling)
        self.groups = get_spatial_grouping(self.spatial_grouping)
        self.spatial_token_names = (
            "trajectory", *[group_name for group_name, _ in self.groups]
        )
        self.token_num_joints = len(self.spatial_token_names)
        self.embed_dim = int(embed_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.token_layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=self.num_frames,
            token_num_frames=self.token_num_frames,
            temporal_patch_size=self.temporal_patch_size,
            raw_num_joints=self.num_joints,
            token_num_joints=self.token_num_joints,
            trajectory_token_index=TRAJECTORY_TOKEN_INDEX,
            body_token_offset=BODY_TOKEN_OFFSET,
            spatial_token_names=self.spatial_token_names,
            trajectory_fields=TRAJECTORY_FIELDS,
        )
        self.input_proj = nn.Linear(self.embed_dim, self.predictor_embed_dim)
        self.positions = _PatchGridPositions(
            self.num_frames,
            self.temporal_patch_size,
            self.token_num_joints,
            self.predictor_embed_dim,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.predictor_embed_dim))
        self.stream_embed = nn.Parameter(
            torch.zeros(1, 2, 1, 1, self.predictor_embed_dim)
        )
        # Config-based construction enables the compact J-wide spatial buffer
        # when context and target masks are guaranteed to be disjoint.
        self._packed_spatial_disjoint = False
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

    def spatial_patch_signature(self) -> dict:
        return spatial_patch_signature(self.spatial_grouping)

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
        context_groups, position_groups, context_masks, target_masks = [], [], [], []
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
        expected = (len(context_grid), self.token_num_frames, self.token_num_joints)
        if tuple(context_active.shape) != expected or tuple(target_active.shape) != expected:
            raise ValueError(f"Predictor masks must have shape {expected}")
        active = torch.stack([context_active, target_active], dim=1)
        use_packed = (
            _PACKED_PREDICTOR_ATTENTION
            and bool(self.blocks)
            and _supports_persistent_packing(self.blocks[0], context)
        )
        if use_packed:
            layout = prepare_packed_axial_layout(
                active,
                spatial_padded_seqlen=(
                    self.token_num_joints
                    if self._packed_spatial_disjoint
                    else 2 * self.token_num_joints
                ),
            )
            cells = self.token_num_frames * self.token_num_joints
            batch_ids = layout.dense_indices // (2 * cells)
            remainder = layout.dense_indices % (2 * cells)
            stream_ids, cell_ids = remainder // cells, remainder % cells
            canonical_positions = torch.arange(
                len(layout.dense_indices), device=context.device
            )
            context_positions = canonical_positions[stream_ids == 0]
            target_positions = canonical_positions[stream_ids == 1]
            context_indices = batch_ids[stream_ids == 0] * cells + cell_ids[stream_ids == 0]
            target_indices = batch_ids[stream_ids == 1] * cells + cell_ids[stream_ids == 1]
            flat_context = context_grid.reshape(-1, context_grid.shape[-1])
            flat_position = position.reshape(-1, position.shape[-1])
            context_tokens = (
                self.input_proj(flat_context.index_select(0, context_indices))
                + flat_position.index_select(0, context_indices)
                + self.stream_embed[:, 0].reshape(1, -1)
            )
            target_tokens = (
                self.mask_token.reshape(1, -1)
                + flat_position.index_select(0, target_indices)
                + self.stream_embed[:, 1].reshape(1, -1)
            )
            canonical = context_tokens.new_empty(
                len(layout.dense_indices), self.predictor_embed_dim
            )
            canonical = canonical.index_copy(0, context_positions, context_tokens)
            canonical = canonical.index_copy(0, target_positions, target_tokens)
            packed = canonical.index_select(0, layout.temporal_order)
            for block in self.blocks:
                packed = block.forward_packed(packed, layout)
            if layout.target_from_temporal is None:
                raise RuntimeError("Predictor packed layout has no target selector")
            target = packed.index_select(0, layout.target_from_temporal)
            target = self.output_proj(self.norm(target))
            return target.reshape(len(context_grid), -1, self.embed_dim)
        context_stream = self.input_proj(context_grid) + position + self.stream_embed[:, 0]
        target_stream = self.mask_token + position + self.stream_embed[:, 1]
        context_stream = context_stream * context_active.unsqueeze(-1).to(context_stream.dtype)
        target_stream = target_stream * target_active.unsqueeze(-1).to(target_stream.dtype)
        x = torch.stack([context_stream, target_stream], dim=1)
        attention_masks = (
            self.blocks[0].prepare_attention_masks(active) if self.blocks else None
        )
        mlp_active_indices = None
        if (
            self.blocks
            and self.blocks[0].mlp.drop1.p == 0.0
            and self.blocks[0].mlp.drop2.p == 0.0
        ):
            mlp_active_indices = active.flatten().nonzero(as_tuple=False).flatten()
        for block in self.blocks:
            x = block(x, active, attention_masks, mlp_active_indices)
        target = self.output_proj(self.norm(x[:, 1]))
        return gather_grid(target, target_active)


def _encoder(size: str, **kwargs) -> MotionPatchTransformer2D:
    return MotionPatchTransformer2D(**MODEL_SPECS[size], **kwargs)


def _predictor(size: str, **kwargs) -> MotionPatchTransformerPredictor2D:
    return MotionPatchTransformerPredictor2D(**PREDICTOR_SPECS[size], **kwargs)


def mot_patch_tiny_2d(**kwargs):
    return _encoder("tiny", **kwargs)


def mot_patch_small_2d(**kwargs):
    return _encoder("small", **kwargs)


def mot_patch_base_2d(**kwargs):
    return _encoder("base", **kwargs)


def mot_patch_large_2d(**kwargs):
    return _encoder("large", **kwargs)


def mot_patch_huge_2d(**kwargs):
    return _encoder("huge", **kwargs)


def mot_patch_giant_2d(**kwargs):
    return _encoder("giant", **kwargs)


def mot_predictor_patch_tiny_2d(**kwargs):
    return _predictor("tiny", **kwargs)


def mot_predictor_patch_small_2d(**kwargs):
    return _predictor("small", **kwargs)


def mot_predictor_patch_base_2d(**kwargs):
    return _predictor("base", **kwargs)


def mot_predictor_patch_large_2d(**kwargs):
    return _predictor("large", **kwargs)


def mot_predictor_patch_huge_2d(**kwargs):
    return _predictor("huge", **kwargs)


def mot_predictor_patch_giant_2d(**kwargs):
    return _predictor("giant", **kwargs)


__all__ = [
    "MotionPatchTransformer2D",
    "MotionPatchTransformerPredictor2D",
    "SPATIAL_GROUPINGS",
    "SPATIAL_GRAPH_VERSION",
    "SPATIAL_POOLING",
    "get_spatial_grouping",
    "spatial_patch_signature",
    "mot_patch_tiny_2d", "mot_patch_small_2d", "mot_patch_base_2d",
    "mot_patch_large_2d", "mot_patch_huge_2d", "mot_patch_giant_2d",
    "mot_predictor_patch_tiny_2d", "mot_predictor_patch_small_2d",
    "mot_predictor_patch_base_2d", "mot_predictor_patch_large_2d",
    "mot_predictor_patch_huge_2d", "mot_predictor_patch_giant_2d",
]
