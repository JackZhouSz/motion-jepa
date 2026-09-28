"""Shared transformer building blocks for MotionJEPA models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn


_PACKED_SPATIAL_SDPA = True


@dataclass(frozen=True)
class PackedAxialLayout:
    dense_indices: torch.Tensor
    temporal_order: torch.Tensor
    canonical_from_temporal: torch.Tensor
    spatial_from_temporal: torch.Tensor
    temporal_from_spatial: torch.Tensor
    temporal_cu_seqlens: torch.Tensor
    spatial_cu_seqlens: torch.Tensor
    temporal_batch_ids: torch.Tensor
    spatial_batch_ids: torch.Tensor
    temporal_max_seqlen: int
    spatial_max_seqlen: int
    logical_batch_size: int
    target_from_temporal: torch.Tensor | None = None
    spatial_padded_indices: torch.Tensor | None = None
    spatial_padded_mask: torch.Tensor | None = None
    spatial_padded_seqlen: int | None = None
    spatial_num_rows: int | None = None


def _cu_seqlens(rows: torch.Tensor) -> torch.Tensor:
    lengths = rows.sum(dim=1, dtype=torch.int32)
    return torch.cat(
        [torch.zeros(1, device=rows.device, dtype=torch.int32),
         lengths.cumsum(0, dtype=torch.int32)]
    )


def prepare_packed_axial_layout(
    active: torch.Tensor,
    spatial_padded_seqlen: int | None = None,
) -> PackedAxialLayout:
    """Prepare persistent packed ordering for encoder or predictor axial grids."""
    if active.ndim not in (3, 4):
        raise ValueError(f"Packed axial mask must be 3D or 4D, got {active.shape}")
    active = active.to(dtype=torch.bool)
    dense_indices = active.flatten().nonzero(as_tuple=False).flatten()
    canonical = torch.arange(len(dense_indices), device=active.device)
    if active.ndim == 3:
        batch, frames, joints = active.shape
        batch_ids = dense_indices // (frames * joints)
        remainder = dense_indices % (frames * joints)
        frame_ids, joint_ids = remainder // joints, remainder % joints
        temporal_key = (batch_ids * joints + joint_ids) * frames + frame_ids
        spatial_key = dense_indices
        temporal_rows = active.permute(0, 2, 1).reshape(batch * joints, frames)
        spatial_rows = active.reshape(batch * frames, joints)
        target_positions = None
        temporal_max, spatial_max = frames, joints
    else:
        batch, streams, frames, joints = active.shape
        batch_ids = dense_indices // (streams * frames * joints)
        remainder = dense_indices % (streams * frames * joints)
        stream_ids = remainder // (frames * joints)
        remainder = remainder % (frames * joints)
        frame_ids, joint_ids = remainder // joints, remainder % joints
        temporal_key = (
            ((batch_ids * joints + joint_ids) * streams + stream_ids) * frames
            + frame_ids
        )
        spatial_key = (
            ((batch_ids * frames + frame_ids) * streams + stream_ids) * joints
            + joint_ids
        )
        temporal_rows = active.permute(0, 3, 1, 2).reshape(
            batch * joints, streams * frames
        )
        spatial_rows = active.permute(0, 2, 1, 3).reshape(
            batch * frames, streams * joints
        )
        target_positions = canonical[stream_ids == 1]
        temporal_max, spatial_max = streams * frames, streams * joints
    temporal_order = temporal_key.argsort()
    spatial_order = spatial_key.argsort()
    inverse_temporal = torch.empty_like(temporal_order)
    inverse_temporal[temporal_order] = canonical
    inverse_spatial = torch.empty_like(spatial_order)
    inverse_spatial[spatial_order] = canonical
    spatial_cu_seqlens = _cu_seqlens(spatial_rows)
    spatial_padded_indices = None
    spatial_padded_mask = None
    spatial_num_rows = None
    if _PACKED_SPATIAL_SDPA:
        if spatial_padded_seqlen is None:
            spatial_padded_seqlen = spatial_max
        if not 0 < spatial_padded_seqlen <= spatial_max:
            raise ValueError(
                f"Spatial padded length must be in [1, {spatial_max}], "
                f"got {spatial_padded_seqlen}"
            )
        spatial_lengths = spatial_cu_seqlens[1:] - spatial_cu_seqlens[:-1]
        torch._assert_async(
            (spatial_lengths <= spatial_padded_seqlen).all(),
            "Packed spatial row exceeds its configured padded length",
        )
        spatial_num_rows = len(spatial_lengths)
        spatial_row_ids = torch.repeat_interleave(
            torch.arange(spatial_num_rows, device=active.device),
            spatial_lengths.to(dtype=torch.long),
        )
        spatial_positions = torch.arange(
            len(dense_indices), device=active.device
        ) - torch.repeat_interleave(
            spatial_cu_seqlens[:-1].to(dtype=torch.long),
            spatial_lengths.to(dtype=torch.long),
        )
        spatial_padded_indices = (
            spatial_row_ids * spatial_padded_seqlen + spatial_positions
        )
        spatial_padded_mask = (
            torch.arange(spatial_padded_seqlen, device=active.device).unsqueeze(0)
            < spatial_lengths.unsqueeze(1)
        )
    return PackedAxialLayout(
        dense_indices=dense_indices,
        temporal_order=temporal_order,
        canonical_from_temporal=inverse_temporal,
        spatial_from_temporal=inverse_temporal.index_select(0, spatial_order),
        temporal_from_spatial=inverse_spatial.index_select(0, temporal_order),
        temporal_cu_seqlens=_cu_seqlens(temporal_rows),
        spatial_cu_seqlens=spatial_cu_seqlens,
        temporal_batch_ids=batch_ids.index_select(0, temporal_order),
        spatial_batch_ids=batch_ids.index_select(0, spatial_order),
        temporal_max_seqlen=temporal_max,
        spatial_max_seqlen=spatial_max,
        logical_batch_size=batch,
        target_from_temporal=(
            inverse_temporal.index_select(0, target_positions)
            if target_positions is not None else None
        ),
        spatial_padded_indices=spatial_padded_indices,
        spatial_padded_mask=spatial_padded_mask,
        spatial_padded_seqlen=spatial_padded_seqlen,
        spatial_num_rows=spatial_num_rows,
    )


def _make_safe_attention_mask(active: torch.Tensor) -> torch.Tensor:
    """Ensure every attention row has a key without synchronizing with the host."""
    safe_active = active.to(dtype=torch.bool).clone()
    empty = ~safe_active.any(dim=1)
    safe_active[:, 0] = safe_active[:, 0] | empty
    return safe_active


def _active_mlp_update(
    block: nn.Module,
    x: torch.Tensor,
    active_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Apply token-wise normalization and MLP only at active grid positions."""
    if (
        active_indices is None
        or block.mlp.drop1.p != 0.0
        or block.mlp.drop2.p != 0.0
    ):
        return block.mlp(block.mlp_norm(x))
    flat = x.reshape(-1, x.shape[-1])
    active = flat.index_select(0, active_indices)
    update = block.mlp(block.mlp_norm(active))
    return update.new_zeros(flat.shape).index_copy(
        0, active_indices, update
    ).reshape_as(x)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random.floor() / keep

    def forward_packed(
        self, x: torch.Tensor, batch_ids: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        random = keep + torch.rand(
            (batch_size, 1), dtype=x.dtype, device=x.device
        )
        mask = random.floor().index_select(0, batch_ids)
        return x * mask / keep


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.activation(self.fc1(x)))))


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional pre-sanitized active-key mask."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"Embedding dimension {dim} is not divisible by {num_heads} heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = float(attn_drop)

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention_mask = None
        if active is not None:
            active = active.to(device=x.device, dtype=torch.bool)
            if active.shape != (batch, tokens):
                raise ValueError(
                    f"Active mask must have shape {(batch, tokens)}, got {tuple(active.shape)}"
                )
            # Callers must ensure that every row contains at least one active key.
            attention_mask = active[:, None, None, :]
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(output))

    def forward_varlen(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        tokens, dim = x.shape
        qkv = self.qkv(x).reshape(tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(1)
        output = varlen_attn(
            query, key, value, cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen, is_causal=False,
        ).reshape(tokens, dim)
        return self.proj_drop(self.proj(output))

    def forward_padded(
        self,
        x: torch.Tensor,
        padded_indices: torch.Tensor,
        active: torch.Tensor,
        num_rows: int,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run fixed-width SDPA for short packed rows and return packed output."""
        tokens, dim = x.shape
        qkv = self.qkv(x).reshape(tokens, 3, self.num_heads, self.head_dim)
        padded = qkv.new_zeros(
            num_rows * max_seqlen, 3, self.num_heads, self.head_dim
        ).index_copy(0, padded_indices, qkv).view(
            num_rows, max_seqlen, 3, self.num_heads, self.head_dim
        )
        query, key, value = padded.unbind(2)
        output = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            attn_mask=active[:, None, None, :],
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = output.permute(0, 2, 1, 3).reshape(
            num_rows * max_seqlen, dim
        ).index_select(0, padded_indices)
        return self.proj_drop(self.proj(output))


def _packed_spatial_attention(
    attention: SelfAttention,
    x: torch.Tensor,
    layout: PackedAxialLayout,
) -> torch.Tensor:
    if layout.spatial_padded_indices is None:
        return attention.forward_varlen(
            x, layout.spatial_cu_seqlens, layout.spatial_max_seqlen
        )
    if (
        layout.spatial_padded_mask is None
        or layout.spatial_padded_seqlen is None
        or layout.spatial_num_rows is None
    ):
        raise RuntimeError("Incomplete packed spatial SDPA metadata")
    return attention.forward_padded(
        x,
        layout.spatial_padded_indices,
        layout.spatial_padded_mask,
        layout.spatial_num_rows,
        layout.spatial_padded_seqlen,
    )


def _packed_drop_path(
    module: nn.Module,
    x: torch.Tensor,
    batch_ids: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if isinstance(module, DropPath):
        return module.forward_packed(x, batch_ids, batch_size)
    return x

class TransformerBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.attn_norm = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.attn_norm(x), active))
        if active is not None:
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        if active is not None:
            x = x * active.unsqueeze(-1).to(dtype=x.dtype)
        return x


class AxialBlock2D(nn.Module):
    """Temporal-then-spatial attention over a masked ``[B,T,J,D]`` grid."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.temporal_norm = norm_layer(dim)
        self.temporal_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.spatial_norm = norm_layer(dim)
        self.spatial_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    @staticmethod
    def _zero_inactive(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return x * active.unsqueeze(-1).to(dtype=x.dtype)

    @staticmethod
    def prepare_attention_masks(
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, joints = active.shape
        temporal = active.permute(0, 2, 1).reshape(batch * joints, frames)
        spatial = active.reshape(batch * frames, joints)
        return (
            _make_safe_attention_mask(temporal),
            _make_safe_attention_mask(spatial),
        )

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        attention_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
        mlp_active_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, frames, joints, dim = x.shape
        if attention_masks is None:
            attention_masks = self.prepare_attention_masks(active)
        temporal_active, spatial_active = attention_masks
        temporal = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(
            batch * joints, frames, dim
        )
        temporal = self.temporal_attn(temporal, temporal_active).reshape(
            batch, joints, frames, dim
        ).permute(0, 2, 1, 3)
        x = self._zero_inactive(x + self.drop_path(temporal), active)

        spatial = self.spatial_norm(x).reshape(batch * frames, joints, dim)
        spatial = self.spatial_attn(spatial, spatial_active).reshape(
            batch, frames, joints, dim
        )
        x = self._zero_inactive(x + self.drop_path(spatial), active)
        return self._zero_inactive(
            x + self.drop_path(_active_mlp_update(self, x, mlp_active_indices)),
            active,
        )

    def forward_packed(
        self, x: torch.Tensor, layout: PackedAxialLayout
    ) -> torch.Tensor:
        update = self.temporal_attn.forward_varlen(
            self.temporal_norm(x), layout.temporal_cu_seqlens,
            layout.temporal_max_seqlen,
        )
        x = x + _packed_drop_path(
            self.drop_path, update, layout.temporal_batch_ids,
            layout.logical_batch_size,
        )
        x = x.index_select(0, layout.spatial_from_temporal)
        update = _packed_spatial_attention(
            self.spatial_attn, self.spatial_norm(x), layout
        )
        x = x + _packed_drop_path(
            self.drop_path, update, layout.spatial_batch_ids,
            layout.logical_batch_size,
        )
        update = self.mlp(self.mlp_norm(x))
        x = x + _packed_drop_path(
            self.drop_path, update, layout.spatial_batch_ids,
            layout.logical_batch_size,
        )
        return x.index_select(0, layout.temporal_from_spatial)


class PredictorAxialBlock2D(nn.Module):
    """Axial attention over separate context and target-query streams."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.temporal_norm = norm_layer(dim)
        self.temporal_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.spatial_norm = norm_layer(dim)
        self.spatial_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    @staticmethod
    def _zero_inactive(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        return x * active.unsqueeze(-1).to(dtype=x.dtype)

    @staticmethod
    def prepare_attention_masks(
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, streams, frames, joints = active.shape
        temporal = active.permute(0, 3, 1, 2).reshape(
            batch * joints, streams * frames
        )
        spatial = active.permute(0, 2, 1, 3).reshape(
            batch * frames, streams * joints
        )
        return (
            _make_safe_attention_mask(temporal),
            _make_safe_attention_mask(spatial),
        )

    def forward(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        attention_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
        mlp_active_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, streams, frames, joints, dim = x.shape
        if attention_masks is None:
            attention_masks = self.prepare_attention_masks(active)
        temporal_active, spatial_active = attention_masks
        temporal = self.temporal_norm(x).permute(0, 3, 1, 2, 4).reshape(
            batch * joints, streams * frames, dim
        )
        temporal = self.temporal_attn(temporal, temporal_active).reshape(
            batch, joints, streams, frames, dim
        ).permute(0, 2, 3, 1, 4)
        x = self._zero_inactive(x + self.drop_path(temporal), active)

        spatial = self.spatial_norm(x).permute(0, 2, 1, 3, 4).reshape(
            batch * frames, streams * joints, dim
        )
        spatial = self.spatial_attn(spatial, spatial_active).reshape(
            batch, frames, streams, joints, dim
        ).permute(0, 2, 1, 3, 4)
        x = self._zero_inactive(x + self.drop_path(spatial), active)
        return self._zero_inactive(
            x + self.drop_path(_active_mlp_update(self, x, mlp_active_indices)),
            active,
        )

    def forward_packed(
        self, x: torch.Tensor, layout: PackedAxialLayout
    ) -> torch.Tensor:
        update = self.temporal_attn.forward_varlen(
            self.temporal_norm(x), layout.temporal_cu_seqlens,
            layout.temporal_max_seqlen,
        )
        x = x + _packed_drop_path(
            self.drop_path, update, layout.temporal_batch_ids,
            layout.logical_batch_size,
        )
        x = x.index_select(0, layout.spatial_from_temporal)
        update = _packed_spatial_attention(
            self.spatial_attn, self.spatial_norm(x), layout
        )
        x = x + _packed_drop_path(
            self.drop_path, update, layout.spatial_batch_ids,
            layout.logical_batch_size,
        )
        update = self.mlp(self.mlp_norm(x))
        x = x + _packed_drop_path(
            self.drop_path, update, layout.spatial_batch_ids,
            layout.logical_batch_size,
        )
        return x.index_select(0, layout.temporal_from_spatial)


def initialize_transformer(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


__all__ = [
    "AxialBlock2D",
    "DropPath",
    "MLP",
    "PackedAxialLayout",
    "PredictorAxialBlock2D",
    "SelfAttention",
    "TransformerBlock1D",
    "initialize_transformer",
    "prepare_packed_axial_layout",
]
