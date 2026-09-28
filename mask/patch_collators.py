"""Mask collation on temporal-patch token coordinates."""

from __future__ import annotations

import torch

from model.token_layout import TokenLayout

from .collators import MaskCollator1D, MaskCollator2D


class PatchMaskCollator1D(MaskCollator1D):
    """Sample 1D masks after mapping raw valid lengths to complete patches."""

    def __init__(
        self,
        raw_num_frames: int,
        temporal_patch_size: int = 3,
        **kwargs,
    ) -> None:
        self.layout = TokenLayout(
            kind="1d",
            patchified=True,
            raw_num_frames=int(raw_num_frames),
            token_num_frames=int(raw_num_frames) // int(temporal_patch_size),
            temporal_patch_size=int(temporal_patch_size),
        )
        super().__init__(num_frames=self.layout.token_num_frames, **kwargs)
        self._configuration = {
            **self._configuration,
            "variant": "patch_1d",
            "raw_num_frames": self.layout.raw_num_frames,
            "token_num_frames": self.layout.token_num_frames,
            "temporal_patch_size": self.layout.temporal_patch_size,
        }

    def __call__(self, batch):
        raw_lengths = torch.tensor(
            [
                int(sample[2]) if len(sample) >= 3 else self.layout.raw_num_frames
                for sample in batch
            ],
            dtype=torch.long,
        )
        token_lengths = self.layout.valid_token_lengths(raw_lengths)
        if (token_lengths < 1).any():
            raise ValueError(
                "Every sample must contain at least one complete temporal patch"
            )
        proxy_batch = []
        for sample, token_length in zip(batch, token_lengths.tolist()):
            values = list(sample)
            if len(values) >= 3:
                values[2] = token_length
            proxy_batch.append(tuple(values))
        _, contexts, targets = super().__call__(proxy_batch)
        return torch.utils.data.default_collate(batch), contexts, targets


class PatchMaskCollator2D(MaskCollator2D):
    """Sample 2D masks on complete temporal patches and pooled body groups."""

    def __init__(
        self,
        raw_num_frames: int,
        raw_num_joints: int,
        token_num_joints: int,
        temporal_patch_size: int = 3,
        spatial_grouping: str = "fine11",
        spatial_pooling: str = "graph_mean",
        **kwargs,
    ) -> None:
        body_groups = {"joint30": 30, "fine11": 11, "coarse7": 7}
        if spatial_grouping not in body_groups:
            raise ValueError(
                f"Unknown spatial_grouping {spatial_grouping!r}; "
                f"choose one of: {', '.join(sorted(body_groups))}"
            )
        if int(raw_num_joints) != 30:
            raise ValueError("PatchMaskCollator2D requires SOMA30")
        accepted_tokens = {
            body_groups[spatial_grouping],
            body_groups[spatial_grouping] + 1,
        }
        if int(token_num_joints) not in accepted_tokens:
            raise ValueError(
                f"spatial_grouping={spatial_grouping!r} requires "
                f"token_num_joints in {sorted(accepted_tokens)}"
            )
        if spatial_pooling != "graph_mean":
            raise ValueError("PatchMaskCollator2D supports only graph_mean pooling")
        self.layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=int(raw_num_frames),
            token_num_frames=int(raw_num_frames) // int(temporal_patch_size),
            temporal_patch_size=int(temporal_patch_size),
            raw_num_joints=int(raw_num_joints),
            token_num_joints=int(token_num_joints),
        )
        self.spatial_grouping = str(spatial_grouping)
        self.spatial_pooling = str(spatial_pooling)
        super().__init__(
            num_frames=self.layout.token_num_frames,
            num_joints=int(self.layout.token_num_joints),
            **kwargs,
        )
        self._configuration = {
            **self._configuration,
            "variant": "patch_2d",
            "raw_num_frames": self.layout.raw_num_frames,
            "token_num_frames": self.layout.token_num_frames,
            "temporal_patch_size": self.layout.temporal_patch_size,
            "raw_num_joints": self.layout.raw_num_joints,
            "token_num_joints": self.layout.token_num_joints,
            "spatial_grouping": self.spatial_grouping,
            "spatial_pooling": self.spatial_pooling,
        }

    def __call__(self, batch):
        raw_lengths = torch.tensor(
            [
                int(sample[2]) if len(sample) >= 3 else self.layout.raw_num_frames
                for sample in batch
            ],
            dtype=torch.long,
        )
        token_lengths = self.layout.valid_token_lengths(raw_lengths)
        if (token_lengths < 1).any():
            raise ValueError(
                "Every sample must contain at least one complete temporal patch"
            )
        proxy_batch = []
        for sample, token_length in zip(batch, token_lengths.tolist()):
            values = list(sample)
            if len(values) >= 3:
                values[2] = token_length
            proxy_batch.append(tuple(values))
        _, contexts, targets = super().__call__(proxy_batch)
        return torch.utils.data.default_collate(batch), contexts, targets


__all__ = ["PatchMaskCollator1D", "PatchMaskCollator2D"]
