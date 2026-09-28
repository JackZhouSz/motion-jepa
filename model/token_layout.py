"""Token-grid geometry shared by MotionJEPA models and data plumbing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenLayout:
    """Describe how a model maps raw motion samples to transformer tokens."""

    kind: str
    patchified: bool
    raw_num_frames: int
    token_num_frames: int
    temporal_patch_size: int = 1
    raw_num_joints: int | None = None
    token_num_joints: int | None = None
    trajectory_token_index: int | None = None
    body_token_offset: int | None = None
    spatial_token_names: tuple[str, ...] | None = None
    trajectory_fields: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"1d", "2d"}:
            raise ValueError(f"Unsupported token layout kind: {self.kind!r}")
        if min(
            int(self.raw_num_frames),
            int(self.token_num_frames),
            int(self.temporal_patch_size),
        ) <= 0:
            raise ValueError("Frame counts and temporal_patch_size must be positive")
        expected = self.raw_num_frames // self.temporal_patch_size
        if self.token_num_frames != expected:
            raise ValueError(
                f"token_num_frames must equal floor(raw_num_frames / patch_size): "
                f"expected {expected}, got {self.token_num_frames}"
            )
        if self.kind == "1d" and (
            self.raw_num_joints is not None or self.token_num_joints is not None
        ):
            raise ValueError("1D token layouts do not define joint dimensions")
        if self.kind == "2d" and min(
            int(self.raw_num_joints or 0), int(self.token_num_joints or 0)
        ) <= 0:
            raise ValueError("2D token layouts require positive joint dimensions")
        if self.spatial_token_names is not None:
            if self.kind != "2d":
                raise ValueError("Only 2D token layouts define spatial token names")
            if len(self.spatial_token_names) != int(self.token_num_joints):
                raise ValueError(
                    "spatial_token_names must match token_num_joints: "
                    f"{len(self.spatial_token_names)} != {self.token_num_joints}"
                )
        trajectory_metadata = (
            self.trajectory_token_index,
            self.body_token_offset,
            self.trajectory_fields,
        )
        if any(value is not None for value in trajectory_metadata):
            if any(value is None for value in trajectory_metadata):
                raise ValueError("Trajectory token metadata must be configured together")
            if not 0 <= int(self.trajectory_token_index) < int(self.token_num_joints or 0):
                raise ValueError("trajectory_token_index is outside the spatial token grid")
            if not 0 <= int(self.body_token_offset) <= int(self.token_num_joints or 0):
                raise ValueError("body_token_offset is outside the spatial token grid")

    def valid_token_lengths(self, raw_lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.as_tensor(raw_lengths)
        if ((lengths < 0) | (lengths > self.raw_num_frames)).any():
            raise ValueError(
                f"Raw valid lengths must be in [0, {self.raw_num_frames}]"
            )
        return torch.div(lengths, self.temporal_patch_size, rounding_mode="floor")

    def valid_token_mask(self, raw_valid_frames: torch.Tensor) -> torch.Tensor:
        active = torch.as_tensor(raw_valid_frames, dtype=torch.bool)
        if active.ndim != 2 or active.shape[1] != self.raw_num_frames:
            raise ValueError(
                f"Expected raw valid mask [B,{self.raw_num_frames}], got {tuple(active.shape)}"
            )
        usable = active[:, : self.token_num_frames * self.temporal_patch_size]
        return usable.reshape(
            len(active), self.token_num_frames, self.temporal_patch_size
        ).all(dim=-1)

    def signature(self) -> dict[str, int | str | bool | list[str] | None]:
        return {
            "kind": self.kind,
            "patchified": self.patchified,
            "raw_num_frames": self.raw_num_frames,
            "token_num_frames": self.token_num_frames,
            "temporal_patch_size": self.temporal_patch_size,
            "raw_num_joints": self.raw_num_joints,
            "token_num_joints": self.token_num_joints,
            "trajectory_token_index": self.trajectory_token_index,
            "body_token_offset": self.body_token_offset,
            "spatial_token_names": (
                list(self.spatial_token_names)
                if self.spatial_token_names is not None else None
            ),
            "trajectory_fields": (
                list(self.trajectory_fields)
                if self.trajectory_fields is not None else None
            ),
        }


__all__ = ["TokenLayout"]
