# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The stable 366-dimensional MotionJEPA motion representation."""

from __future__ import annotations

import torch

from skeleton import SOMASkeleton30, global_rots_to_local_rots

from ..geometry import cont6d_to_matrix, matrix_to_cont6d, velocity, y_rotation
from ..feet import foot_detect_from_pos_and_vel
from .base import MotionRepBase


class MotionJEPAMotionRep(MotionRepBase):
    FORMAT_NAME = "motion_jepa_366_v1"
    FEATURE_DIM = 366

    ROOT_POSITION = slice(0, 3)
    ROOT_HEADING = slice(3, 5)
    LOCAL_POSITIONS = slice(5, 92)
    GLOBAL_ROTATIONS = slice(92, 272)
    VELOCITIES = slice(272, 362)
    FOOT_CONTACTS = slice(362, 366)

    def __init__(self, skeleton: SOMASkeleton30 | None = None, fps: int = 30):
        super().__init__(skeleton or SOMASkeleton30(), fps)
        if self.skeleton.nbjoints != 30:
            raise ValueError("motion_jepa_366_v1 requires SOMA30.")

    def __call__(
        self,
        local_rotations: torch.Tensor,
        root_positions: torch.Tensor,
        to_canonicalize: bool = True,
    ) -> torch.Tensor:
        unbatched = local_rotations.ndim == 4
        if unbatched:
            local_rotations = local_rotations.unsqueeze(0)
            root_positions = root_positions.unsqueeze(0)
        global_rotations, global_positions, local_positions = self.skeleton.fk(local_rotations, root_positions)
        right_hip, left_hip = self.skeleton.hip_joint_idx
        difference = global_positions[:, :, right_hip] - global_positions[:, :, left_hip]
        angle = torch.atan2(difference[..., 2], -difference[..., 0])
        heading = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        ground_offset = torch.zeros_like(root_positions)
        ground_offset[..., 1] = root_positions[..., 1]
        local_positions = local_positions[:, :, 1:] + ground_offset[:, :, None]
        velocities = velocity(global_positions, self.fps)
        contacts = foot_detect_from_pos_and_vel(
            global_positions, velocities, self.skeleton, 0.15, 0.10
        )
        features = torch.cat(
            [
                root_positions,
                heading,
                local_positions.flatten(-2),
                matrix_to_cont6d(global_rotations).flatten(-2),
                velocities.flatten(-2),
                contacts,
            ],
            dim=-1,
        )
        if to_canonicalize:
            features = self.canonicalize(features)
        if features.shape[-1] != self.FEATURE_DIM:
            raise RuntimeError(f"Unexpected MotionJEPA feature shape: {features.shape}")
        return features[0] if unbatched else features

    def encode(
        self,
        local_rotations: torch.Tensor,
        root_positions: torch.Tensor,
        canonicalize: bool = True,
    ) -> torch.Tensor:
        """Compatibility wrapper around the Ardy/Kimodo-style call interface."""
        return self(
            local_rotations,
            root_positions,
            to_canonicalize=canonicalize,
        )

    def canonicalize(self, features: torch.Tensor) -> torch.Tensor:
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        output = features.clone()
        heading = output[..., self.ROOT_HEADING]
        first_angle = torch.atan2(heading[:, 0, 1], heading[:, 0, 0])
        rotation = y_rotation(-first_angle)
        output[..., self.ROOT_POSITION] = output[..., self.ROOT_POSITION] @ rotation.transpose(-2, -1)[:, None]
        local_positions = output[..., self.LOCAL_POSITIONS].reshape(*output.shape[:2], 29, 3)
        output[..., self.LOCAL_POSITIONS] = (
            local_positions @ rotation.transpose(-2, -1)[:, None, None]
        ).flatten(-2)
        velocities = output[..., self.VELOCITIES].reshape(*output.shape[:2], 30, 3)
        output[..., self.VELOCITIES] = (
            velocities @ rotation.transpose(-2, -1)[:, None, None]
        ).flatten(-2)
        global_rotations = cont6d_to_matrix(
            output[..., self.GLOBAL_ROTATIONS].reshape(*output.shape[:2], 30, 6)
        )
        global_rotations = rotation[:, None, None] @ global_rotations
        output[..., self.GLOBAL_ROTATIONS] = matrix_to_cont6d(global_rotations).flatten(-2)
        cos, sin = torch.cos(-first_angle), torch.sin(-first_angle)
        rotation2d = torch.stack([cos, sin, -sin, cos], dim=-1).reshape(-1, 2, 2)
        output[..., self.ROOT_HEADING] = heading @ rotation2d[:, None]
        origin = output[:, 0, self.ROOT_POSITION][:, [0, 2]].clone()
        output[..., 0] -= origin[:, None, 0]
        output[..., 2] -= origin[:, None, 1]
        return output[0] if unbatched else output

    def inverse(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        root_positions = features[..., self.ROOT_POSITION]
        heading = features[..., self.ROOT_HEADING]
        global_rotations = cont6d_to_matrix(
            features[..., self.GLOBAL_ROTATIONS].reshape(*features.shape[:2], 30, 6)
        )
        batch, frames = features.shape[:2]
        flat_global = global_rotations.reshape(-1, 30, 3, 3)
        local_rotations = global_rots_to_local_rots(
            flat_global, self.skeleton.parents
        ).reshape(
            batch, frames, 30, 3, 3
        )
        _, positions, _ = self.skeleton.fk(local_rotations, root_positions)
        output = {
            "local_rot_mats": local_rotations,
            "global_rot_mats": global_rotations,
            "posed_joints": positions,
            "root_positions": root_positions,
            "smooth_root_pos": root_positions,
            "foot_contacts": features[..., self.FOOT_CONTACTS] > 0.5,
            "global_root_heading": heading,
        }
        if unbatched:
            return {key: value[0] for key, value in output.items()}
        return output

    def decode(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compatibility alias for :meth:`inverse`."""
        return self.inverse(features)


# Backward-compatible name for already processed data tooling.
MotionJEPARepresentation = MotionJEPAMotionRep

__all__ = ["MotionJEPAMotionRep", "MotionJEPARepresentation"]
