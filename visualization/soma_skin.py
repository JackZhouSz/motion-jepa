# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""SOMA linear-blend skinning used by the MotionJEPA viewer."""

from __future__ import annotations

import numpy as np
import torch

from skeleton import SOMASkeleton30, SOMASkeleton77
from skeleton.asset_paths import skeleton_asset_path

SKEL_PATH = "somaskel77"
SKIN_NAME = "skin_standard.npz"
ASSETS_SKIN_PATH = skeleton_asset_path(SKEL_PATH, SKIN_NAME)


class SOMASkin:
    def __init__(self, skeleton: SOMASkeleton30 | SOMASkeleton77 | None = None):
        self.skeleton_input = skeleton or SOMASkeleton77()
        if not isinstance(self.skeleton_input, (SOMASkeleton30, SOMASkeleton77)):
            raise TypeError("SOMASkin only supports SOMASkeleton30 or SOMASkeleton77.")
        with np.load(ASSETS_SKIN_PATH, allow_pickle=False) as data:
            bind = np.asarray(data["bind_rig_transform"], dtype=np.float32)
            self.bind_inverse = torch.from_numpy(np.linalg.inv(bind))
            self.vertices = torch.from_numpy(np.asarray(data["bind_vertices"], dtype=np.float32))
            self.faces = np.asarray(data["faces"], dtype=np.int32)
            self.indices = torch.from_numpy(np.asarray(data["lbs_indices"], dtype=np.int64))
            self.weights = torch.from_numpy(np.asarray(data["lbs_weights"], dtype=np.float32))

    def pose(self, global_rotations: torch.Tensor, positions: torch.Tensor) -> np.ndarray:
        device, dtype = global_rotations.device, global_rotations.dtype
        transform = torch.eye(4, device=device, dtype=dtype).repeat(77, 1, 1)
        transform[:, :3, :3] = global_rotations
        transform[:, :3, 3] = positions
        affine = (transform @ self.bind_inverse.to(device=device, dtype=dtype))[:, :3]
        vertices = self.vertices.to(device=device, dtype=dtype)
        homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
        influenced = affine[self.indices.to(device)] @ homogeneous[:, None, :, None]
        posed = (influenced * self.weights.to(device=device, dtype=dtype)[..., None, None]).sum(dim=1)
        return posed.squeeze(-1).detach().cpu().numpy()

    def skin(
        self,
        joint_rotmat: torch.Tensor,
        joint_pos: torch.Tensor,
        rot_is_global: bool = True,
    ) -> np.ndarray:
        if not rot_is_global:
            joint_rotmat, _, _ = self.skeleton_input.fk(joint_rotmat, joint_pos)
        return self.pose(joint_rotmat, joint_pos)


__all__ = ["SOMASkin"]
