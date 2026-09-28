# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rotation, velocity, and feature-space helpers for MotionJEPA."""

from __future__ import annotations

import torch


def matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    x_raw, y_raw = cont6d[..., :3], cont6d[..., 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, y_raw, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def y_rotation(angle: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    return torch.stack((cos, zero, sin, zero, one, zero, -sin, zero, cos), dim=-1).reshape(
        angle.shape + (3, 3)
    )


def velocity(positions: torch.Tensor, fps: int) -> torch.Tensor:
    if positions.shape[1] <= 1:
        return torch.zeros_like(positions)
    difference = float(fps) * (positions[:, 1:] - positions[:, :-1])
    return torch.cat([difference, difference[:, -1:]], dim=1)
