# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small, self-contained BVH reader used by MotionJEPA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _parse_hierarchy(lines: list[str]) -> tuple[list[str], list[list[str]], int]:
    names: list[str] = []
    channels: list[list[str]] = []
    stack: list[int | None] = []
    pending: int | None = None
    motion_line = -1
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if fields[0] in {"ROOT", "JOINT"}:
            names.append(fields[1])
            channels.append([])
            pending = len(names) - 1
        elif fields[:2] == ["End", "Site"]:
            pending = None
        elif fields[0] == "{":
            stack.append(pending)
        elif fields[0] == "}":
            stack.pop()
        elif fields[0] == "CHANNELS":
            joint_index = stack[-1]
            if joint_index is None:
                raise ValueError("End Site cannot declare BVH channels.")
            count = int(fields[1])
            channels[joint_index] = fields[2 : 2 + count]
        elif fields[0] == "MOTION":
            motion_line = line_index
            break
    if motion_line < 0 or not names:
        raise ValueError("Invalid BVH: missing hierarchy or MOTION section.")
    return names, channels, motion_line


def parse_bvh_motion(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return local rotation matrices, root translation in metres, and source FPS."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    names, channels, motion_line = _parse_hierarchy(lines)

    frame_count = None
    frame_time = None
    data_start = None
    for line_index in range(motion_line + 1, len(lines)):
        fields = lines[line_index].strip().split()
        if not fields:
            continue
        if fields[0].startswith("Frames"):
            frame_count = int(fields[-1])
        elif len(fields) >= 3 and fields[0] == "Frame" and fields[1] == "Time:":
            frame_time = float(fields[2])
            data_start = line_index + 1
            break
    if frame_count is None or frame_time is None or data_start is None:
        raise ValueError(f"Invalid BVH motion header: {path}")

    values = np.fromstring("\n".join(lines[data_start:]), sep=" ", dtype=np.float32)
    channel_count = sum(len(item) for item in channels)
    if values.size != frame_count * channel_count:
        raise ValueError(
            f"BVH frame data size mismatch: got {values.size}, expected {frame_count * channel_count}"
        )
    frames = values.reshape(frame_count, channel_count)

    rotations: list[np.ndarray] = []
    root_translation = np.zeros((frame_count, 3), dtype=np.float32)
    offset = 0
    output_joint_index = 0
    for joint_index, joint_channels in enumerate(channels):
        joint_values = frames[:, offset : offset + len(joint_channels)]
        offset += len(joint_channels)
        if names[joint_index] == "Root":
            continue
        position_channels = [name for name in joint_channels if name.endswith("position")]
        rotation_channels = [name for name in joint_channels if name.endswith("rotation")]
        if output_joint_index == 0:
            for axis_index, axis in enumerate("XYZ"):
                channel_name = f"{axis}position"
                if channel_name in position_channels:
                    root_translation[:, axis_index] = joint_values[:, joint_channels.index(channel_name)]
        if len(rotation_channels) != 3:
            raise ValueError(f"Joint {names[joint_index]} does not have three rotation channels.")
        rotation_order = "".join(name[0] for name in rotation_channels)
        euler = np.stack(
            [joint_values[:, joint_channels.index(name)] for name in rotation_channels],
            axis=-1,
        )
        matrix = Rotation.from_euler(rotation_order, np.deg2rad(euler)).as_matrix()
        rotations.append(matrix)
        output_joint_index += 1

    local_rotations = torch.from_numpy(np.stack(rotations, axis=1))
    root_positions = torch.from_numpy(root_translation * 0.01)
    return local_rotations, root_positions, 1.0 / frame_time
