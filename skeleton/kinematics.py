"""Forward-kinematics utilities for MotionJEPA skeletons."""

import torch


def local_rots_to_global_rots(
    local_rotations: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    global_rotations = torch.empty_like(local_rotations)
    global_rotations[:, 0] = local_rotations[:, 0]
    for joint in range(1, len(parents)):
        global_rotations[:, joint] = (
            global_rotations[:, int(parents[joint])] @ local_rotations[:, joint]
        )
    return global_rotations


def global_rots_to_local_rots(
    global_rotations: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    local_rotations = torch.empty_like(global_rotations)
    local_rotations[:, 0] = global_rotations[:, 0]
    for joint in range(1, len(parents)):
        parent_rotation = global_rotations[:, int(parents[joint])]
        local_rotations[:, joint] = (
            parent_rotation.transpose(-2, -1) @ global_rotations[:, joint]
        )
    return local_rotations


def fk(
    local_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    neutral_joints: torch.Tensor,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return global rotations, posed joints, and root-relative joints."""
    unbatched = local_rotations.ndim == 4
    if unbatched:
        local_rotations = local_rotations.unsqueeze(0)
        root_positions = root_positions.unsqueeze(0)
    batch, frames, num_joints = local_rotations.shape[:3]
    rotations = local_rotations.reshape(-1, num_joints, 3, 3)
    roots = root_positions.reshape(-1, 3)
    neutral = neutral_joints.to(device=rotations.device, dtype=rotations.dtype)
    relative = neutral.clone()
    relative[1:] -= neutral[parents[1:]]
    global_rotations = local_rots_to_global_rots(rotations, parents)
    positions = torch.empty(
        len(rotations), num_joints, 3, device=rotations.device, dtype=rotations.dtype
    )
    positions[:, 0] = 0
    for joint in range(1, num_joints):
        parent = int(parents[joint])
        positions[:, joint] = positions[:, parent] + (
            global_rotations[:, parent] @ relative[joint, :, None]
        ).squeeze(-1)
    posed = positions + roots[:, None]
    shape = (batch, frames)
    result = (
        global_rotations.reshape(*shape, num_joints, 3, 3),
        posed.reshape(*shape, num_joints, 3),
        positions.reshape(*shape, num_joints, 3),
    )
    return tuple(value[0] for value in result) if unbatched else result


__all__ = ["fk", "global_rots_to_local_rots", "local_rots_to_global_rots"]
