"""Mask tensor helpers shared by MotionJEPA variants."""

from __future__ import annotations

import torch


def apply_index_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather 1D token indices and concatenate mask blocks along the batch axis."""
    gathered = []
    for mask in masks:
        index = mask.to(device=x.device, dtype=torch.long).unsqueeze(-1)
        gathered.append(torch.gather(x, 1, index.expand(-1, -1, x.shape[-1])))
    if not gathered:
        raise ValueError("At least one mask block is required")
    return torch.cat(gathered, dim=0)


def gather_grid(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Gather equal-count active cells from a ``[B,T,J,D]`` grid."""
    if x.ndim != 4 or active.shape != x.shape[:-1]:
        raise ValueError(
            f"Grid/mask shape mismatch: grid={tuple(x.shape)}, mask={tuple(active.shape)}"
        )
    active = active.to(device=x.device, dtype=torch.bool)
    counts = active.flatten(1).sum(dim=1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError(f"Every batch item must expose the same target count, got {counts.tolist()}")
    count = int(counts[0])
    if count == 0:
        raise ValueError("Target masks cannot be empty")
    return x[active].reshape(x.shape[0], count, x.shape[-1])


def gather_grid_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather 2D mask blocks and concatenate them along the batch axis."""
    if not masks:
        raise ValueError("At least one mask block is required")
    return torch.cat([gather_grid(x, mask) for mask in masks], dim=0)


def repeat_mask_blocks(x: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    """Repeat each mask-major batch block before advancing to the next block."""
    if repeat == 1:
        return x
    blocks = len(x) // batch_size
    return torch.cat(
        [
            torch.cat([x[index * batch_size : (index + 1) * batch_size]] * repeat, dim=0)
            for index in range(blocks)
        ],
        dim=0,
    )


__all__ = [
    "apply_index_masks",
    "gather_grid",
    "gather_grid_masks",
    "repeat_mask_blocks",
]
