"""SOMA skeleton definitions used by MotionJEPA."""

import torch

from .base import SkeletonBase


class SOMASkeleton77(SkeletonBase):
    name = "somaskel77"
    metadata_key = "soma77"


class SOMASkeleton30(SkeletonBase):
    name = "somaskel30"
    metadata_key = "soma30"

    def from_soma77(self, local_rotations: torch.Tensor) -> torch.Tensor:
        return local_rotations[:, self.subset_indices.to(local_rotations.device)]

    def to_soma77(self, local_rotations: torch.Tensor) -> torch.Tensor:
        skeleton77 = SOMASkeleton77()
        relaxed = skeleton77.relaxed_hands.to(
            device=local_rotations.device, dtype=local_rotations.dtype
        )
        expanded = relaxed.expand(len(local_rotations), -1, -1, -1).clone()
        expanded[:, self.subset_indices.to(local_rotations.device)] = local_rotations
        return expanded

    def expand_output(
        self, output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        skeleton77 = SOMASkeleton77()
        local = self.to_soma77(output["local_rot_mats"])
        global_rotations, positions, _ = skeleton77.fk(
            local, output["root_positions"]
        )
        expanded = dict(output)
        expanded.update(
            local_rot_mats=local,
            global_rot_mats=global_rotations,
            posed_joints=positions,
        )
        if "foot_contacts" in output:
            contacts = output["foot_contacts"]
            expanded["foot_contacts"] = torch.cat(
                [
                    contacts[..., :2],
                    contacts[..., 1:2],
                    contacts[..., 2:4],
                    contacts[..., 3:4],
                ],
                dim=-1,
            )
        return expanded


__all__ = ["SOMASkeleton30", "SOMASkeleton77"]
