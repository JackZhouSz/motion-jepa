"""Globally weighted reconstruction errors in feature and motion coordinates."""

from __future__ import annotations

import math

import torch

from motion_rep import MotionJEPAMotionRep


FEATURE_BLOCKS = {
    "root_position": MotionJEPAMotionRep.ROOT_POSITION,
    "root_heading": MotionJEPAMotionRep.ROOT_HEADING,
    "local_positions": MotionJEPAMotionRep.LOCAL_POSITIONS,
    "global_rotations": MotionJEPAMotionRep.GLOBAL_ROTATIONS,
    "velocities": MotionJEPAMotionRep.VELOCITIES,
    "foot_contacts": MotionJEPAMotionRep.FOOT_CONTACTS,
}


def _validated_reconstruction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.ndim != 3 or prediction.shape != target.shape or min(prediction.shape) <= 0:
        raise ValueError("Prediction and target must have equal nonempty shape [B,T,C]")
    if prediction.device != target.device:
        raise ValueError("Prediction and target must be on the same device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Prediction and target must have floating-point dtypes")
    active = torch.as_tensor(valid_frames, device=prediction.device, dtype=torch.bool)
    if active.shape != prediction.shape[:2]:
        raise ValueError(f"valid_frames must have shape {tuple(prediction.shape[:2])}")
    if not active.any():
        raise ValueError("Reconstruction requires at least one valid frame")
    prediction = prediction.float().masked_fill(~active.unsqueeze(-1), 0)
    target = target.float().masked_fill(~active.unsqueeze(-1), 0)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Valid reconstruction frames must be finite")
    return prediction, target, active


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_frames: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over valid frame/channel elements, including tail frames."""
    prediction, target, active = _validated_reconstruction(prediction, target, valid_frames)
    return (prediction - target).square().sum() / (active.sum() * prediction.shape[-1])


class ReconstructionMetrics:
    """Accumulate sums and counts rather than averaging unequal batch means."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, fps: int) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        self.std = torch.as_tensor(std, dtype=torch.float32).detach().clone()
        if self.mean.shape != (366,) or self.std.shape != (366,):
            raise ValueError("Physical reconstruction metrics require 366-dimensional MotionJEPA statistics")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Motion statistics must be finite with positive standard deviations")
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.representation = MotionJEPAMotionRep(fps=fps)
        self.frame_count = 0
        self.sums = {"mse": 0.0, **{f"{name}_mse": 0.0 for name in FEATURE_BLOCKS}}
        self.sums.update(mpjpe_mm=0.0, root_error_mm=0.0, rotation_error_deg=0.0)
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    @torch.no_grad()
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> None:
        prediction, target, active = _validated_reconstruction(prediction, target, valid_frames)
        if prediction.shape[-1] != 366:
            raise ValueError("Physical reconstruction metrics support only motion_jepa_366_v1")
        prediction, target = prediction[active], target[active]
        errors = (prediction - target).square()
        totals = {"mse": errors.double().sum()}
        totals.update({f"{name}_mse": errors[:, block].double().sum() for name, block in FEATURE_BLOCKS.items()})

        mean, std = self.mean.to(prediction.device), self.std.to(prediction.device)
        raw_prediction, raw_target = prediction * std + mean, target * std + mean
        self.representation.skeleton.to(prediction.device)
        # Geometry must retain float32 accuracy even when callers evaluate the
        # decoder inside an autocast context.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            decoded_prediction = self.representation.inverse(raw_prediction)
            decoded_target = self.representation.inverse(raw_target)
            relative_rotation = (
                decoded_prediction["global_rot_mats"].transpose(-1, -2)
                @ decoded_target["global_rot_mats"]
            )
        position_error = torch.linalg.vector_norm(
            decoded_prediction["posed_joints"] - decoded_target["posed_joints"], dim=-1
        )
        root_error = torch.linalg.vector_norm(raw_prediction[:, :3] - raw_target[:, :3], dim=-1)
        skew = torch.stack(
            (
                relative_rotation[..., 2, 1] - relative_rotation[..., 1, 2],
                relative_rotation[..., 0, 2] - relative_rotation[..., 2, 0],
                relative_rotation[..., 1, 0] - relative_rotation[..., 0, 1],
            ),
            dim=-1,
        )
        sin_angle = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
        cos_angle = 0.5 * (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
        rotation_error = torch.atan2(sin_angle, cos_angle) * (180.0 / math.pi)
        totals.update(
            mpjpe_mm=position_error.double().sum() * 1000.0,
            root_error_mm=root_error.double().sum() * 1000.0,
            rotation_error_deg=rotation_error.double().sum(),
        )
        predicted_contacts = raw_prediction[:, MotionJEPAMotionRep.FOOT_CONTACTS] > 0.5
        target_contacts = raw_target[:, MotionJEPAMotionRep.FOOT_CONTACTS] > 0.5
        totals.update(
            true_positive=(predicted_contacts & target_contacts).sum(),
            false_positive=(predicted_contacts & ~target_contacts).sum(),
            false_negative=(~predicted_contacts & target_contacts).sum(),
        )
        values = torch.stack([value.double() for value in totals.values()]).cpu().tolist()
        for name, value in zip(totals, values):
            if name in self.sums:
                self.sums[name] += value
            else:
                setattr(self, name, getattr(self, name) + int(value))
        self.frame_count += len(prediction)

    def compute(self) -> dict[str, float]:
        if self.frame_count == 0:
            raise ValueError("No valid reconstruction frames have been evaluated")
        result = {"mse": self.sums["mse"] / (self.frame_count * 366)}
        for name, block in FEATURE_BLOCKS.items():
            result[f"{name}_mse"] = self.sums[f"{name}_mse"] / (self.frame_count * (block.stop - block.start))
        result["mpjpe_mm"] = self.sums["mpjpe_mm"] / (self.frame_count * 30)
        result["root_error_mm"] = self.sums["root_error_mm"] / self.frame_count
        result["rotation_error_deg"] = self.sums["rotation_error_deg"] / (self.frame_count * 30)
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        result["contact_f1"] = 2 * self.true_positive / denominator if denominator else 1.0
        return result


__all__ = ["FEATURE_BLOCKS", "masked_mse", "ReconstructionMetrics"]
