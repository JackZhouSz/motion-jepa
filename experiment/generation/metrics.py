"""Draw-aware reconstruction errors and conditional motion diversity."""

from __future__ import annotations

import torch

from experiment.prediction.metrics import FEATURE_BLOCKS, ReconstructionMetrics, masked_mse
from motion_rep import MotionJEPAMotionRep


def per_draw_mse(generated: torch.Tensor, target: torch.Tensor,
                 valid_frames: torch.Tensor) -> torch.Tensor:
    """Return [B,K] MSEs, reducing each complete draw over valid frames/channels."""
    if (generated.ndim != 4 or target.ndim != 3 or generated.shape[0] != target.shape[0]
            or generated.shape[2:] != target.shape[1:] or min(generated.shape) < 1):
        raise ValueError("Expected generated [B,K,T,C] and target [B,T,C]")
    if generated.device != target.device or not generated.is_floating_point() or not target.is_floating_point():
        raise ValueError("Generated motion and target must be floating point on the same device")
    active = torch.as_tensor(valid_frames, dtype=torch.bool, device=generated.device)
    if active.shape != target.shape[:2] or not active.any(dim=1).all():
        raise ValueError("Every draw must contain at least one valid frame")
    error = (generated.float() - target[:, None].float()).masked_fill(~active[:, None, :, None], 0)
    if not torch.isfinite(error).all():
        raise ValueError("Valid generated motion and targets must be finite")
    return error.square().sum(dim=(2, 3)) / (active.sum(dim=1)[:, None] * target.shape[-1])


def select_best_draw(generated: torch.Tensor, target: torch.Tensor,
                     valid_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose one entire draw per clip; never select independently by frame or metric."""
    selected = per_draw_mse(generated, target, valid_frames).argmin(dim=1)
    return generated[torch.arange(len(generated), device=generated.device), selected], selected


class DiversityMetrics:
    """Mean pairwise FK distance within each condition, excluding padding."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, fps: int):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        self.std = torch.as_tensor(std, dtype=torch.float32).detach().clone()
        if (self.mean.shape != (366,) or self.std.shape != (366,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError("Diversity requires finite 366-D train statistics with positive scales")
        self.representation = MotionJEPAMotionRep(fps=fps)
        self.valid_pair_frames = 0
        self.clip_pairs = 0
        self.root_sum = 0.
        self.relative_joint_sum = 0.

    @torch.no_grad()
    def update(self, generated: torch.Tensor, valid_frames: torch.Tensor) -> None:
        if generated.ndim != 4 or generated.shape[-1] != 366 or min(generated.shape) < 1:
            raise ValueError("Diversity expects generated motion [B,K,T,366]")
        active = torch.as_tensor(valid_frames, dtype=torch.bool, device=generated.device)
        if active.shape != (generated.shape[0], generated.shape[2]) or not active.any(dim=1).all():
            raise ValueError("Diversity requires valid frames for every clip")
        cleaned = generated.float().masked_fill(~active[:, None, :, None], 0)
        if not torch.isfinite(cleaned).all():
            raise ValueError("Valid diversity frames must be finite")
        draws = generated.shape[1]
        if draws == 1:
            return
        raw = cleaned * self.std.to(generated.device) + self.mean.to(generated.device)
        raw = raw.masked_fill(~active[:, None, :, None], 0)
        self.representation.skeleton.to(generated.device)
        with torch.autocast(device_type=generated.device.type, enabled=False):
            joints = self.representation.inverse(raw.flatten(0, 1))["posed_joints"]
        joints = joints.reshape(*generated.shape[:3], 30, 3)
        roots = joints[..., 0, :]
        relative = joints[..., 1:, :] - roots[..., None, :]
        count = int(active.sum())
        for left in range(draws):
            for right in range(left + 1, draws):
                root_error = torch.linalg.vector_norm(roots[:, left] - roots[:, right], dim=-1)
                joint_error = torch.linalg.vector_norm(relative[:, left] - relative[:, right], dim=-1)
                self.root_sum += float(root_error[active].double().sum())
                self.relative_joint_sum += float(joint_error[active].double().sum())
                self.valid_pair_frames += count
                self.clip_pairs += len(generated)

    def compute(self) -> dict:
        denominator = self.valid_pair_frames
        return {
            "root_pairwise_mm": self.root_sum * 1000. / denominator if denominator else 0.,
            "root_relative_nonroot_joint_pairwise_mm": (
                self.relative_joint_sum * 1000. / (denominator * 29) if denominator else 0.
            ),
            "clip_pairs": self.clip_pairs,
            "valid_pair_frames": denominator,
        }


class MultiSampleMetrics:
    """MC averages and a single whole-clip normalized-MSE oracle on the same bank."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, fps: int):
        self.mean, self.std, self.fps = mean, std, fps
        self.draw_metrics: list[ReconstructionMetrics] | None = None
        self.best_of_k = ReconstructionMetrics(mean, std, fps)
        self.diversity = DiversityMetrics(mean, std, fps)
        self.best_draw_counts: list[int] | None = None

    @torch.no_grad()
    def update(self, generated: torch.Tensor, target: torch.Tensor,
               valid_frames: torch.Tensor) -> None:
        best, indices = select_best_draw(generated, target, valid_frames)
        draws = generated.shape[1]
        if self.best_draw_counts is None:
            self.best_draw_counts = [0] * draws
            self.draw_metrics = [ReconstructionMetrics(self.mean, self.std, self.fps) for _ in range(draws)]
        if len(self.best_draw_counts) != draws:
            raise ValueError("All diagnostic batches must use the same number of draws")
        for draw in range(draws):
            self.draw_metrics[draw].update(generated[:, draw], target, valid_frames)
            self.best_draw_counts[draw] += int((indices == draw).sum())
        self.best_of_k.update(best, target, valid_frames)
        self.diversity.update(generated, valid_frames)

    def compute(self) -> dict:
        if self.draw_metrics is None:
            raise ValueError("No diagnostic motion draws have been evaluated")
        per_draw = [metric.compute() for metric in self.draw_metrics]
        monte_carlo = {name: sum(draw[name] for draw in per_draw) / len(per_draw) for name in per_draw[0]}
        return {"monte_carlo": monte_carlo,
                "best_of_k": self.best_of_k.compute(),
                "diversity": self.diversity.compute(),
                "best_draw_counts": self.best_draw_counts}


__all__ = ["FEATURE_BLOCKS", "ReconstructionMetrics", "masked_mse", "per_draw_mse",
           "select_best_draw", "DiversityMetrics", "MultiSampleMetrics"]
