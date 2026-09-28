"""Online collapse and held-out JEPA diagnostics for MotionJEPA pretraining."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset.motion_dataset import MotionDataset
from mask.utils import gather_grid_masks


_EPS = 1.0e-12
_DISTANCE_BUCKETS = ("1", "2-3", "4-7", "8+", "no-context")


def effective_rank(values: torch.Tensor) -> float:
    """RankMe effective rank from the singular values of an uncentered matrix."""
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"RankMe expects a non-empty matrix, got {matrix.shape}")
    singular = torch.linalg.svdvals(matrix)
    total = singular.sum()
    if float(total) <= _EPS:
        return 1.0
    probabilities = singular / total
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def mean_off_diagonal_cosine(values: torch.Tensor) -> float:
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or len(matrix) < 2:
        return 0.0
    normalized = F.normalize(matrix, dim=-1, eps=_EPS)
    total = normalized.sum(dim=0).square().sum() - normalized.square().sum()
    return float(total / (len(matrix) * (len(matrix) - 1)))


def covariance_metrics(values: torch.Tensor) -> dict[str, float]:
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or len(matrix) < 2:
        return {
            "largest_eigenvalue_ratio": 1.0,
            "effective_rank": 1.0,
            "condition_number": 1.0,
            "off_diagonal_abs_mean": 0.0,
        }
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(matrix) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) <= _EPS:
        largest_ratio = rank = condition = 1.0
    else:
        probabilities = eigenvalues / total
        positive = probabilities[probabilities > 0]
        largest_ratio = float(probabilities[-1])
        rank = float(torch.exp(-(positive * positive.log()).sum()))
        floor = max(_EPS, float(eigenvalues[-1]) * _EPS)
        condition = float(eigenvalues[-1] / eigenvalues[0].clamp_min(floor))
    if covariance.shape[0] < 2:
        off_diagonal = 0.0
    else:
        off_diagonal = float(
            (covariance.abs().sum() - covariance.diagonal().abs().sum())
            / (covariance.numel() - covariance.shape[0])
        )
    return {
        "largest_eigenvalue_ratio": largest_ratio,
        "effective_rank": rank,
        "condition_number": condition,
        "off_diagonal_abs_mean": off_diagonal,
    }


def feature_matrix_metrics(values: torch.Tensor) -> dict[str, float]:
    matrix = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    return {
        "rankme": effective_rank(matrix),
        "mean_std": float(matrix.std(dim=0, correction=0).mean()),
        "mean_off_diagonal_cosine": mean_off_diagonal_cosine(matrix),
        "covariance": covariance_metrics(matrix),
    }


def prediction_gain(mse: float, baseline_mse: float) -> float:
    if baseline_mse <= _EPS:
        return 0.0 if mse <= _EPS else 1.0 - mse / _EPS
    return 1.0 - mse / baseline_mse


@dataclass
class _PredictionAccumulator:
    count: int = 0
    squared_error: float = 0.0
    smooth_l1: float = 0.0
    cosine: float = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        selected: torch.Tensor | None = None,
    ) -> None:
        prediction = prediction.detach().float().cpu()
        target = target.detach().float().cpu()
        if selected is not None:
            selected = selected.to(dtype=torch.bool, device="cpu")
            prediction = prediction[selected]
            target = target[selected]
        if prediction.numel() == 0:
            return
        if prediction.ndim != 2:
            prediction = prediction.reshape(-1, prediction.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        self.count += len(prediction)
        self.squared_error += float((prediction - target).square().mean(dim=-1).sum())
        self.smooth_l1 += float(
            F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1).sum()
        )
        self.cosine += float(F.cosine_similarity(prediction, target, dim=-1).sum())

    def summary(self) -> dict[str, float]:
        denominator = max(1, self.count)
        return {
            "count": int(self.count),
            "mse": self.squared_error / denominator,
            "smooth_l1": self.smooth_l1 / denominator,
            "cosine_similarity": self.cosine / denominator,
        }


class _TargetMoments:
    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.square_sum = 0.0

    def update(self, target: torch.Tensor, selected: torch.Tensor | None = None) -> None:
        values = target.detach().to(device="cpu", dtype=torch.float64)
        if selected is not None:
            values = values[selected.to(dtype=torch.bool, device="cpu")]
        values = values.reshape(-1, values.shape[-1])
        if not len(values):
            return
        self.count += len(values)
        batch_sum = values.sum(dim=0)
        self.sum = batch_sum if self.sum is None else self.sum + batch_sum
        self.square_sum += float(values.square().sum())

    def baseline_mse(self) -> float:
        if self.count == 0 or self.sum is None:
            return 0.0
        total_variation = self.square_sum - float(self.sum.square().sum()) / self.count
        return max(0.0, total_variation / (self.count * len(self.sum)))


def _distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "no-context"
    if distance <= 1:
        return "1"
    if distance <= 3:
        return "2-3"
    if distance <= 7:
        return "4-7"
    return "8+"


def _target_metadata(
    masks_enc: list[torch.Tensor],
    masks_pred: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if len(masks_enc) != 1:
        raise ValueError("Online JEPA metrics currently require one encoder mask")
    context = masks_enc[0].to(dtype=torch.bool, device="cpu")
    mask_ids, spatial_ids, distance_buckets = [], [], []
    for mask_index, target in enumerate(masks_pred):
        target = target.to(dtype=torch.bool, device="cpu")
        for batch_index in range(len(target)):
            coordinates = torch.nonzero(target[batch_index], as_tuple=False)
            mask_ids.extend([mask_index] * len(coordinates))
            spatial_ids.extend(coordinates[:, 1].tolist())
            for frame, spatial in coordinates.tolist():
                visible = torch.nonzero(
                    context[batch_index, :, spatial], as_tuple=False
                ).flatten()
                distance = (
                    None
                    if not len(visible)
                    else int((visible - frame).abs().min())
                )
                distance_buckets.append(_distance_bucket(distance))
    return (
        torch.tensor(mask_ids, dtype=torch.long),
        torch.tensor(spatial_ids, dtype=torch.long),
        distance_buckets,
    )


class OnlineRepresentationMetrics:
    """Evaluate a fixed unlabeled subset without retaining token grids in memory."""

    def __init__(
        self,
        training_config: dict[str, Any],
        metric_config: dict[str, Any],
        *,
        device: torch.device,
        collator,
    ) -> None:
        self.device = device
        self.batch_size = int(metric_config.get("batch_size", 256))
        self.num_workers = int(metric_config.get("num_workers", 0))
        self.num_samples = int(metric_config.get("num_samples", 1024))
        self.seed = int(metric_config.get("seed", 0))
        if min(self.batch_size, self.num_samples) <= 0 or self.num_workers < 0:
            raise ValueError("online_metrics batch/sample counts must be positive")
        data = training_config["data"]
        root = Path(str(data["root_path"])).expanduser()
        self.dataset = MotionDataset(
            root_path=root,
            meta_files=[str(metric_config.get("meta_file", "val.txt"))],
            num_frames=int(data["num_frames"]),
            fps=int(data["fps"]),
            motion_dim=int(data["motion_dim"]),
            normalize=bool(data.get("normalize", False)),
            stats_path=data.get("stats_path"),
        )
        generator = torch.Generator().manual_seed(self.seed)
        count = min(self.num_samples, len(self.dataset))
        self.indices = torch.randperm(len(self.dataset), generator=generator)[:count].tolist()
        self.collator = collator
        self.initial_collator_state = collator.state_dict()
        self.initial_collator_state["counter"] = self.seed - 1
        self.use_bfloat16 = bool(training_config["meta"].get("use_bfloat16", False))

    def _loader(self) -> DataLoader:
        self.collator.load_state_dict(self.initial_collator_state)
        return DataLoader(
            Subset(self.dataset, self.indices),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=self.collator,
        )

    @torch.no_grad()
    def evaluate(self, encoder, predictor) -> dict[str, Any]:
        if encoder.training or predictor.training:
            raise ValueError("Online representation metrics require eval-mode models")
        spatial_names = tuple(encoder.token_layout.spatial_token_names or ())
        if not spatial_names or spatial_names[0] != "trajectory":
            raise ValueError("Online representation metrics require a trajectory-first layout")

        pooled: dict[str, list[torch.Tensor]] = defaultdict(list)
        variation = defaultdict(float)
        variation_count = 0
        overall = _PredictionAccumulator()
        by_mask = defaultdict(_PredictionAccumulator)
        by_spatial = defaultdict(_PredictionAccumulator)
        by_distance = defaultdict(_PredictionAccumulator)
        no_trajectory_body = _PredictionAccumulator()
        target_moments = _TargetMoments()
        body_target_moments = _TargetMoments()

        amp = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.use_bfloat16 and self.device.type in {"cpu", "cuda"}
            else torch.autocast(device_type=self.device.type, enabled=False)
        )
        with amp:
            for batch, masks_enc, masks_pred in self._loader():
                motion = batch[0].to(self.device, dtype=torch.float32)
                fps = batch[1].to(self.device, dtype=torch.float32)
                lengths = batch[2].to(self.device)
                valid_frames = (
                    torch.arange(motion.shape[1], device=self.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                token_lengths = encoder.token_layout.valid_token_lengths(lengths).cpu()
                full = encoder(motion, fps, valid_frames=valid_frames)
                normalized_target = F.layer_norm(full, (full.shape[-1],))

                for sample_index, token_length in enumerate(token_lengths.tolist()):
                    values = full[sample_index, :token_length].float().cpu()
                    trajectory = values[:, 0]
                    body = values[:, 1:]
                    pooled["trajectory"].append(trajectory.mean(dim=0))
                    pooled["body"].append(body.mean(dim=(0, 1)))
                    pooled["combined"].append(values.mean(dim=(0, 1)))
                    variation["body_temporal_std"] += float(
                        body.mean(dim=1).std(dim=0, correction=0).mean()
                    )
                    variation["body_joint_std"] += float(
                        body.mean(dim=0).std(dim=0, correction=0).mean()
                    )
                    variation["trajectory_temporal_std"] += float(
                        trajectory.std(dim=0, correction=0).mean()
                    )
                    variation_count += 1

                device_enc = [mask.to(self.device) for mask in masks_enc]
                device_pred = [mask.to(self.device) for mask in masks_pred]
                target = gather_grid_masks(normalized_target, device_pred)
                context = encoder(motion, fps, device_enc, valid_frames=valid_frames)
                prediction = predictor(context, fps, device_enc, device_pred)
                flat_prediction = prediction.reshape(-1, prediction.shape[-1])
                flat_target = target.reshape(-1, target.shape[-1])
                mask_ids, spatial_ids, distance_buckets = _target_metadata(
                    masks_enc, masks_pred
                )
                overall.update(flat_prediction, flat_target)
                target_moments.update(flat_target)
                body_selected = spatial_ids != 0
                body_target_moments.update(flat_target, body_selected)
                for mask_index in range(len(masks_pred)):
                    selected = mask_ids == mask_index
                    by_mask[mask_index].update(flat_prediction, flat_target, selected)
                for spatial_index, spatial_name in enumerate(spatial_names):
                    selected = spatial_ids == spatial_index
                    by_spatial[spatial_name].update(flat_prediction, flat_target, selected)
                for bucket in _DISTANCE_BUCKETS:
                    selected = torch.tensor(
                        [value == bucket for value in distance_buckets], dtype=torch.bool
                    )
                    by_distance[bucket].update(flat_prediction, flat_target, selected)

                hidden_trajectory = [mask.clone() for mask in masks_enc]
                for mask in hidden_trajectory:
                    mask[..., 0] = False
                hidden_device = [mask.to(self.device) for mask in hidden_trajectory]
                hidden_context = encoder(
                    motion, fps, hidden_device, valid_frames=valid_frames
                )
                hidden_prediction = predictor(
                    hidden_context, fps, hidden_device, device_pred
                ).reshape_as(flat_prediction)
                no_trajectory_body.update(
                    hidden_prediction, flat_target, body_selected
                )

        representation = {
            name: feature_matrix_metrics(torch.stack(values))
            for name, values in pooled.items()
        }
        representation["variation"] = {
            name: value / max(1, variation_count)
            for name, value in variation.items()
        }
        heldout = overall.summary()
        baseline_mse = target_moments.baseline_mse()
        heldout["trivial_mean_mse"] = baseline_mse
        heldout["prediction_gain"] = prediction_gain(heldout["mse"], baseline_mse)
        heldout["by_mask"] = {
            str(index): accumulator.summary()
            for index, accumulator in sorted(by_mask.items())
        }
        heldout["by_spatial_token"] = {
            name: by_spatial[name].summary() for name in spatial_names
        }
        heldout["by_temporal_distance"] = {
            bucket: by_distance[bucket].summary() for bucket in _DISTANCE_BUCKETS
        }
        body_baseline = body_target_moments.baseline_mse()
        body_normal_mse = (
            sum(by_spatial[name].squared_error for name in spatial_names[1:])
            / max(1, sum(by_spatial[name].count for name in spatial_names[1:]))
        )
        body_normal_gain = prediction_gain(body_normal_mse, body_baseline)
        hidden_summary = no_trajectory_body.summary()
        hidden_gain = prediction_gain(hidden_summary["mse"], body_baseline)
        heldout["trajectory_ablation"] = {
            "body_trivial_mean_mse": body_baseline,
            "body_prediction_gain": body_normal_gain,
            "body_prediction_gain_without_trajectory": hidden_gain,
            "trajectory_reliance": body_normal_gain - hidden_gain,
            **{f"without_trajectory_{key}": value for key, value in hidden_summary.items()},
        }
        return {
            "num_samples": len(self.indices),
            "representation": representation,
            "heldout_jepa": heldout,
        }


__all__ = [
    "OnlineRepresentationMetrics",
    "covariance_metrics",
    "effective_rank",
    "feature_matrix_metrics",
    "mean_off_diagonal_cosine",
    "prediction_gain",
]
