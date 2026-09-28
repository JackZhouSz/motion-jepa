"""CNN and Transformer classification from raw motion or frozen JEPA tokens."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from .cnn import MotionCNNClassifier
from .dataset import (
    BabelLabelIndex,
    StyleTokenDataset,
    build_classification_datasets,
    classification_dataset_kind,
)
from .linear import RawMotionLinearClassifier
from .features import (
    Metrics,
    _atomic_json_save,
    _atomic_torch_save,
    _seed_all,
    _sha256_file,
    _torch_load_checkpoint,
    load_frozen_encoder,
    resolve_pretraining_stats,
    resolve_device,
)
from .transformer import MotionTransformerClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_FINDINGS_ROOT = (
    PROJECT_ROOT / "findings/000-100style-classification/classifiers"
)
MODELS = ("cnn", "transformer")
AVAILABLE_MODELS = (*MODELS, "linear")
METRIC_FIELDS = (
    "loss",
    "top1_accuracy",
    "macro_accuracy",
    "top5_accuracy",
)
MULTILABEL_METRIC_FIELDS = (
    "loss", "mean_average_precision", "top1_hit", "top1_label_row_accuracy", "top5_hit",
    "classes_with_positives", "classes_without_positives",
)
TOKEN_CACHE_FORMAT_VERSION = 1
CLASSIFIER_CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class PreparedInput:
    datasets: dict[str, Any]
    label_index: Any
    input_dim: int
    num_frames: int
    stats_root: Path
    input_source: str
    jepa_source: dict[str, Any] | None
    task: str = "single_label"
    dataset_name: str = "100style"


class MetricAccumulator:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.loss_sum = 0.0
        self.total = 0
        self.correct = 0
        self.top5_correct = 0
        self.class_total = torch.zeros(self.num_classes, dtype=torch.long)
        self.class_correct = torch.zeros(self.num_classes, dtype=torch.long)

    def update(
        self, logits: torch.Tensor, labels: torch.Tensor, loss_sum: float
    ) -> None:
        predictions = logits.detach().argmax(dim=1)
        labels = labels.detach()
        matches = predictions.eq(labels)
        self.loss_sum += float(loss_sum)
        self.total += len(labels)
        self.correct += int(matches.sum())
        topk = min(5, self.num_classes)
        self.top5_correct += int(
            logits.detach()
            .topk(topk, dim=1)
            .indices.eq(labels[:, None])
            .any(dim=1)
            .sum()
        )
        cpu_labels = labels.cpu()
        self.class_total += torch.bincount(cpu_labels, minlength=self.num_classes)
        self.class_correct += torch.bincount(
            cpu_labels[matches.cpu()], minlength=self.num_classes
        )

    def compute(self) -> Metrics:
        if self.total == 0:
            raise ValueError("Cannot compute metrics for an empty split")
        present = self.class_total > 0
        macro = (
            self.class_correct[present].float() / self.class_total[present].float()
        ).mean()
        return Metrics(
            loss=self.loss_sum / self.total,
            top1_accuracy=self.correct / self.total,
            macro_accuracy=float(macro),
            top5_accuracy=self.top5_correct / self.total,
        )


@dataclass(frozen=True)
class MultiLabelMetrics:
    loss: float
    mean_average_precision: float
    top1_hit: float
    top1_label_row_accuracy: float
    top5_hit: float
    classes_with_positives: int
    classes_without_positives: int


class MultiLabelMetricAccumulator:
    def __init__(
        self,
        num_classes: int,
        row_labels_by_sample: Mapping[str, Sequence[int]],
    ) -> None:
        self.num_classes = num_classes
        self.row_labels_by_sample = row_labels_by_sample
        self.loss_sum = 0.0
        self.scores: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.sample_ids: list[str] = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_sum: float,
        sample_ids: Sequence[str],
    ) -> None:
        if labels.shape != logits.shape or labels.ndim != 2:
            raise ValueError("Multi-label targets must match logits [batch, classes]")
        if len(sample_ids) != len(labels):
            raise ValueError("Sample IDs must match the multi-label batch size")
        self.loss_sum += float(loss_sum)
        self.scores.append(logits.detach().float().cpu())
        self.targets.append(labels.detach().float().cpu())
        self.sample_ids.extend(sample_ids)

    def compute(self) -> MultiLabelMetrics:
        if not self.scores:
            raise ValueError("Cannot compute metrics for an empty split")
        scores = torch.cat(self.scores)
        targets = torch.cat(self.targets)
        if not torch.all((targets == 0) | (targets == 1)) or not torch.all(targets.sum(1) > 0):
            raise ValueError("Multi-label targets must be nonempty binary vectors")
        predictions = scores.argmax(dim=1)
        top1 = targets.gather(1, predictions[:, None]).sum().item()
        top5 = targets.gather(1, scores.topk(min(5, self.num_classes), dim=1).indices)
        label_row_hits = 0
        label_row_count = 0
        for index, sample_id in enumerate(self.sample_ids):
            row_labels = self.row_labels_by_sample[sample_id]
            if not row_labels or any(targets[index, label] != 1 for label in row_labels):
                raise ValueError(f"Label rows do not match multi-hot target: {sample_id}")
            label_row_hits += sum(label == predictions[index].item() for label in row_labels)
            label_row_count += len(row_labels)
        average_precisions = []
        for class_id in range(self.num_classes):
            positives = targets[:, class_id]
            positive_count = int(positives.sum())
            if positive_count == 0:
                continue
            order = torch.argsort(scores[:, class_id], descending=True, stable=True)
            ranked = positives[order]
            precision = ranked.cumsum(0) / torch.arange(1, len(ranked) + 1)
            average_precisions.append(float((precision * ranked).sum() / positive_count))
        return MultiLabelMetrics(
            loss=self.loss_sum / len(scores),
            mean_average_precision=sum(average_precisions) / len(average_precisions),
            top1_hit=float(top1 / len(scores)),
            top1_label_row_accuracy=label_row_hits / label_row_count,
            top5_hit=float(top5.any(dim=1).float().mean()),
            classes_with_positives=len(average_precisions),
            classes_without_positives=self.num_classes - len(average_precisions),
        )


def make_classifier(
    model_name: str,
    *,
    motion_dim: int | None = None,
    input_dim: int | None = None,
    num_frames: int,
    num_classes: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if motion_dim is not None and input_dim is not None and motion_dim != input_dim:
        raise ValueError("motion_dim and input_dim must match when both are provided")
    resolved_input_dim = int(
        input_dim if input_dim is not None else motion_dim if motion_dim is not None else 366
    )
    if model_name == "cnn":
        config: dict[str, Any] = {
            "name": "MotionCNNClassifier",
            "input_dim": resolved_input_dim,
            "num_classes": num_classes,
            "widths": [128, 192, 256],
            "blocks_per_stage": 1,
            "dropout": 0.1,
        }
        model = MotionCNNClassifier(
            input_dim=resolved_input_dim,
            num_classes=num_classes,
            widths=tuple(config["widths"]),
            blocks_per_stage=config["blocks_per_stage"],
            dropout=config["dropout"],
        )
    elif model_name == "transformer":
        config = {
            "name": "MotionTransformerClassifier",
            "input_dim": resolved_input_dim,
            "num_frames": num_frames,
            "num_classes": num_classes,
            "embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "drop_path_rate": 0.1,
        }
        model = MotionTransformerClassifier(
            **{key: value for key, value in config.items() if key != "name"}
        )
    elif model_name == "linear":
        config = {
            "name": "RawMotionLinearClassifier",
            "input_dim": resolved_input_dim,
            "num_frames": num_frames,
            "num_classes": num_classes,
            "pooling": "valid_frame_mean",
        }
        model = RawMotionLinearClassifier(
            **{key: value for key, value in config.items() if key != "name"}
        )
    else:
        raise ValueError(f"Unknown classifier model: {model_name}")
    return model, config


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def _restore_rng_state(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


def _lr_factor(
    epoch: int, *, epochs: int, warmup_epochs: int, final_factor: float
) -> float:
    if warmup_epochs and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    cosine_epochs = epochs - warmup_epochs
    if cosine_epochs <= 1:
        return final_factor
    progress = (epoch - warmup_epochs) / (cosine_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    return final_factor + (1.0 - final_factor) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _amp_context(device: torch.device, use_bfloat16: bool):
    if device.type == "cuda" and use_bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _valid_frames(motion: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    return (
        torch.arange(motion.shape[1], device=motion.device).unsqueeze(0)
        < length.unsqueeze(1)
    )


def _positive_class_weights(targets: torch.Tensor, cap: float) -> torch.Tensor:
    """Weight train positives by capped square-root inverse frequency."""
    if targets.ndim != 2 or not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Positive weights require a binary [samples, classes] target matrix")
    positives = targets.float().sum(dim=0)
    negatives = len(targets) - positives
    weights = torch.sqrt(negatives / positives.clamp_min(1)).clamp(min=1, max=cap)
    return torch.where(positives > 0, weights, torch.ones_like(weights))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
    gradient_clip: float,
    task: str = "single_label",
    pos_weight: torch.Tensor | None = None,
    row_labels_by_sample: Mapping[str, Sequence[int]] | None = None,
) -> Metrics | MultiLabelMetrics:
    model.train()
    if task == "multilabel" and row_labels_by_sample is None:
        raise ValueError("BABEL label rows are required for multi-label metrics")
    metrics = (
        MultiLabelMetricAccumulator(num_classes, row_labels_by_sample)
        if task == "multilabel" else MetricAccumulator(num_classes)
    )
    for motion, _, length, labels, sample_ids in loader:
        motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
        length = length.to(device=device, non_blocking=True)
        labels = labels.to(
            device=device, dtype=torch.float32 if task == "multilabel" else torch.long,
            non_blocking=True,
        )
        active = _valid_frames(motion, length)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, use_bfloat16):
            logits = model(motion, active)
            loss = (
                F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                if task == "multilabel" else F.cross_entropy(logits, labels)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if task == "multilabel":
            metrics.update(logits, labels, float(loss.detach()) * len(labels), sample_ids)
        else:
            metrics.update(logits, labels, float(loss.detach()) * len(labels))
    return metrics.compute()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    use_bfloat16: bool,
    task: str = "single_label",
    pos_weight: torch.Tensor | None = None,
    row_labels_by_sample: Mapping[str, Sequence[int]] | None = None,
) -> Metrics | MultiLabelMetrics:
    model.eval()
    if task == "multilabel" and row_labels_by_sample is None:
        raise ValueError("BABEL label rows are required for multi-label metrics")
    metrics = (
        MultiLabelMetricAccumulator(num_classes, row_labels_by_sample)
        if task == "multilabel" else MetricAccumulator(num_classes)
    )
    with torch.inference_mode():
        for motion, _, length, labels, sample_ids in loader:
            motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
            length = length.to(device=device, non_blocking=True)
            labels = labels.to(
                device=device, dtype=torch.float32 if task == "multilabel" else torch.long,
                non_blocking=True,
            )
            with _amp_context(device, use_bfloat16):
                logits = model(motion, _valid_frames(motion, length))
                loss_sum = (
                    F.binary_cross_entropy_with_logits(
                        logits, labels, pos_weight=pos_weight, reduction="sum"
                    )
                    / num_classes
                    if task == "multilabel" else F.cross_entropy(logits, labels, reduction="sum")
                )
            if task == "multilabel":
                metrics.update(logits, labels, float(loss_sum), sample_ids)
            else:
                metrics.update(logits, labels, float(loss_sum))
    return metrics.compute()


def _read_dataset_meta(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return {
        "num_frames": int(metadata["num_frames"]),
        "fps": int(metadata["fps"]),
        "motion_dim": int(metadata["motion_dim"]),
    }


def _argument(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _token_cache_metadata(
    *,
    split: str,
    dataset_root: Path,
    stats_root: Path,
    checkpoint_path: Path,
    checkpoint_key: str,
    model_info: dict[str, Any],
    class_names: list[str],
    task: str = "single_label",
) -> dict[str, Any]:
    metadata = {
        "format_version": TOKEN_CACHE_FORMAT_VERSION,
        "kind": "motion_jepa_frame_tokens",
        "split": split,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "dataset_index_sha256": _sha256_file(dataset_root / "index.json"),
        "stats_mean_sha256": _sha256_file(stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(stats_root / "std.npy"),
        "model_name": model_info["model_name"],
        "num_frames": model_info["num_frames"],
        "motion_dim": model_info["motion_dim"],
        "fps": model_info["fps"],
        "feature_dim": model_info["feature_dim"],
        "token_num_frames": model_info["token_num_frames"],
        "temporal_patch_size": model_info["temporal_patch_size"],
        "patchified": model_info["patchified"],
        "layout_kind": model_info["kind"],
        "dtype": "bfloat16",
        "class_names": class_names,
    }
    if task == "multilabel":
        metadata["task"] = task
        metadata["label_format"] = "multi_hot_float32"
    for key in (
        "token_num_joints", "spatial_grouping", "spatial_pooling",
        "spatial_token_names", "trajectory_token_index", "body_token_offset",
        "trajectory_fields",
    ):
        if key in model_info:
            metadata[key] = model_info[key]
    if model_info["kind"] == "2d":
        metadata["spatial_token_reduction"] = "group_mean"
    return metadata


def _validate_token_cache(
    payload: dict[str, Any],
    expected_metadata: dict[str, Any],
    *,
    label_index: Any,
) -> None:
    if payload.get("metadata") != expected_metadata:
        raise ValueError(
            "JEPA token cache metadata is stale; rerun with --recompute-features"
        )
    dataset = StyleTokenDataset(
        payload,
        label_index=label_index,
        fps=int(expected_metadata["fps"]),
    )
    expected_shape = (
        int(expected_metadata["token_num_frames"]),
        int(expected_metadata["feature_dim"]),
    )
    if dataset.features.shape[1:] != expected_shape:
        raise ValueError(
            f"Token cache feature shape must end in {expected_shape}, "
            f"got {tuple(dataset.features.shape)}"
        )


def _extract_token_features(
    encoder: nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feature_batches: list[torch.Tensor] = []
    length_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for motion, fps, length, labels, ids in tqdm(
            loader, desc="Extract JEPA tokens"
        ):
            motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
            fps = fps.to(device=device, dtype=torch.float32, non_blocking=True)
            length_device = length.to(device=device, dtype=torch.long, non_blocking=True)
            active = _valid_frames(motion, length_device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                encoded = encoder(motion, fps, valid_frames=active)
            if encoded.ndim == 4:
                encoded = encoded.mean(dim=2)
            if encoded.ndim != 3:
                raise ValueError(
                    "JEPA token classifiers require encoder output [B,T,D] or [B,T,J,D], "
                    f"got {tuple(encoded.shape)}"
                )
            token_active = encoder.token_layout.valid_token_mask(active)
            encoded = encoded * token_active.unsqueeze(-1).to(dtype=encoded.dtype)
            feature_batches.append(encoded.to(dtype=torch.bfloat16).cpu())
            token_lengths = encoder.token_layout.valid_token_lengths(length_device)
            length_batches.append(token_lengths.to(dtype=torch.long).cpu())
            label_batches.append(labels.cpu())
            sample_ids.extend(list(ids))
    if not feature_batches:
        raise ValueError("Cannot extract JEPA tokens from an empty split")
    return {
        "features": torch.cat(feature_batches),
        "lengths": torch.cat(length_batches),
        "labels": torch.cat(label_batches),
        "sample_ids": sample_ids,
    }


def _load_or_extract_token_split(
    *,
    cache_path: Path,
    metadata: dict[str, Any],
    label_index: Any,
    dataset: Any,
    encoder: nn.Module,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
    recompute: bool,
) -> dict[str, Any]:
    if cache_path.is_file() and not recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        _validate_token_cache(payload, metadata, label_index=label_index)
        return payload
    payload = _extract_token_features(
        encoder,
        dataset,
        device=device,
        batch_size=feature_batch_size,
        num_workers=num_workers,
    )
    payload["metadata"] = metadata
    _validate_token_cache(payload, metadata, label_index=label_index)
    _atomic_torch_save(payload, cache_path)
    return payload


def _prepare_input(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> PreparedInput:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    dataset_info = _read_dataset_meta(dataset_root)
    task, dataset_name = classification_dataset_kind(dataset_root)
    input_source = str(_argument(args, "input_source", "raw"))
    if input_source == "raw":
        stats_root = dataset_root / "stats"
        datasets, label_index = build_classification_datasets(
            dataset_root,
            num_frames=dataset_info["num_frames"],
            fps=dataset_info["fps"],
            motion_dim=dataset_info["motion_dim"],
            stats_root=stats_root,
        )
        return PreparedInput(
            datasets=datasets,
            label_index=label_index,
            input_dim=dataset_info["motion_dim"],
            num_frames=dataset_info["num_frames"],
            stats_root=stats_root.resolve(),
            input_source="raw",
            jepa_source=None,
            task=task,
            dataset_name=dataset_name,
        )
    if input_source != "jepa":
        raise ValueError(f"Unknown input source: {input_source}")
    checkpoint_value = _argument(args, "jepa_checkpoint", None)
    if checkpoint_value is None:
        raise ValueError("--jepa-checkpoint is required for --input-source jepa")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"JEPA checkpoint does not exist: {checkpoint_path}")
    checkpoint_key = str(_argument(args, "checkpoint_key", "target_encoder"))
    encoder, config, model_info = load_frozen_encoder(
        checkpoint_path, checkpoint_key, device
    )
    for key in ("num_frames", "motion_dim", "fps"):
        if int(model_info[key]) != int(dataset_info[key]):
            raise ValueError(
                f"JEPA checkpoint {key}={model_info[key]} does not match "
                f"dataset {key}={dataset_info[key]}"
            )
    stats_value = _argument(args, "stats_path", None)
    stats_root = resolve_pretraining_stats(
        config, Path(stats_value) if stats_value is not None else None
    )
    motion_datasets, label_index = build_classification_datasets(
        dataset_root,
        num_frames=model_info["num_frames"],
        fps=model_info["fps"],
        motion_dim=model_info["motion_dim"],
        stats_root=stats_root,
    )
    cache_value = _argument(args, "feature_cache_root", None)
    default_cache_root = checkpoint_path.parent / "linear-probe/token-features"
    if task == "multilabel":
        default_cache_root /= dataset_name
    cache_root = (
        Path(cache_value).expanduser().resolve()
        if cache_value is not None else default_cache_root
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    token_datasets: dict[str, StyleTokenDataset] = {}
    feature_batch_size = int(_argument(args, "feature_batch_size", 256))
    class_names = list(label_index.class_names)
    for split in ("train", "val", "test"):
        metadata = _token_cache_metadata(
            split=split,
            dataset_root=dataset_root,
            stats_root=stats_root,
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
            model_info=model_info,
            class_names=class_names,
            task=task,
        )
        if len(motion_datasets[split]) == 0:
            payload = {
                "features": torch.empty(
                    (0, model_info["token_num_frames"], model_info["feature_dim"]),
                    dtype=torch.bfloat16,
                ),
                "lengths": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(
                    (0, label_index.num_classes), dtype=torch.float32
                ) if task == "multilabel" else torch.empty(0, dtype=torch.long),
                "sample_ids": [],
                "metadata": metadata,
            }
        else:
            payload = _load_or_extract_token_split(
                cache_path=cache_root / f"{split}.pt",
                metadata=metadata,
                label_index=label_index,
                dataset=motion_datasets[split],
                encoder=encoder,
                device=device,
                feature_batch_size=feature_batch_size,
                num_workers=args.num_workers,
                recompute=bool(_argument(args, "recompute_features", False)),
            )
        token_datasets[split] = StyleTokenDataset(
            payload, label_index=label_index, fps=model_info["fps"]
        )
    jepa_source = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "model_name": model_info["model_name"],
        "feature_dim": model_info["feature_dim"],
        "num_frames": model_info["num_frames"],
        "token_num_frames": model_info["token_num_frames"],
        "temporal_patch_size": model_info["temporal_patch_size"],
        "stats_root": str(stats_root),
        "stats_mean_sha256": _sha256_file(stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(stats_root / "std.npy"),
        "token_cache_root": str(cache_root),
        "token_cache_dtype": "bfloat16",
        "token_cache_format_version": TOKEN_CACHE_FORMAT_VERSION,
    }
    for key in (
        "token_num_joints", "spatial_grouping", "spatial_pooling",
        "spatial_token_names", "trajectory_token_index", "body_token_offset",
        "trajectory_fields",
    ):
        if key in model_info:
            jepa_source[key] = model_info[key]
    del encoder, motion_datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return PreparedInput(
        datasets=token_datasets,
        label_index=label_index,
        input_dim=int(model_info["feature_dim"]),
        num_frames=int(model_info["token_num_frames"]),
        stats_root=stats_root,
        input_source="jepa",
        jepa_source=jepa_source,
        task=task,
        dataset_name=dataset_name,
    )


def _make_loaders(
    datasets: dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }


def _signature(
    args: argparse.Namespace,
    model_name: str,
    dataset_root: Path,
    model_config: dict[str, Any],
    prepared: PreparedInput,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    signature = {
        "model": model_name,
        "architecture": model_config,
        "dataset_index_sha256": _sha256_file(dataset_root / "index.json"),
        "stats_mean_sha256": _sha256_file(prepared.stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(prepared.stats_root / "std.npy"),
        "input_source": prepared.input_source,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_lr": args.final_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "use_bfloat16": args.use_bfloat16,
    }
    if prepared.task == "multilabel":
        signature["task"] = prepared.task
        signature["selection_metric"] = "val_mean_average_precision"
        if pos_weight is not None:
            signature["pos_weight"] = {
                "mode": "sqrt_inverse_frequency",
                "cap": float(_argument(args, "pos_weight_cap", 10.0)),
                "values": pos_weight.tolist(),
            }
    if prepared.jepa_source is not None:
        signature["jepa_source"] = prepared.jepa_source
    return signature


def _result_fields(task: str = "single_label") -> list[str]:
    metric_fields = MULTILABEL_METRIC_FIELDS if task == "multilabel" else METRIC_FIELDS
    return [
        "epoch",
        "learning_rate",
        *[f"train_{field}" for field in metric_fields],
        *[f"val_{field}" for field in metric_fields],
    ]


def run_model(
    args: argparse.Namespace,
    model_name: str,
    *,
    prepared: PreparedInput,
    output_root: Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output = output_root / model_name / f"seed-{args.seed}"
    device = resolve_device(args.device) if device is None else device
    datasets = prepared.datasets
    label_index = prepared.label_index
    row_labels_by_sample = (
        label_index.row_labels_by_path
        if isinstance(label_index, BabelLabelIndex) else None
    )
    validation_used = len(datasets["val"]) > 0
    selection = "validation_best" if validation_used else "fixed_last_epoch"
    if prepared.task == "multilabel":
        labels = datasets["train"].labels
        target_matrix = labels if isinstance(labels, torch.Tensor) else torch.stack(labels)
        train_labels = set(
            target_matrix.any(dim=0).nonzero(as_tuple=True)[0].tolist()
        )
        pos_weight_cpu = (
            _positive_class_weights(
                target_matrix, float(_argument(args, "pos_weight_cap", 10.0))
            )
            if _argument(args, "pos_weight", "none") == "sqrt_inverse_frequency"
            else None
        )
    else:
        train_labels = {int(label) for label in datasets["train"].labels}
        pos_weight_cpu = None
    missing_train = set(range(label_index.num_classes)) - train_labels
    if missing_train and prepared.task != "multilabel":
        raise ValueError(f"Training split is missing class IDs: {sorted(missing_train)}")
    _seed_all(args.seed)
    model, model_config = make_classifier(
        model_name,
        input_dim=prepared.input_dim,
        num_frames=prepared.num_frames,
        num_classes=label_index.num_classes,
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    signature = _signature(
        args, model_name, dataset_root, model_config, prepared, pos_weight_cpu
    )
    summary_path = output / "summary.json"
    if summary_path.is_file() and not args.overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete" and summary.get("signature") == signature:
            if (
                prepared.task == "multilabel"
                and summary.get("best_val") is not None
                and "top1_label_row_accuracy" not in summary["best_val"]
            ):
                best_path = output / summary["head_filename"]
                best = _torch_load_checkpoint(best_path)
                if best.get("format_version") != CLASSIFIER_CHECKPOINT_FORMAT_VERSION:
                    raise ValueError("Unsupported best classifier checkpoint format")
                model.load_state_dict(best["model"], strict=True)
                model = model.to(device)
                pos_weight = pos_weight_cpu.to(device) if pos_weight_cpu is not None else None
                generator = torch.Generator().manual_seed(args.seed)
                loaders = _make_loaders(
                    datasets, batch_size=args.batch_size, num_workers=args.num_workers,
                    device=device, generator=generator,
                )
                summary["best_val"] = asdict(evaluate(
                    model, loaders["val"], device=device,
                    num_classes=label_index.num_classes,
                    use_bfloat16=args.use_bfloat16, task=prepared.task,
                    pos_weight=pos_weight, row_labels_by_sample=row_labels_by_sample,
                ))
                if len(datasets["test"]):
                    summary["test"] = asdict(evaluate(
                        model, loaders["test"], device=device,
                        num_classes=label_index.num_classes,
                        use_bfloat16=args.use_bfloat16, task=prepared.task,
                        pos_weight=pos_weight, row_labels_by_sample=row_labels_by_sample,
                    ))
                _atomic_json_save(summary, summary_path)
            return summary
        raise FileExistsError(f"Classifier result already exists with another config: {output}")
    output.mkdir(parents=True, exist_ok=True)
    existing = [
        name
        for name in (
            "metrics.csv",
            "classifier-best.pth.tar",
            "classifier-final.pth.tar",
            "classifier-latest.pth.tar",
        )
        if (output / name).exists()
    ]
    if existing and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"Partial classifier output exists under {output}: {existing}; use --resume"
        )
    if args.overwrite:
        for name in existing + ["summary.json", "model-config.json"]:
            path = output / name
            if path.exists():
                path.unlink()

    model = model.to(device)
    pos_weight = pos_weight_cpu.to(device) if pos_weight_cpu is not None else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    final_factor = args.final_lr / args.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _lr_factor(
            epoch,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            final_factor=final_factor,
        ),
    )
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch = 0
    best_epoch = 0
    best_accuracy = -1.0
    latest_path = output / "classifier-latest.pth.tar"
    best_path = output / (
        "classifier-best.pth.tar" if validation_used else "classifier-final.pth.tar"
    )
    if args.resume and latest_path.is_file():
        checkpoint = _torch_load_checkpoint(latest_path)
        if checkpoint.get("format_version") != CLASSIFIER_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported latest classifier checkpoint format")
        if checkpoint.get("signature") != signature:
            raise ValueError("Latest classifier checkpoint config does not match this run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_accuracy = float(checkpoint["best_accuracy"])
        _restore_rng_state(checkpoint["rng_state"], generator)
    loaders = _make_loaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        generator=generator,
    )
    _atomic_json_save(label_index.to_json(), output / "class-index.json")

    metrics_path = output / "metrics.csv"
    fields = _result_fields(prepared.task)
    mode = "a" if start_epoch else "w"
    if start_epoch:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Resume metrics do not exist: {metrics_path}")
        with metrics_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            previous_fields = reader.fieldnames
            previous_rows = list(reader)
        if len(previous_rows) != start_epoch:
            raise ValueError("Metrics row count does not match checkpoint next_epoch")
        if previous_fields != fields:
            legacy_fields = [
                field for field in fields if not field.endswith("top1_label_row_accuracy")
            ]
            if previous_fields != legacy_fields:
                raise ValueError("Resume metrics columns do not match this run")
            temporary_path = metrics_path.with_suffix(".csv.tmp")
            with temporary_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(previous_rows)
            temporary_path.replace(metrics_path)
    with metrics_path.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not start_epoch:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_epoch(
                model,
                loaders["train"],
                optimizer,
                device=device,
                num_classes=label_index.num_classes,
                use_bfloat16=args.use_bfloat16,
                gradient_clip=args.gradient_clip,
                task=prepared.task,
                pos_weight=pos_weight,
                row_labels_by_sample=row_labels_by_sample,
            )
            val_metrics = None
            if validation_used:
                val_metrics = evaluate(
                    model,
                    loaders["val"],
                    device=device,
                    num_classes=label_index.num_classes,
                    use_bfloat16=args.use_bfloat16,
                    task=prepared.task,
                    pos_weight=pos_weight,
                    row_labels_by_sample=row_labels_by_sample,
                )
            row = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
                **(
                    {f"val_{key}": value for key, value in asdict(val_metrics).items()}
                    if val_metrics is not None
                    else {f"val_{key}": "" for key in (
                        MULTILABEL_METRIC_FIELDS if prepared.task == "multilabel" else METRIC_FIELDS
                    )}
                ),
            }
            writer.writerow(row)
            file.flush()
            score = (
                val_metrics.mean_average_precision
                if prepared.task == "multilabel" and val_metrics is not None
                else val_metrics.top1_accuracy if val_metrics is not None else None
            )
            if score is not None and score > best_accuracy:
                best_accuracy = score
                best_epoch = epoch + 1
                _atomic_torch_save(
                    {
                        "format_version": CLASSIFIER_CHECKPOINT_FORMAT_VERSION,
                        "model": model.state_dict(),
                        "architecture": model_config,
                    },
                    best_path,
                )
            scheduler.step()
            _atomic_torch_save(
                {
                    "format_version": CLASSIFIER_CHECKPOINT_FORMAT_VERSION,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "next_epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_accuracy": best_accuracy,
                    "rng_state": _capture_rng_state(generator),
                    "signature": signature,
                },
                latest_path,
            )
            validation_text = f", val={score:.4f}" if score is not None else ""
            train_score = (
                train_metrics.mean_average_precision if prepared.task == "multilabel"
                else train_metrics.top1_accuracy
            )
            print(
                f"{model_name} epoch {epoch + 1:03d}/{args.epochs}: "
                f"train={train_score:.4f}{validation_text}, "
                f"lr={current_lr:.3e}",
                flush=True,
            )

    if not validation_used:
        best_epoch = args.epochs
        best_accuracy = float("nan")
        _atomic_torch_save(
            {
                "format_version": CLASSIFIER_CHECKPOINT_FORMAT_VERSION,
                "model": model.state_dict(),
                "architecture": model_config,
                "selection": selection,
            },
            best_path,
        )

    best = _torch_load_checkpoint(best_path)
    if best.get("format_version") != CLASSIFIER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported best classifier checkpoint format")
    model.load_state_dict(best["model"], strict=True)
    best_val_metrics = None
    if validation_used:
        best_val_metrics = evaluate(
            model,
            loaders["val"],
            device=device,
            num_classes=label_index.num_classes,
            use_bfloat16=args.use_bfloat16,
            task=prepared.task,
            pos_weight=pos_weight,
            row_labels_by_sample=row_labels_by_sample,
        )
    test_metrics = (
        evaluate(
            model, loaders["test"], device=device,
            num_classes=label_index.num_classes,
            use_bfloat16=args.use_bfloat16, task=prepared.task,
            pos_weight=pos_weight,
            row_labels_by_sample=row_labels_by_sample,
        )
        if len(datasets["test"]) else None
    )
    summary = {
        "status": "complete",
        "model": model_name,
        "num_parameters": num_parameters,
        "best_epoch": best_epoch,
        "best_val": None if best_val_metrics is None else asdict(best_val_metrics),
        "selection": selection,
        "validation_used": validation_used,
        "head_filename": best_path.name,
        "test": None if test_metrics is None else asdict(test_metrics),
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "signature": signature,
    }
    if prepared.task == "multilabel":
        summary["task"] = prepared.task
        summary["missing_train_class_ids"] = sorted(missing_train)
        summary["selection_metric"] = "val_mean_average_precision"
        if pos_weight_cpu is not None:
            summary["pos_weight"] = signature["pos_weight"]
    _atomic_json_save(summary, summary_path)
    latest_path.unlink(missing_ok=True)
    (output / "model-config.json").unlink(missing_ok=True)
    return summary


def _copy_metrics(output_root: Path, findings_root: Path, model_name: str, seed: int) -> Path:
    source = output_root / model_name / f"seed-{seed}" / "metrics.csv"
    destination = findings_root / f"{model_name}-metrics.csv"
    shutil.copyfile(source, destination)
    return destination


def _architecture_descriptions(summaries: dict[str, dict[str, Any]]) -> list[str]:
    cnn = summaries["cnn"]["signature"]["architecture"]
    transformer = summaries["transformer"]["signature"]["architecture"]
    cnn_blocks = cnn["blocks_per_stage"]
    return [
        (
            f"- CNN: temporal ResNet, widths {'/'.join(map(str, cnn['widths']))}, "
            f"{cnn_blocks} block{'s' if cnn_blocks != 1 else ''} per stage, "
            f"dropout {cnn['dropout']}, "
            "masked mean pooling."
        ),
        (
            f"- Transformer: dim {transformer['embed_dim']}, "
            f"{transformer['depth']} blocks, {transformer['num_heads']} heads, "
            f"MLP ratio {transformer['mlp_ratio']}, dropout {transformer['dropout']}, "
            "learnable CLS-token pooling."
        ),
    ]


def write_findings(
    summaries: dict[str, dict[str, Any]],
    *,
    output_root: Path,
    findings_root: Path,
    seed: int,
    input_source: str,
    jepa_source: dict[str, Any] | None,
    task: str = "single_label",
    dataset_name: str = "100style",
) -> None:
    validation_used = all(
        bool(summary.get("validation_used", True)) for summary in summaries.values()
    )
    findings_root.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(
        {
            "format_version": 1,
            "input_source": input_source,
            "seed": seed,
            "summaries": summaries,
        },
        findings_root / "results.json",
    )
    metrics: dict[str, list[dict[str, str]]] = {}
    for model_name in MODELS:
        path = _copy_metrics(output_root, findings_root, model_name, seed)
        with path.open(encoding="utf-8", newline="") as file:
            metrics[model_name] = list(csv.DictReader(file))

    score_field = "mean_average_precision" if task == "multilabel" else "top1_accuracy"
    score_title = "Mean AP (%)" if task == "multilabel" else "Top-1 (%)"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for model_name, color in (("cnn", "tab:blue"), ("transformer", "tab:orange")):
        rows = metrics[model_name]
        epochs = [int(row["epoch"]) for row in rows]
        axes[0, 0].plot(epochs, [float(row["train_loss"]) for row in rows], label=model_name, color=color)
        axes[1, 0].plot(epochs, [float(row[f"train_{score_field}"]) * 100 for row in rows], label=model_name, color=color)
        if validation_used:
            axes[0, 1].plot(epochs, [float(row["val_loss"]) for row in rows], label=model_name, color=color)
            axes[1, 1].plot(epochs, [float(row[f"val_{score_field}"]) * 100 for row in rows], label=model_name, color=color)
    titles = (("Train loss", "Validation loss"), (f"Train {score_title}", f"Validation {score_title}"))
    for row_index, row_axes in enumerate(axes):
        for column_index, axis in enumerate(row_axes):
            axis.set_title(titles[row_index][column_index])
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend()
    if not validation_used:
        for axis in (axes[0, 1], axes[1, 1]):
            axis.text(0.5, 0.5, "Validation disabled", ha="center", va="center", transform=axis.transAxes)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(findings_root / "training-curves.png", dpi=180)
    plt.close(figure)

    if task == "multilabel":
        weight_config = summaries["cnn"].get("pos_weight")
        weighting_text = (
            "BCEWithLogitsLoss is used without class weights or balanced sampling."
            if weight_config is None else (
                "BCEWithLogitsLoss uses train-only positive weights "
                "sqrt(negative/positive), clipped to [1, "
                f"{weight_config['cap']:g}]; weight range "
                f"{min(weight_config['values']):.2f}–"
                f"{max(weight_config['values']):.2f}. "
                "Train and validation loss both use these weights."
            )
        )
        lines = [
            f"# {dataset_name.upper()} multi-label classifiers",
            "",
            f"Input: {'raw motion' if input_source == 'raw' else 'frozen JEPA tokens'}; seed {seed}.",
            "One motion chunk has one binary target vector containing all its action labels.",
            weighting_text,
            "The best checkpoint is selected by validation mean AP.",
            "Classes with no positives in a split are excluded from that split's mean AP.",
            "Top-1/top-5 hit means at least one predicted class is in the target set.",
            "Top-1 label-row accuracy counts each original BABEL index row separately, "
            "matching the public 2s-AGCN validation metric's unit of evaluation.",
            "Validation is used for model selection and reporting; the public test split is empty.",
            "",
            "![Training curves](training-curves.png)",
            "",
            "| Model | Parameters | Best epoch | Val mean AP | Val top-1 hit | Val top-1 label-row accuracy | Val top-5 hit | Missing train classes |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        if jepa_source is not None:
            lines[2:2] = [
                f"JEPA checkpoint: `{jepa_source['checkpoint_path']}`",
                f"Checkpoint key: `{jepa_source['checkpoint_key']}`",
            ]
        if any(
            rows and "val_top1_label_row_accuracy" not in rows[0]
            for rows in metrics.values()
        ):
            lines.insert(
                lines.index("![Training curves](training-curves.png)"),
                "Older epoch CSVs lack label-row accuracy; the best checkpoints were re-evaluated below.",
            )
        for model_name in MODELS:
            summary = summaries[model_name]
            val = summary["best_val"]
            values = (
                f"{val['mean_average_precision'] * 100:.2f}",
                f"{val['top1_hit'] * 100:.2f}",
                f"{val['top1_label_row_accuracy'] * 100:.2f}",
                f"{val['top5_hit'] * 100:.2f}",
            ) if val is not None else ("-", "-", "-", "-")
            lines.append(
                f"| {model_name} | {summary['num_parameters']:,} | {summary['best_epoch']} | "
                f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | "
                f"{summary['missing_train_class_ids']} |"
            )
        lines.extend([
            "", "## Model architecture", "",
            *_architecture_descriptions(summaries),
            "", "## Files", "",
            "- [CNN metrics](cnn-metrics.csv)",
            "- [Transformer metrics](transformer-metrics.csv)",
            "- [Classifier summaries](results.json)", "",
        ])
        (findings_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
        return

    source_description = (
        "100STYLE raw motion `[90,366]`"
        if input_source == "raw"
        else (
            f"frozen `{jepa_source['model_name']}` frame-token features "
            f"`[90,{jepa_source['feature_dim']}]`"
        )
    )
    lines = [
        "# 100STYLE Raw-Motion Classifiers",
        "",
        f"Both models were trained from {source_description} with seed {seed}.",
        "",
        "## Shared settings",
        "",
        (
            "- 100STYLE train statistics normalization"
            if input_source == "raw"
            else "- Frozen JEPA target-encoder tokens using pretraining statistics"
        ),
        "- CrossEntropyLoss; no class weighting, balanced sampling, or augmentation",
        "- AdamW, LR 3e-4, weight decay 0.05",
        "- 5-epoch warmup followed by cosine decay, 100 epochs",
        "- CUDA BF16 autocast; float32 parameters and optimizer",
        (
            "- Best validation top-1 checkpoint restored before one test evaluation"
            if validation_used
            else "- Fixed last-epoch checkpoint followed by one test evaluation"
        ),
        "",
        "![Training curves](training-curves.png)",
        "",
        "## Results",
        "",
        "| Model | Parameters | Best epoch | Val top-1 | Val macro | Val top-5 | Test top-1 | Test macro | Test top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if jepa_source is not None:
        lines[3:3] = [
            "",
            f"- JEPA checkpoint: `{jepa_source['checkpoint_path']}`",
            f"- SHA256: `{jepa_source['checkpoint_sha256']}`",
            f"- Checkpoint key: `{jepa_source['checkpoint_key']}`",
        ]
    for model_name in MODELS:
        summary = summaries[model_name]
        val = summary["best_val"]
        test = summary["test"]
        val_values = (
            ("-", "-", "-")
            if val is None
            else (
                f"{val['top1_accuracy'] * 100:.2f}",
                f"{val['macro_accuracy'] * 100:.2f}",
                f"{val['top5_accuracy'] * 100:.2f}",
            )
        )
        lines.append(
            f"| {model_name} | {summary['num_parameters']:,} | {summary['best_epoch']} | "
            f"{val_values[0]} | {val_values[1]} | {val_values[2]} | "
            f"{test['top1_accuracy'] * 100:.2f} | "
            f"{test['macro_accuracy'] * 100:.2f} | {test['top5_accuracy'] * 100:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Model architecture",
            "",
            *_architecture_descriptions(summaries),
            "",
            "## Raw metrics",
            "",
            "- [CNN metrics](cnn-metrics.csv)",
            "- [Transformer metrics](transformer-metrics.csv)",
            "- [Classifier summaries](results.json)",
            "",
        ]
    )
    (findings_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("epochs must be positive and warmup_epochs must be in [0, epochs)")
    if args.lr <= 0 or args.final_lr < 0 or args.final_lr > args.lr:
        raise ValueError("Require 0 <= final_lr <= lr and lr > 0")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if int(_argument(args, "feature_batch_size", 256)) <= 0:
        raise ValueError("feature_batch_size must be positive")
    device = resolve_device(args.device)
    input_source = str(_argument(args, "input_source", "raw"))
    if args.model == "linear" and input_source != "raw":
        raise ValueError("The linear baseline supports only normalized raw motion")
    checkpoint_value = _argument(args, "jepa_checkpoint", None)
    output_value = _argument(args, "output_root", None)
    findings_value = _argument(args, "findings_root", None)
    task, dataset_name = classification_dataset_kind(
        Path(args.dataset_root).expanduser().resolve()
    )
    pos_weight_mode = str(_argument(args, "pos_weight", "none"))
    pos_weight_cap = float(_argument(args, "pos_weight_cap", 10.0))
    if pos_weight_mode not in ("none", "sqrt_inverse_frequency"):
        raise ValueError(f"Unknown positive class weighting mode: {pos_weight_mode}")
    if not math.isfinite(pos_weight_cap) or pos_weight_cap < 1:
        raise ValueError("--pos-weight-cap must be finite and at least 1")
    if task != "multilabel" and pos_weight_mode != "none":
        raise ValueError("Positive class weighting is supported only for BABEL")
    if input_source == "jepa":
        if checkpoint_value is None:
            raise ValueError("--jepa-checkpoint is required for --input-source jepa")
        checkpoint_root = Path(checkpoint_value).expanduser().resolve().parent
        default_output = checkpoint_root / "linear-probe/classifiers"
        if task == "multilabel":
            default_output = default_output / dataset_name
    else:
        default_output = PROJECT_ROOT / f"output/{dataset_name}-classifiers"
    output_root = (
        Path(output_value).expanduser().resolve()
        if output_value is not None
        else default_output
    )
    default_findings = (
        output_root / "findings"
        if task == "multilabel" or input_source == "jepa"
        else DEFAULT_RAW_FINDINGS_ROOT
    )
    findings_root = (
        Path(findings_value).expanduser().resolve()
        if findings_value is not None
        else default_findings
    )
    prepared = _prepare_input(args, device=device)
    selected = MODELS if args.model == "all" else (args.model,)
    summaries = {
        model_name: run_model(
            args,
            model_name,
            prepared=prepared,
            output_root=output_root,
            device=device,
        )
        for model_name in selected
    }
    if set(summaries) == set(MODELS):
        write_findings(
            summaries,
            output_root=output_root,
            findings_root=findings_root,
            seed=args.seed,
            input_source=prepared.input_source,
            jepa_source=prepared.jepa_source,
            task=prepared.task,
            dataset_name=prepared.dataset_name,
        )
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 100STYLE or BABEL classifiers from raw motion or frozen JEPA tokens"
    )
    parser.add_argument("--model", choices=(*AVAILABLE_MODELS, "all"), default="all")
    parser.add_argument("--input-source", choices=("raw", "jepa"), default="raw")
    parser.add_argument("--jepa-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-key",
        choices=("target_encoder", "encoder"),
        default="target_encoder",
    )
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--feature-cache-root", type=Path, default=None)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-soma77-processed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: raw output root or <JEPA checkpoint dir>/linear-probe/classifiers",
    )
    parser.add_argument(
        "--findings-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--final-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--pos-weight",
        choices=("none", "sqrt_inverse_frequency"),
        default="none",
        help="BABEL only: weight each positive class by capped sqrt(negative/positive)",
    )
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument(
        "--use-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    summaries = run(build_parser().parse_args())
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
