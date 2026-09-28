"""Frozen MotionJEPA encoder loading, feature extraction, and cache utilities."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helper import init_mjepa_encoder_from_config  # noqa: E402

SPLITS = ("train", "val", "test")
CACHE_FORMAT_VERSION = 1
GLOBAL_MEAN_POOLING = "valid_token_mean"
SPATIAL_FLATTEN_POOLING = "temporal_mean_spatial_flatten"


@dataclass(frozen=True)
class Metrics:
    loss: float
    top1_accuracy: float
    macro_accuracy: float
    top5_accuracy: float


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_version(path, stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _sha256_file_version(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(device)
    return device


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoints written with NumPy 1.x or 2.x module names."""
    aliases: dict[str, Any] = {}
    try:
        import numpy._core  # type: ignore[attr-defined]  # noqa: F401
    except ModuleNotFoundError:
        module_aliases = {
            "numpy._core": np.core,
            "numpy._core.multiarray": np.core.multiarray,
            "numpy._core.numeric": np.core.numeric,
        }
        for name, module in module_aliases.items():
            if name not in sys.modules:
                sys.modules[name] = module
                aliases[name] = module
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for name, module in aliases.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    return checkpoint


def load_frozen_encoder(
    checkpoint_path: Path,
    checkpoint_key: str,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """Reconstruct an encoder from checkpoint config and strictly load its weights."""
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported MotionJEPA checkpoint format: {checkpoint_path}")
    if checkpoint_key not in checkpoint:
        raise KeyError(f"Checkpoint has no {checkpoint_key!r} weights: {checkpoint_path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint does not contain a training config: {checkpoint_path}")
    try:
        data_config = config["data"]
        meta_config = config["meta"]
        model_name = str(meta_config["model_name"])
        num_frames = int(data_config["num_frames"])
        motion_dim = int(data_config["motion_dim"])
        num_joints = int(data_config["num_joints"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid model config in checkpoint: {checkpoint_path}") from error
    encoder = init_mjepa_encoder_from_config(config, torch.device("cpu"))
    encoder.load_state_dict(checkpoint[checkpoint_key], strict=True)
    encoder.to(device).eval().requires_grad_(False)
    info = {
        "model_name": model_name,
        "num_frames": num_frames,
        "motion_dim": motion_dim,
        "num_joints": num_joints,
        "fps": int(data_config["fps"]),
        "feature_dim": int(encoder.embed_dim),
        "kind": encoder.token_layout.kind,
        "token_num_frames": int(encoder.token_layout.token_num_frames),
        "temporal_patch_size": int(encoder.token_layout.temporal_patch_size),
        "patchified": bool(encoder.token_layout.patchified),
        "use_bfloat16": bool(meta_config.get("use_bfloat16", False)),
    }
    if encoder.token_layout.kind == "2d":
        info["token_num_joints"] = int(encoder.token_layout.token_num_joints)
        if encoder.token_layout.spatial_token_names is not None:
            info["spatial_token_names"] = list(
                encoder.token_layout.spatial_token_names
            )
            info["trajectory_token_index"] = int(
                encoder.token_layout.trajectory_token_index
            )
            info["body_token_offset"] = int(encoder.token_layout.body_token_offset)
            info["trajectory_fields"] = list(encoder.token_layout.trajectory_fields)
    if hasattr(encoder, "spatial_grouping"):
        info.update(
            spatial_grouping=str(encoder.spatial_grouping),
            spatial_pooling=str(encoder.spatial_pooling),
        )
    return encoder, config, info


def resolve_pretraining_stats(
    config: dict[str, Any], explicit_path: Path | None
) -> Path:
    """Resolve the normalization directory used during pretraining."""
    if explicit_path is not None:
        stats_root = explicit_path.expanduser().resolve()
    else:
        data_config = config["data"]
        root_path = Path(str(data_config["root_path"])).expanduser()
        if not root_path.is_absolute():
            root_path = PROJECT_ROOT / root_path
        stats_path = Path(str(data_config.get("stats_path") or "stats")).expanduser()
        stats_root = stats_path if stats_path.is_absolute() else root_path / stats_path
        stats_root = stats_root.resolve()
    missing = [name for name in ("mean.npy", "std.npy") if not (stats_root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pretraining statistics are missing under {stats_root}: {', '.join(missing)}"
        )
    return stats_root


def pool_encoder_output(
    output: torch.Tensor,
    valid_length: torch.Tensor,
    token_layout=None,
    pooling: str = GLOBAL_MEAN_POOLING,
) -> torch.Tensor:
    """Pool valid encoder tokens for a frozen linear probe."""
    frames = output.shape[1]
    if token_layout is not None:
        valid_length = token_layout.valid_token_lengths(valid_length)
    valid_frames = (
        torch.arange(frames, device=output.device).unsqueeze(0)
        < valid_length.to(device=output.device).unsqueeze(1)
    )
    if pooling == SPATIAL_FLATTEN_POOLING:
        if output.ndim != 4:
            raise ValueError(
                f"{SPATIAL_FLATTEN_POOLING} requires 2D output [B,T,J,D], "
                f"got {tuple(output.shape)}"
            )
        weights = valid_frames[:, :, None, None].to(dtype=output.dtype)
        temporal_mean = (output * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
            1.0
        )
        return temporal_mean.flatten(start_dim=1)
    if pooling != GLOBAL_MEAN_POOLING:
        raise ValueError(f"Unknown linear-probe pooling: {pooling!r}")
    if output.ndim == 3:
        weights = valid_frames.unsqueeze(-1).to(dtype=output.dtype)
        return (output * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if output.ndim == 4:
        weights = valid_frames[:, :, None, None].to(dtype=output.dtype)
        numerator = (output * weights).sum(dim=(1, 2))
        denominator = weights.sum(dim=(1, 2)) * output.shape[2]
        return numerator / denominator.clamp_min(1.0)
    raise ValueError(f"Expected encoder output [B,T,D] or [B,T,J,D], got {output.shape}")


def build_cache_metadata(
    *,
    split: str,
    checkpoint_path: Path,
    checkpoint_key: str,
    dataset_root: Path,
    stats_root: Path,
    model_info: dict[str, Any],
    class_names: list[str],
    pooling: str = GLOBAL_MEAN_POOLING,
) -> dict[str, Any]:
    feature_dim = int(model_info["feature_dim"])
    if pooling == SPATIAL_FLATTEN_POOLING:
        if model_info.get("kind") != "2d":
            raise ValueError(f"{SPATIAL_FLATTEN_POOLING} is only valid for 2D encoders")
        feature_dim *= int(model_info["token_num_joints"])
    elif pooling != GLOBAL_MEAN_POOLING:
        raise ValueError(f"Unknown linear-probe pooling: {pooling!r}")
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "split": split,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "dataset_index_sha256": _sha256_file(dataset_root / "index.json"),
        "stats_mean_sha256": _sha256_file(stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(stats_root / "std.npy"),
        "model_name": model_info["model_name"],
        "num_frames": model_info["num_frames"],
        "motion_dim": model_info["motion_dim"],
        "fps": model_info["fps"],
        "feature_dim": feature_dim,
        "token_num_frames": model_info["token_num_frames"],
        "temporal_patch_size": model_info["temporal_patch_size"],
        "patchified": model_info["patchified"],
        "kind": model_info["kind"],
        "pooling": pooling,
        "class_names": class_names,
    }
    for key in (
        "token_num_joints",
        "spatial_grouping",
        "spatial_pooling",
        "spatial_token_names",
        "trajectory_token_index",
        "body_token_offset",
        "trajectory_fields",
    ):
        if key in model_info:
            metadata[key] = model_info[key]
    return metadata


def _validate_feature_cache(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    if payload.get("metadata") != expected:
        raise ValueError(
            "Feature cache metadata is stale; rerun with --recompute-features"
        )
    features = payload.get("features")
    labels = payload.get("labels")
    sample_ids = payload.get("sample_ids")
    if not isinstance(features, torch.Tensor) or features.dtype != torch.float32:
        raise ValueError("Feature cache must contain a float32 feature tensor")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long:
        raise ValueError("Feature cache must contain an int64 label tensor")
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Feature cache tensors have incompatible shapes")
    if features.shape[1] != int(expected["feature_dim"]):
        raise ValueError("Feature cache dimension does not match the encoder")
    if not isinstance(sample_ids, list) or len(sample_ids) != len(features):
        raise ValueError("Feature cache sample IDs do not match feature rows")
    if not torch.isfinite(features).all():
        raise ValueError("Feature cache contains non-finite values")


def extract_features(
    encoder: nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_bfloat16: bool,
    show_progress: bool = True,
    pooling: str = GLOBAL_MEAN_POOLING,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feature_batches = []
    label_batches = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for motion, fps, length, labels, ids in tqdm(
            loader,
            desc="Extract features",
            disable=not show_progress,
        ):
            motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
            fps = fps.to(device=device, dtype=torch.float32, non_blocking=True)
            length = length.to(device=device, non_blocking=True)
            valid_frames = (
                torch.arange(motion.shape[1], device=device).unsqueeze(0)
                < length.unsqueeze(1)
            )
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and use_bfloat16
                else nullcontext()
            )
            with amp_context:
                encoded = encoder(motion, fps, valid_frames=valid_frames)
                pooled = pool_encoder_output(
                    encoded, length, encoder.token_layout, pooling=pooling
                )
            feature_batches.append(pooled.float().cpu())
            label_batches.append(labels.cpu())
            sample_ids.extend(list(ids))
    if not feature_batches:
        raise ValueError("Cannot extract features from an empty split")
    return {
        "features": torch.cat(feature_batches),
        "labels": torch.cat(label_batches),
        "sample_ids": sample_ids,
    }


def load_or_extract_split(
    *,
    split: str,
    cache_path: Path,
    metadata: dict[str, Any],
    dataset: Dataset,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_bfloat16: bool,
    recompute: bool,
    pooling: str = GLOBAL_MEAN_POOLING,
) -> dict[str, Any]:
    if cache_path.is_file() and not recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        _validate_feature_cache(payload, metadata)
        return payload
    payload = extract_features(
        encoder,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_bfloat16=use_bfloat16,
        pooling=pooling,
    )
    payload["metadata"] = metadata
    _validate_feature_cache(payload, metadata)
    _atomic_torch_save(payload, cache_path)
    return payload


__all__ = [
    "CACHE_FORMAT_VERSION",
    "GLOBAL_MEAN_POOLING",
    "Metrics",
    "PROJECT_ROOT",
    "SPLITS",
    "SPATIAL_FLATTEN_POOLING",
    "build_cache_metadata",
    "extract_features",
    "load_frozen_encoder",
    "load_or_extract_split",
    "pool_encoder_output",
    "resolve_device",
    "resolve_pretraining_stats",
]
