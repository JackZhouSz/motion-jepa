"""Bounded, provenance-checked caches of full frozen JEPA target tokens."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from dataset.motion_dataset import MotionDataset
from experiment.linear_probe.features import (
    PROJECT_ROOT,
    _atomic_json_save,
    _sha256_file,
    load_frozen_encoder,
    resolve_device,
    resolve_pretraining_stats,
)


CACHE_FORMAT_VERSION = 1
FEATURE_TRANSFORM = "jepa_target_layer_norm"
FEATURE_LAYER_NORM_EPS = 1.0e-5
_SPLITS = ("train", "val", "test")


def _path(value: Any) -> Path:
    result = Path(str(value)).expanduser()
    return (result if result.is_absolute() else PROJECT_ROOT / result).resolve()


def _split_name(split: str) -> str:
    if split not in _SPLITS:
        raise ValueError(f"Unknown prediction split: {split!r}")
    return split


def _source(config: dict, device: torch.device):
    checkpoint_path = _path(config["jepa_checkpoint"])
    encoder, source_config, model_info = load_frozen_encoder(
        checkpoint_path, "target_encoder", device
    )
    if not bool(source_config["data"].get("normalize", False)):
        raise ValueError("Prediction requires a train-stat normalized JEPA checkpoint")
    explicit_stats = config.get("stats_path")
    stats_root = resolve_pretraining_stats(
        source_config, None if explicit_stats is None else _path(explicit_stats)
    )
    info = dict(model_info)
    info["token_layout"] = encoder.token_layout.signature()
    return encoder, source_config, info, checkpoint_path, stats_root


def _split_metadata(
    config: dict,
    split: str,
    source_config: dict,
    model_info: dict,
    checkpoint_path: Path,
    stats_root: Path,
) -> tuple[MotionDataset, int, dict]:
    _split_name(split)
    root = _path(config["dataset_root"])
    dataset = MotionDataset(
        root_path=root,
        meta_files=f"{split}.txt",
        num_frames=model_info["num_frames"],
        fps=model_info["fps"],
        motion_dim=model_info["motion_dim"],
        normalize=True,
        stats_path=stats_root,
    )
    limit = int(config.get(f"limit_{split}", 0))
    if limit < 0:
        raise ValueError(f"limit_{split} must be non-negative")
    count = min(limit, len(dataset)) if limit else len(dataset)
    if count == 0:
        raise ValueError(f"Prediction split {split!r} contains no samples")
    files = [root / "meta.json", root / f"{split}.txt", root / "motions" / f"{split}.json"]
    # BONES has an index with source provenance; small generic fixtures may not.
    if (root / "index.json").is_file():
        files.append(root / "index.json")
    fingerprint = {str(path.relative_to(root)): _sha256_file(path) for path in files}
    layout = model_info["token_layout"]
    token_shape = [count, int(layout["token_num_frames"])]
    if layout["kind"] == "2d":
        token_shape.append(int(layout["token_num_joints"]))
    token_shape.append(int(model_info["feature_dim"]))
    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_key": "target_encoder",
        "source_model_config": {
            "data": source_config["data"],
            "meta": source_config["meta"],
            "patch": source_config.get("patch"),
        },
        "dataset_root": str(root),
        "dataset_fingerprints": fingerprint,
        # The hashed split is the authority for selected sample IDs, relative
        # paths, FPS and lengths; first-N selection never reorders its rows.
        "sample_index_format": "sample_id,relative_npy_path,fps,actual_length",
        "sample_selection": "first_n_manifest_rows",
        "stats_root": str(stats_root),
        "stats_mean_sha256": _sha256_file(stats_root / "mean.npy"),
        "stats_std_sha256": _sha256_file(stats_root / "std.npy"),
        "motion_normalization": "pretraining_train_mean_std_floor_1e-6_to_one",
        "feature_transform": FEATURE_TRANSFORM,
        "feature_layer_norm_eps": FEATURE_LAYER_NORM_EPS,
        "feature_storage": "bfloat16_bits_numpy_uint16",
        "use_bfloat16": bool(config.get("use_bfloat16", True)),
        "encoder_precision_policy": "bf16_autocast_on_cuda_otherwise_float32",
        "split": split,
        "limit": limit,
        "num_samples": count,
    }
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "provenance": provenance,
        "model_info": model_info,
        "token_shape": token_shape,
    }
    return dataset, count, metadata


def _cache_directory(config: dict, split: str) -> Path:
    return _path(config["cache_root"]) / _split_name(split)


def _read_cache(directory: Path, expected: dict) -> np.memmap:
    marker = directory / "completed.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Prediction cache is incomplete or missing: {directory}")
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid prediction cache completion marker: {marker}") from error
    if actual != expected:
        raise ValueError(
            f"Prediction cache provenance, source, statistics or token layout differs: {directory}; "
            "use a new cache root"
        )
    try:
        tokens = np.load(directory / "tokens.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid prediction cache token file: {directory}") from error
    if (not isinstance(tokens, np.memmap) or tokens.dtype != np.uint16
            or list(tokens.shape) != expected["token_shape"]):
        raise ValueError(f"Prediction cache token shape or BF16 storage differs: {directory}")
    return tokens


def prepare_caches(config: dict, *, splits=("train", "val", "test")) -> dict:
    """Extract fixed EMA tokens in bounded batches, publishing completion last.

    Completed caches are reused only after exact provenance validation. An
    unpublished extraction can be restarted; readers never consume its files.
    """
    requested = tuple(_split_name(split) for split in splits)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Cache splits must be non-empty and unique")
    batch_size = int(config.get("cache_batch_size", 256))
    workers = int(config.get("num_workers", 8))
    if batch_size < 1 or workers < 0:
        raise ValueError("Cache batch size must be positive and worker count non-negative")
    device = resolve_device(str(config.get("device", "auto")))
    if (device.type == "cuda" and config.get("use_bfloat16", True)
            and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 encoder autocast is unavailable on this CUDA device")
    encoder, source_config, model_info, checkpoint_path, stats_root = _source(config, device)
    result = {}
    for split in requested:
        dataset, count, metadata = _split_metadata(
            config, split, source_config, model_info, checkpoint_path, stats_root
        )
        directory = _cache_directory(config, split)
        if (directory / "completed.json").exists():
            existing = _read_cache(directory, metadata)
            del existing
            print(f"prediction_cache cache_reused split={split} samples={count} path={directory}", flush=True)
            result[split] = metadata
            continue
        started = time.perf_counter()
        print(f"prediction_cache extraction_start split={split} samples={count} "
              f"batch_size={batch_size} device={device} path={directory}", flush=True)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "tokens.tmp.npy"
        tokens = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.uint16, shape=tuple(metadata["token_shape"])
        )
        loader = DataLoader(
            Subset(dataset, range(count)), batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(0),
        )
        offset = 0
        try:
            with torch.no_grad():
                for motion, fps, length in tqdm(
                    loader, desc=f"Cache {split}", file=sys.stdout, mininterval=1.0
                ):
                    motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
                    fps = fps.to(device=device, dtype=torch.float32, non_blocking=True)
                    length = length.to(device=device, dtype=torch.long, non_blocking=True)
                    active = torch.arange(motion.shape[1], device=device)[None] < length[:, None]
                    context = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and config.get("use_bfloat16", True)
                        else nullcontext()
                    )
                    with context:
                        encoded = encoder(motion, fps, valid_frames=active)
                    valid = encoder.token_layout.valid_token_mask(active)
                    if encoded.ndim == 4:
                        valid = valid[:, :, None].expand(encoded.shape[:-1])
                    if encoded.ndim not in (3, 4) or encoded.shape[1:] != tuple(metadata["token_shape"][1:]):
                        raise ValueError("Frozen encoder output differs from the source token layout")
                    transformed = torch.zeros_like(encoded, dtype=torch.float32)
                    transformed[valid] = F.layer_norm(
                        encoded[valid].float(), (encoded.shape[-1],), eps=FEATURE_LAYER_NORM_EPS
                    )
                    if not torch.isfinite(transformed).all():
                        raise ValueError("Frozen encoder produced non-finite prediction features")
                    bits = transformed.to(dtype=torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy()
                    tokens[offset:offset + len(motion)] = bits
                    offset += len(motion)
            if offset != count:
                raise RuntimeError("Prediction feature extraction did not cover its selected split")
            tokens.flush()
        finally:
            del tokens
        os.replace(temporary, directory / "tokens.npy")
        _atomic_json_save(metadata, directory / "completed.json")
        print(f"prediction_cache extraction_complete split={split} samples={count} "
              f"elapsed_seconds={time.perf_counter() - started:.2f} path={directory}", flush=True)
        result[split] = metadata
    return result


class PredictionDataset(Dataset):
    """Pair read-only BF16 token caches with lazy normalized raw-motion targets."""

    def __init__(self, config: dict, split: str):
        encoder, source_config, model_info, checkpoint_path, stats_root = _source(
            config, torch.device("cpu")
        )
        self.token_layout = encoder.token_layout
        del encoder
        self._raw, self._count, metadata = _split_metadata(
            config, split, source_config, model_info, checkpoint_path, stats_root
        )
        self.model_info = model_info
        self.provenance = metadata["provenance"]
        self.mean = torch.from_numpy(self._raw.mean.copy()).float()
        self.std = torch.from_numpy(self._raw.std.copy()).float()
        self.entries = self._raw.entries[:self._count]
        self._cache_directory = _cache_directory(config, split)
        self._cache_metadata = metadata
        self._tokens = _read_cache(self._cache_directory, metadata)

    def __getstate__(self):
        # NumPy otherwise pickles the entire memmap for spawn-based workers.
        state = dict(self.__dict__)
        state["_tokens"] = None
        return state

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = _read_cache(self._cache_directory, self._cache_metadata)
        motion, fps, length = self._raw[index]
        # Copy one slice so torch never receives a read-only NumPy buffer.
        bits = torch.from_numpy(np.array(self._tokens[index], dtype=np.uint16, copy=True))
        features = bits.view(torch.bfloat16)
        if not torch.isfinite(features).all():
            raise ValueError(f"Prediction cache contains non-finite tokens: {self.entries[index].sample_id}")
        return {
            "tokens": features,
            "motion": torch.from_numpy(motion),
            "fps": float(fps),
            "valid_frames": torch.arange(len(motion)) < int(length),
            "length": int(length),
            "sample_id": self.entries[index].sample_id,
        }


__all__ = ["PredictionDataset", "prepare_caches", "FEATURE_TRANSFORM", "FEATURE_LAYER_NORM_EPS"]
