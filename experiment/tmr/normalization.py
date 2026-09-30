"""Train-only, streaming channel statistics for packed frozen JEPA tokens."""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from experiment.linear_probe.features import _atomic_json_save
from .dataset import RaggedTokenBank


FEATURE_STD_EPSILON = 1e-6


def _file_digest(path):
    # These files can be rewritten within a filesystem timestamp tick during a
    # repair. Read their contents rather than reusing a path/mtime hash cache.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_feature_statistics(
    train_bank: RaggedTokenBank, *, recompute: bool = False, chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute population moments from cached valid tokens, never padded frames.

    Chunk moments are merged in FP64 (Chan's algorithm). Fingerprints cover both
    the training features and saved statistics, including for resumed training.
    """
    if chunk_size < 1:
        raise ValueError("Statistics chunk size must be positive")
    count, dim = int(train_bank.offsets[-1]), int(train_bank.metadata["feature_dim"])
    if count < 1:
        raise ValueError("JEPA feature statistics require nonempty training tokens")
    source = {
        f"train_{name}_sha256": _file_digest(train_bank.root / filename)
        for name, filename in (
            ("metadata", "complete.json"), ("values", "values.npy"), ("offsets", "offsets.npy"),
        )
    }
    signature = {
        "format_version": 1, "method": "train_valid_tokens_per_channel",
        "num_tokens": count, "feature_dim": dim, "ddof": 0,
        "epsilon": FEATURE_STD_EPSILON, **source,
    }
    root = train_bank.root.parent / "stats"
    if root.exists() and not recompute:
        marker = root / "complete.json"
        if not marker.is_file():
            raise ValueError("Incomplete JEPA feature statistics; run prepare_stats --recompute")
        metadata = json.loads(marker.read_text())
        if any(metadata.get(key) != value for key, value in signature.items()):
            raise ValueError("JEPA feature statistics are stale; run prepare_stats --recompute")
    else:
        values = np.load(train_bank.root / "values.npy", mmap_mode="r", allow_pickle=False)
        mean, m2, seen = np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64), 0
        for first in tqdm(range(0, count, chunk_size), desc="Train JEPA mean/std"):
            # BF16 bits occupy the high half of an IEEE float32; NumPy has no BF16 dtype.
            chunk = (values[first:first + chunk_size].astype(np.uint32) << 16).view(np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError("Training JEPA cache contains non-finite tokens")
            chunk = chunk.astype(np.float64)
            batch_mean = chunk.mean(axis=0)
            centered = chunk - batch_mean
            batch_m2 = np.einsum("ij,ij->j", centered, centered)
            delta, size = batch_mean - mean, len(chunk)
            total = seen + size
            m2 += batch_m2 + delta * delta * (seen * size / total)
            mean += delta * (size / total)
            seen = total
        std = np.sqrt(np.maximum(m2 / seen, 0))
        temporary = root.with_name(f"{root.name}.tmp-{os.getpid()}")
        root.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            np.save(temporary / "mean.npy", mean.astype(np.float32), allow_pickle=False)
            np.save(temporary / "std.npy", std.astype(np.float32), allow_pickle=False)
            metadata = {
                **signature,
                "mean_sha256": _file_digest(temporary / "mean.npy"),
                "std_sha256": _file_digest(temporary / "std.npy"),
            }
            _atomic_json_save(metadata, temporary / "complete.json")
            if root.exists():
                shutil.rmtree(root)
            os.replace(temporary, root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    for name in ("mean", "std"):
        if not (root / f"{name}.npy").is_file() or _file_digest(root / f"{name}.npy") != metadata.get(f"{name}_sha256"):
            raise ValueError("JEPA feature statistics are stale; run prepare_stats --recompute")
    mean = np.load(root / "mean.npy", allow_pickle=False)
    std = np.load(root / "std.npy", allow_pickle=False)
    if (
        mean.shape != (dim,) or std.shape != (dim,)
        or mean.dtype != np.float32 or std.dtype != np.float32
        or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any()
    ):
        raise ValueError("Invalid JEPA feature mean/std")
    # Preserve the actual std in the cache; clamp only the normalization divisor.
    return torch.from_numpy(mean), torch.from_numpy(std).clamp_min(FEATURE_STD_EPSILON), metadata
