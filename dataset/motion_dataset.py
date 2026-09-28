"""Validated, lazy NPY MotionJEPA dataset loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


NPY_FORMAT = "motion_jepa_npy_v1"


@dataclass(frozen=True)
class _NPYEntry:
    sample_id: str
    path: Path
    fps: int
    length: int


class MotionDataset(torch.utils.data.Dataset):
    """Load one NPY clip on demand, normalize valid rows, and end-pad it."""

    REPRESENTATION = "motion_jepa_366_v1"

    def __init__(
        self,
        root_path: str | Path,
        meta_files: str | list[str],
        num_frames: int,
        fps: int,
        motion_dim: int = 366,
        normalize: bool = False,
        stats_path: str | Path | None = None,
    ) -> None:
        self.root_path = Path(root_path)
        self.meta_files = [meta_files] if isinstance(meta_files, str) else list(meta_files)
        self.num_frames = int(num_frames)
        self.fps = int(fps)
        self.motion_dim = int(motion_dim)
        self.normalize = bool(normalize)
        self.dataset_min_frames: int | None = None
        self._validate_dataset_metadata()

        stats_location = Path(stats_path or "stats")
        if not stats_location.is_absolute():
            stats_location = self.root_path / stats_location
        self.stats_root = stats_location.parent if stats_location.suffix else stats_location
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        if self.normalize:
            self.mean, self.std = self._load_statistics(self.stats_root)

        self.entries: list[_NPYEntry] = []
        for meta_file in self.meta_files:
            self._read_npy_index(meta_file)

    def _load_statistics(self, stats_root: Path) -> tuple[np.ndarray, np.ndarray]:
        mean_path = stats_root / "mean.npy"
        std_path = stats_root / "std.npy"
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                f"Statistics do not exist: expected both {mean_path} and {std_path}"
            )
        mean = np.load(mean_path, allow_pickle=False).astype(np.float32, copy=False)
        std = np.load(std_path, allow_pickle=False).astype(np.float32, copy=False)
        expected_shape = (self.motion_dim,)
        if mean.shape != expected_shape or std.shape != expected_shape:
            raise ValueError(
                f"Statistics must have shape {expected_shape}, got "
                f"mean={mean.shape}, std={std.shape}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"Statistics contain non-finite values: {stats_root}")
        return mean, np.where(std < 1.0e-6, 1.0, std)

    def _validate_dataset_metadata(self) -> None:
        path = self.root_path / "meta.json"
        if not path.is_file():
            raise FileNotFoundError(f"MotionJEPA dataset metadata does not exist: {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if "min_frames" in metadata:
            self.dataset_min_frames = int(metadata["min_frames"])
            if not 1 <= self.dataset_min_frames <= self.num_frames:
                raise ValueError(
                    f"Dataset metadata min_frames={self.dataset_min_frames} is outside "
                    f"[1, {self.num_frames}]"
                )
        expected = {
            "representation": self.REPRESENTATION,
            "motion_storage": "npy_float32_v1",
            "motion_dim": self.motion_dim,
            "num_frames": self.num_frames,
            "fps": self.fps,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Dataset metadata {key}={metadata.get(key)!r} does not match "
                    f"configured {value!r}"
                )

    def _valid_length(self, length: int, description: str) -> None:
        if not 1 <= length <= self.num_frames:
            raise ValueError(
                f"Motion sample frame count ({length}) is outside configured range "
                f"[1, {self.num_frames}]: {description}"
            )
        if (
            self.dataset_min_frames is not None
            and length < self.num_frames
            and length < self.dataset_min_frames
        ):
            raise ValueError(
                f"Motion sample frame count ({length}) is below dataset min_frames "
                f"({self.dataset_min_frames}): {description}"
            )

    def _read_npy_index(self, meta_file: str) -> None:
        requested = Path(meta_file)
        if requested.suffix.lower() == ".json":
            manifest_path = self.root_path / requested
            split = requested.stem
            index_path = self.root_path / f"{split}.txt"
        elif requested.suffix.lower() == ".txt":
            split = requested.stem
            index_path = self.root_path / requested
            manifest_path = self.root_path / "motions" / f"{split}.json"
        else:
            raise ValueError(f"Unsupported NPY index format: {requested}")
        if not index_path.is_file():
            raise FileNotFoundError(f"NPY split index does not exist: {index_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"NPY manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected: dict[str, Any] = {
            "format": NPY_FORMAT,
            "split": split,
            "dtype": "float32",
            "representation": self.REPRESENTATION,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "motion_dim": self.motion_dim,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"NPY manifest {key}={manifest.get(key)!r} does not match "
                    f"configured {value!r}: {manifest_path}"
                )
        if manifest.get("split_sha256") != hashlib.sha256(index_path.read_bytes()).hexdigest():
            raise ValueError(f"NPY manifest split hash is stale: {index_path}")
        motion_root = Path(str(manifest.get("motion_root", "")))
        if motion_root.is_absolute() or ".." in motion_root.parts or not motion_root.parts:
            raise ValueError(f"Invalid NPY motion root in {manifest_path}: {motion_root}")

        count_before = len(self.entries)
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise ValueError(f"Malformed NPY index row at {index_path}:{line_number}")
            sample_id, relative_text, fps_text, length_text = fields
            relative_path = Path(relative_text)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe NPY path at {index_path}:{line_number}: {relative_text}")
            if relative_path.parts[: len(motion_root.parts)] != motion_root.parts:
                raise ValueError(
                    f"NPY path at {index_path}:{line_number} is outside {motion_root}"
                )
            try:
                sample_fps = int(fps_text)
                length = int(length_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid NPY FPS or length at {index_path}:{line_number}"
                ) from error
            if not sample_id:
                raise ValueError(f"NPY sample id is empty at {index_path}:{line_number}")
            if sample_fps != self.fps:
                raise ValueError(
                    f"NPY sample FPS {sample_fps} does not match configured {self.fps}: "
                    f"{index_path}:{line_number}"
                )
            path = self.root_path / relative_path
            self._valid_length(length, str(path))
            self.entries.append(_NPYEntry(sample_id, path, sample_fps, length))
        added = len(self.entries) - count_before
        if added != int(manifest.get("num_samples", -1)):
            raise ValueError(
                f"NPY split has {added} rows but manifest records "
                f"{manifest.get('num_samples')!r}: {manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _normalize_and_pad(self, motion: np.ndarray, length: int) -> np.ndarray:
        if self.normalize:
            motion = (motion - self.mean) / self.std
        padded = np.zeros((self.num_frames, self.motion_dim), dtype=np.float32)
        padded[:length] = motion
        return padded

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry = self.entries[index]
        motion = np.load(entry.path, allow_pickle=False)
        expected_shape = (entry.length, self.motion_dim)
        if motion.shape != expected_shape:
            raise ValueError(
                f"NPY motion has shape {motion.shape}, expected {expected_shape}: {entry.path}"
            )
        if motion.dtype != np.float32:
            raise ValueError(
                f"NPY motion has dtype {motion.dtype}, expected float32: {entry.path}"
            )
        if not np.isfinite(motion).all():
            raise ValueError(f"NPY motion contains non-finite values: {entry.path}")
        return self._normalize_and_pad(motion, entry.length), entry.fps, entry.length


def make_motion_dataset(
    root_path,
    meta_files,
    batch_size,
    num_frames,
    fps,
    motion_dim=366,
    normalize=False,
    stats_path=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
):
    dataset = MotionDataset(
        root_path=root_path,
        meta_files=meta_files,
        num_frames=num_frames,
        fps=fps,
        motion_dim=motion_dim,
        normalize=normalize,
        stats_path=stats_path,
    )
    if not dataset:
        raise ValueError(f"No valid MotionJEPA samples found under {root_path}")
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0 and persistent_workers),
    )
    return dataset, loader, sampler


__all__ = ["MotionDataset", "make_motion_dataset"]
