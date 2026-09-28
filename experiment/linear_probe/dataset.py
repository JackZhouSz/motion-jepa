"""Single-label and multi-label motion classification datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from dataset import MotionDataset


DEFAULT_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SingleLabelIndex:
    """Stable mappings between class names, IDs, and sample IDs."""

    class_names: tuple[str, ...]
    class_to_index: dict[str, int]
    label_name_by_id: dict[str, str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def label_for_sample(self, sample_id: str) -> int:
        try:
            label_name = self.label_name_by_id[sample_id]
        except KeyError as error:
            raise KeyError(f"Sample ID is missing from the classification index: {sample_id}") from error
        return self.class_to_index[label_name]

    def to_json(self) -> dict[str, object]:
        return {
            "class_names": list(self.class_names),
            "class_to_index": dict(self.class_to_index),
        }


@dataclass(frozen=True)
class MultiLabelIndex:
    """Class IDs and all positive labels for each unique motion clip."""

    class_names: tuple[str, ...]
    class_to_index: dict[str, int]
    labels_by_path: dict[str, tuple[int, ...]]
    row_labels_by_path: dict[str, tuple[int, ...]]
    path_by_id: dict[str, str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def label_for_sample(self, sample_id: str) -> torch.Tensor:
        labels = self.labels_by_path[sample_id]
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[list(labels)] = 1.0
        return target

    def to_json(self) -> dict[str, object]:
        return {
            "class_names": list(self.class_names),
            "class_to_index": dict(self.class_to_index),
        }


def classification_dataset_kind(root: str | Path) -> tuple[str, str]:
    """Use the producer metadata, rather than a directory-name heuristic."""
    metadata = json.loads((Path(root) / "meta.json").read_text(encoding="utf-8"))
    source = str(metadata.get("source_dataset", ""))
    if source.startswith("BABEL-"):
        subset = metadata.get("subset")
        if subset not in (60, 120) or int(metadata.get("num_classes", -1)) != subset:
            raise ValueError("BABEL metadata must declare subset and matching num_classes")
        return "multilabel", f"babel-{subset}"
    if source.startswith("100STYLE"):
        return "single_label", "100style"
    dataset_name = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    return "single_label", dataset_name or "classification"


ClassificationLabelIndex = SingleLabelIndex | MultiLabelIndex


def _load_multilabel_label_index(root: str | Path) -> MultiLabelIndex:
    root = Path(root)
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    class_index = json.loads((root / "class-index.json").read_text(encoding="utf-8"))
    class_names = tuple(class_index["class_names"])
    if (
        len(class_names) != metadata.get("num_classes")
        or list(class_names) != metadata.get("class_names")
        or class_index.get("class_to_index") != {
            name: index for index, name in enumerate(class_names)
        }
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("BABEL class-index.json does not match meta.json")
    records = json.loads((root / "index.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("BABEL index.json is empty or malformed")
    row_labels: dict[str, list[int]] = {}
    path_by_id: dict[str, str] = {}
    path_split: dict[str, str] = {}
    for record in records:
        sample_id = record["id"]
        path = record["motion_path"]
        split = record["split"]
        label = record["metadata"]["label"]
        label_name = record["metadata"]["label_name"]
        if (
            not isinstance(sample_id, str) or not sample_id
            or not isinstance(path, str) or not path
            or split not in DEFAULT_SPLITS
            or isinstance(label, bool) or not isinstance(label, int)
            or not 0 <= label < len(class_names)
            or class_names[label] != label_name
            or (path in path_split and path_split[path] != split)
            or sample_id in path_by_id
        ):
            raise ValueError(f"Invalid or conflicting BABEL index row: {record!r}")
        path_by_id[sample_id] = path
        path_split[path] = split
        row_labels.setdefault(path, []).append(label)
    return MultiLabelIndex(
        class_names,
        {name: index for index, name in enumerate(class_names)},
        {path: tuple(sorted(set(labels))) for path, labels in row_labels.items()},
        {path: tuple(labels) for path, labels in row_labels.items()},
        path_by_id,
    )


def _load_single_label_index(dataset_root: str | Path) -> SingleLabelIndex:
    """Read named or explicit numeric single-label classes from ``index.json``."""
    root = Path(dataset_root)
    path = root / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Classification index does not exist: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Classification index is empty or malformed: {path}")

    label_name_by_id: dict[str, str] = {}
    explicit_labels: dict[int, str] = {}
    uses_explicit_labels: bool | None = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Index record must be an object: {record!r}")
        sample_id = record.get("id")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Index record has no valid metadata: {record}")
        has_explicit_label = "label" in metadata or "label_name" in metadata
        if uses_explicit_labels is None:
            uses_explicit_labels = has_explicit_label
        elif uses_explicit_labels != has_explicit_label:
            raise ValueError("index.json mixes named and explicit action labels")
        if has_explicit_label:
            label = metadata.get("label")
            label_name = metadata.get("label_name")
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise ValueError(f"Index record has no valid numeric label: {record}")
            if not isinstance(label_name, str) or not label_name:
                raise ValueError(f"Index record has no valid label_name: {record}")
            previous = explicit_labels.setdefault(label, label_name)
            if previous != label_name:
                raise ValueError(
                    f"Conflicting names for explicit label {label}: {previous!r}, {label_name!r}"
                )
        else:
            label = None
            label_name = metadata.get("style")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Index record has no valid sample ID: {record}")
        if not isinstance(label_name, str) or not label_name:
            raise ValueError(f"Index record has no valid class label: {record}")
        if sample_id in label_name_by_id:
            raise ValueError(f"Duplicate sample ID in index: {sample_id}")
        label_name_by_id[sample_id] = label_name

    if uses_explicit_labels:
        metadata_path = root / "meta.json"
        dataset_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        declared_names = dataset_metadata.get("class_names")
        if declared_names is not None:
            if not isinstance(declared_names, list) or not all(
                isinstance(name, str) and name for name in declared_names
            ):
                raise ValueError(f"Dataset class_names is malformed: {metadata_path}")
            class_names = tuple(declared_names)
            for label, name in explicit_labels.items():
                if label >= len(class_names) or class_names[label] != name:
                    raise ValueError(
                        f"Explicit label {label}={name!r} conflicts with dataset class_names"
                    )
        else:
            expected = set(range(len(explicit_labels)))
            if set(explicit_labels) != expected:
                raise ValueError("Explicit action labels must be contiguous from zero")
            class_names = tuple(explicit_labels[index] for index in range(len(explicit_labels)))
        class_to_index = {name: index for index, name in enumerate(class_names)}
        if len(class_to_index) != len(class_names):
            raise ValueError("Explicit action label names must be unique")
    else:
        class_names = tuple(sorted(set(label_name_by_id.values())))
        class_to_index = {name: index for index, name in enumerate(class_names)}
    return SingleLabelIndex(class_names, class_to_index, label_name_by_id)


def load_classification_label_index(root: str | Path) -> ClassificationLabelIndex:
    """Load the label index indicated by the dataset metadata."""
    root = Path(root)
    if (root / "meta.json").is_file() and classification_dataset_kind(root)[0] == "multilabel":
        return _load_multilabel_label_index(root)
    return _load_single_label_index(root)


class SingleLabelMotionDataset(Dataset):
    """Attach single class IDs and sample IDs to MotionDataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        num_frames: int,
        fps: int,
        motion_dim: int,
        stats_root: str | Path,
        label_index: SingleLabelIndex,
    ) -> None:
        self.motion = MotionDataset(
            root_path=root,
            meta_files=f"{split}.txt",
            num_frames=num_frames,
            fps=fps,
            motion_dim=motion_dim,
            normalize=True,
            stats_path=stats_root,
        )
        self.label_index = label_index
        self.sample_ids = [entry.sample_id for entry in self.motion.entries]
        missing = [
            sample_id
            for sample_id in self.sample_ids
            if sample_id not in label_index.label_name_by_id
        ]
        if missing:
            raise ValueError(
                f"Split {split!r} has sample IDs missing from index.json: {missing[:3]}"
            )
        self.labels = [label_index.label_for_sample(sample_id) for sample_id in self.sample_ids]

    def __len__(self) -> int:
        return len(self.motion)

    def __getitem__(self, index: int):
        motion, fps, length = self.motion[index]
        return motion, fps, length, self.labels[index], self.sample_ids[index]


class MultiLabelMotionDataset(Dataset):
    """Load each motion file once and attach its complete multi-hot target."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        num_frames: int,
        fps: int,
        motion_dim: int,
        stats_root: str | Path,
        label_index: MultiLabelIndex,
    ) -> None:
        self.motion = MotionDataset(
            root_path=root,
            meta_files=f"{split}.txt",
            num_frames=num_frames,
            fps=fps,
            motion_dim=motion_dim,
            normalize=True,
            stats_path=stats_root,
        )
        self.label_index = label_index
        self.entry_indices: list[int] = []
        self.sample_ids: list[str] = []
        self.labels: list[torch.Tensor] = []
        seen: set[str] = set()
        for index, entry in enumerate(self.motion.entries):
            path = label_index.path_by_id.get(entry.sample_id)
            if path is None or Path(root) / path != entry.path:
                raise ValueError(f"Multi-label split {split} has an unmatched index row: {entry.sample_id}")
            if path in seen:
                continue
            seen.add(path)
            self.entry_indices.append(index)
            self.sample_ids.append(path)
            self.labels.append(label_index.label_for_sample(path))

    def __len__(self) -> int:
        return len(self.entry_indices)

    def __getitem__(self, index: int):
        motion, fps, length = self.motion[self.entry_indices[index]]
        return motion, fps, length, self.labels[index], self.sample_ids[index]


class EmptyClassificationDataset(Dataset):
    """Represent an unavailable test split without borrowing validation data."""

    def __init__(self) -> None:
        self.labels: list[object] = []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError(index)


class ClassificationTokenDataset(Dataset):
    """Expose a validated cached JEPA token split through the motion batch API."""

    def __init__(
        self,
        payload: dict[str, object],
        *,
        label_index: ClassificationLabelIndex,
        fps: int,
    ) -> None:
        features = payload.get("features")
        lengths = payload.get("lengths")
        labels = payload.get("labels")
        sample_ids = payload.get("sample_ids")
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("Token cache features must have shape [N,T,D]")
        if features.dtype != torch.bfloat16:
            raise ValueError("Token cache features must use bfloat16")
        if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long:
            raise ValueError("Token cache lengths must be int64")
        is_multilabel = isinstance(label_index, MultiLabelIndex)
        expected_dtype = torch.float32 if is_multilabel else torch.long
        if not isinstance(labels, torch.Tensor) or labels.dtype != expected_dtype:
            raise ValueError(f"Token cache labels must use {expected_dtype}")
        if not isinstance(sample_ids, list) or not all(
            isinstance(sample_id, str) for sample_id in sample_ids
        ):
            raise ValueError("Token cache sample_ids must be a list of strings")
        count = len(features)
        if len(lengths) != count or len(labels) != count or len(sample_ids) != count:
            raise ValueError("Token cache fields have inconsistent sample counts")
        if (lengths < 0).any() or (lengths > features.shape[1]).any():
            raise ValueError("Token cache contains invalid sequence lengths")
        if is_multilabel:
            expected_labels = (
                torch.stack([label_index.label_for_sample(sample_id) for sample_id in sample_ids])
                if sample_ids else torch.empty((0, label_index.num_classes), dtype=torch.float32)
            )
        else:
            expected_labels = torch.tensor(
                [label_index.label_for_sample(sample_id) for sample_id in sample_ids],
                dtype=torch.long,
            )
        if not torch.equal(labels, expected_labels):
            raise ValueError("Token cache labels do not match the classification index")
        if not torch.isfinite(features).all():
            raise ValueError("Token cache contains non-finite features")
        self.features = features
        self.lengths = lengths
        self.labels = labels
        self.sample_ids = sample_ids
        self.fps = int(fps)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.fps,
            self.lengths[index],
            self.labels[index],
            self.sample_ids[index],
        )


def build_classification_datasets(
    root: str | Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    num_frames: int,
    fps: int,
    motion_dim: int,
    stats_root: str | Path,
    label_index: ClassificationLabelIndex | None = None,
) -> tuple[dict[str, Dataset], ClassificationLabelIndex]:
    task, _ = classification_dataset_kind(root)
    root = Path(root)
    requested = tuple(splits)
    available = []
    for split in requested:
        txt_exists = (root / f"{split}.txt").is_file()
        manifest_exists = (root / "motions" / f"{split}.json").is_file()
        if split == "test" and not txt_exists and not manifest_exists:
            continue
        available.append(split)
    resolved = label_index or load_classification_label_index(root)
    if task == "single_label":
        if not isinstance(resolved, SingleLabelIndex):
            raise TypeError("Single-label classification requires a SingleLabelIndex")
        dataset_type = SingleLabelMotionDataset
    else:
        if not isinstance(resolved, MultiLabelIndex):
            raise TypeError("Multi-label classification requires a MultiLabelIndex")
        dataset_type = MultiLabelMotionDataset
    datasets = {
        split: dataset_type(
            root, split, num_frames=num_frames, fps=fps, motion_dim=motion_dim,
            stats_root=stats_root, label_index=resolved,
        )
        for split in available
    }
    return {
        split: datasets[split] if split in datasets else EmptyClassificationDataset()
        for split in requested
    }, resolved


__all__ = [
    "ClassificationLabelIndex",
    "ClassificationTokenDataset",
    "DEFAULT_SPLITS",
    "EmptyClassificationDataset",
    "MultiLabelIndex",
    "MultiLabelMotionDataset",
    "SingleLabelIndex",
    "SingleLabelMotionDataset",
    "build_classification_datasets",
    "classification_dataset_kind",
    "load_classification_label_index",
]
