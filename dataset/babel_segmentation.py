"""Validated continuous-window, multi-label BABEL frame annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dataset.motion_dataset import MotionDataset


FORMAT = "babel_frame_segmentation_v1"
SUPERVISION_POLICY = "at_least_one_in_vocabulary_120_positive"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BabelSegmentationDataset(MotionDataset):
    """Return motion, FPS, length, padded labels, supervision and sample ID.

    Frame supervision is true only when at least one of the 120 action
    classes is annotated. Transition/OOV-only and uncovered frames remain
    available to the encoder but are excluded from segmentation loss/metrics.
    """

    def __init__(
        self, root: str | Path, split: str, *, num_frames: int = 150,
        fps: int = 30, motion_dim: int = 366, stats_root: str | Path | None = None,
        normalize: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown BABEL segmentation split: {split}")
        self.root = Path(root)
        self.split = split
        self.metadata = json.loads((self.root / "meta.json").read_text())
        if self.metadata.get("segmentation_format") != FORMAT:
            raise ValueError("Not a continuous BABEL frame segmentation dataset")
        if self.metadata.get("supervision_policy") != SUPERVISION_POLICY:
            raise ValueError("BABEL segmentation supervision policy mismatch")
        names = self.metadata.get("class_names", [])
        names60 = self.metadata.get("class_names_60", [])
        if (len(names) != 120 or len(set(names)) != 120 or len(names60) != 60
                or len(set(names60)) != 60 or not set(names60).issubset(names)):
            raise ValueError("Invalid BABEL 120/60 class-name mapping")
        self.class_names = tuple(names)
        self.class_indices_60 = tuple(names.index(name) for name in names60)
        if list(self.class_indices_60) != self.metadata.get("class_indices_60"):
            raise ValueError("Stored BABEL-60 class indices disagree with class names")
        self.num_classes = len(self.class_names)
        index_path = self.root / "index.json"
        if file_sha256(index_path) != self.metadata.get("index_sha256"):
            raise ValueError("BABEL segmentation index hash mismatch")
        records = json.loads(index_path.read_text())
        self.records_by_id = {}
        sources = {}
        for record in records:
            sample_id = record["id"]
            source = record["source_amass_relpath"]
            if sample_id in self.records_by_id:
                raise ValueError(f"Duplicate BABEL segmentation sample: {sample_id}")
            if source in sources and sources[source] != record["split"]:
                raise ValueError(f"BABEL source overlaps splits: {source}")
            sources[source] = record["split"]
            self.records_by_id[sample_id] = record
        super().__init__(
            root_path=self.root, meta_files=f"{split}.txt", num_frames=num_frames,
            fps=fps, motion_dim=motion_dim, normalize=normalize, stats_path=stats_root,
        )
        self.records = []
        for entry in self.entries:
            record = self.records_by_id.get(entry.sample_id)
            if (record is None or record["split"] != split
                    or self.root / record["motion_path"] != entry.path
                    or int(record["length"]) != entry.length):
                raise ValueError(f"Segmentation index disagrees with motion: {entry.sample_id}")
            relative = Path(record["labels_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("labels", split):
                raise ValueError(f"Unsafe segmentation label path: {relative}")
            self.records.append(record)
        if len(self.records) != sum(record["split"] == split for record in records):
            raise ValueError("Segmentation index contains samples absent from the split")

    def __getitem__(self, index: int):
        record = self.records[index]
        label_path = self.root / record["labels_path"]
        if file_sha256(label_path) != record["labels_sha256"]:
            raise ValueError(f"Segmentation label hash mismatch: {label_path}")
        if file_sha256(self.entries[index].path) != record["motion_sha256"]:
            raise ValueError(f"Segmentation motion hash mismatch: {self.entries[index].path}")
        motion, fps, length = super().__getitem__(index)
        with np.load(label_path, allow_pickle=False) as arrays:
            labels = arrays["labels"]
            supervision = arrays["supervision"]
        if labels.shape != (length, self.num_classes) or supervision.shape != (length,):
            raise ValueError(f"Invalid segmentation label shape: {label_path}")
        if labels.dtype != np.bool_ or supervision.dtype != np.bool_:
            raise ValueError(f"Segmentation labels must be boolean: {label_path}")
        if not np.array_equal(supervision, labels.any(axis=1)):
            raise ValueError(f"Segmentation supervision disagrees with labels: {label_path}")
        padded_labels = np.zeros((self.num_frames, self.num_classes), dtype=np.float32)
        padded_labels[:length] = labels
        padded_supervision = np.zeros(self.num_frames, dtype=np.bool_)
        padded_supervision[:length] = supervision
        return motion, fps, length, padded_labels, padded_supervision, record["id"]


__all__ = ["BabelSegmentationDataset", "FORMAT", "SUPERVISION_POLICY", "file_sha256"]
