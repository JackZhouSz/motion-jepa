"""Temporal BONES caption joins and lazy, cache-only alignment datasets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import MotionDataset
from experiment.linear_probe.features import _sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset/bones-seed-processed-nframes150"
DEFAULT_ANNOTATIONS_PATH = (
    PROJECT_ROOT / "dataset/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl"
)
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "output/tmr-cache"
DEFAULT_TEXT_MODEL = "google/t5gemma-2b-2b-ul2"
SPLITS = ("train", "val", "test")
CAPTION_POLICY = "floor_halfopen_overlap_chronological_unique_candidates_v2"


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def collect_caption_candidates(
    events: Sequence[dict[str, Any]], start_frame: int, end_frame: int, fps: int
) -> tuple[list[str], list[int]]:
    """Keep each distinct overlapping event description as a separate candidate."""
    selected = []
    for index, event in enumerate(events):
        start = float(event["start_time"])
        end = float(event["end_time"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid temporal event boundaries: {event!r}")
        first, last = math.floor(start * fps), math.floor(end * fps)
        if min(end_frame, last) > max(start_frame, first):
            description = " ".join(str(event.get("description", "")).split())
            if description:
                selected.append((start, end, index, description))
    selected.sort(key=lambda item: item[:3])
    descriptions: list[str] = []
    indices = []
    for _, _, index, description in selected:
        indices.append(index)
        if description not in descriptions:
            descriptions.append(description)
    return descriptions, indices


def build_paired_index(
    dataset_root: str | Path,
    annotations_path: str | Path,
    max_samples_per_split: int | None = None,
) -> dict[str, Any]:
    """Derive captions without changing the producer's motions or split files."""
    root = Path(dataset_root).expanduser().resolve()
    annotations = Path(annotations_path).expanduser().resolve()
    if max_samples_per_split is not None and max_samples_per_split < 1:
        raise ValueError("max_samples_per_split must be positive")
    metadata = json.loads((root / "meta.json").read_text())
    fps, num_frames = int(metadata["fps"]), int(metadata["num_frames"])
    if metadata.get("representation") != MotionDataset.REPRESENTATION:
        raise ValueError("TMR requires a MotionJEPA raw representation dataset")
    events_by_filename = {}
    with annotations.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            filename, events = item.get("filename"), item.get("events")
            if not isinstance(filename, str) or not isinstance(events, list):
                raise ValueError(f"Malformed temporal annotation at line {line_number}")
            if filename in events_by_filename:
                raise ValueError(f"Duplicate temporal annotation filename: {filename}")
            events_by_filename[filename] = events
    original = json.loads((root / "index.json").read_text())
    records, catalog = [], {}
    counts = {split: 0 for split in SPLITS}
    skipped = {split: 0 for split in SPLITS}
    seen = set()
    for record in original:
        split, sample_id = record["split"], record["id"]
        if split not in SPLITS or sample_id in seen:
            raise ValueError(f"Invalid or duplicate paired sample: {sample_id}")
        seen.add(sample_id)
        source_id = str(record["source_id"])
        filename = Path(source_id).name
        if filename not in events_by_filename:
            raise ValueError(f"Missing temporal annotations for source {source_id}")
        first, last = int(record["start_frame"]), int(record["end_frame"])
        length = int(record["length"])
        if int(record["fps"]) != fps or first < 0 or last - first != length:
            raise ValueError(f"Inconsistent interval/FPS for paired sample {sample_id}")
        relative = Path(record["motion_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe motion path: {relative}")
        captions, indices = collect_caption_candidates(events_by_filename[filename], first, last, fps)
        if not captions:
            skipped[split] += 1
            continue
        if max_samples_per_split is not None and counts[split] >= max_samples_per_split:
            continue
        caption_ids = []
        for caption in captions:
            caption_id = hashlib.sha256(caption.encode()).hexdigest()
            catalog[caption_id] = caption
            caption_ids.append(caption_id)
        records.append({
            "sample_id": sample_id, "motion_path": relative.as_posix(),
            "source_id": source_id, "split": split, "fps": fps, "length": length,
            "start_frame": first, "end_frame": last,
            "caption_id": caption_ids[0], "caption_ids": caption_ids, "event_indices": indices,
        })
        counts[split] += 1
    # Sorting makes caption cache rows independent of incidental index traversal.
    catalog = dict(sorted(catalog.items()))
    provenance = {
        "dataset_root": str(root), "annotations_path": str(annotations),
        "dataset_meta_sha256": _sha256_file(root / "meta.json"),
        "dataset_index_sha256": _sha256_file(root / "index.json"),
        "annotations_sha256": _sha256_file(annotations),
        "split_files": {
            str(path.relative_to(root)): _sha256_file(path)
            for split in SPLITS
            for path in (root / f"{split}.txt", root / "motions" / f"{split}.json")
        },
        "caption_policy": CAPTION_POLICY,
        "max_samples_per_split": max_samples_per_split,
        "fps": fps, "num_frames": num_frames,
        "raw_motion_dim": int(metadata["motion_dim"]),
    }
    return {
        "records": records, "catalog": catalog, "provenance": provenance,
        "paired_index_sha256": json_digest(records), "catalog_sha256": json_digest(catalog),
        "split_counts": counts, "filtered_counts": skipped,
    }


class RaggedTokenBank:
    """Memory-map BF16 bit patterns; convert only the requested sequence to FP32."""

    def __init__(self, root: str | Path, expected: dict[str, Any] | None = None):
        self.root = Path(root)
        marker = self.root / "complete.json"
        if not marker.is_file():
            raise ValueError(f"Incomplete token cache: {self.root}; run prepare_cache")
        self.metadata = json.loads(marker.read_text())
        if expected is not None and self.metadata.get("signature") != expected:
            raise ValueError("Token cache metadata is stale; use --recompute-features")
        self.keys = self.metadata["keys"]
        if not isinstance(self.keys, list) or len(set(self.keys)) != len(self.keys):
            raise ValueError("Token cache keys must be unique")
        self.key_to_row = {key: index for index, key in enumerate(self.keys)}
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r", allow_pickle=False)
        values = np.load(self.root / "values.npy", mmap_mode="r", allow_pickle=False)
        if self.offsets.dtype != np.int64 or self.offsets.shape != (len(self.keys) + 1,):
            raise ValueError("Invalid ragged token-cache offsets")
        if (
            self.offsets[0] != 0 or (np.diff(self.offsets) <= 0).any()
            or values.dtype != np.uint16 or values.ndim != 2
            or values.shape != (int(self.offsets[-1]), int(self.metadata["feature_dim"]))
        ):
            raise ValueError("Token cache has invalid lengths, shape, or BF16 storage")
        self._values = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_values"] = None
        return state

    def __getitem__(self, key: str) -> torch.Tensor:
        if self._values is None:
            self._values = np.load(self.root / "values.npy", mmap_mode="r", allow_pickle=False)
        row = self.key_to_row[key]
        bits = np.array(self._values[int(self.offsets[row]):int(self.offsets[row + 1])], copy=True)
        tokens = torch.from_numpy(bits).view(torch.bfloat16).float()
        if not torch.isfinite(tokens).all():
            raise ValueError(f"Token cache contains non-finite values for {key}")
        return tokens


class PreparedPairDataset(Dataset):
    def __init__(
        self, records, text_bank, *, motion_bank=None, raw_dataset=None,
        sample_captions=False, motion_mean=None, motion_std=None,
    ):
        self.records = records
        self.text_bank = text_bank
        self.motion_bank = motion_bank
        self.raw_dataset = raw_dataset
        self.sample_captions = sample_captions
        if (motion_mean is None) != (motion_std is None):
            raise ValueError("Motion normalization requires both mean and std")
        self.motion_mean = motion_mean
        self.motion_std = motion_std
        if motion_mean is not None:
            dim = int(motion_bank.metadata["feature_dim"]) if motion_bank is not None else None
            if (
                motion_mean.shape != (dim,) or motion_std.shape != (dim,)
                or not torch.isfinite(motion_mean).all() or not torch.isfinite(motion_std).all()
                or (motion_std <= 0).any()
            ):
                raise ValueError("Motion normalization must match the JEPA feature channels")
        self.sample_ids = [record["sample_id"] for record in records]
        self.caption_ids = [record["caption_id"] for record in records]
        self.caption_candidates = [record["caption_ids"] for record in records]
        self.raw_rows = (
            {entry.sample_id: row for row, entry in enumerate(raw_dataset.entries)}
            if raw_dataset is not None else {}
        )
        for record in records:
            candidates = record["caption_ids"]
            if not candidates or len(set(candidates)) != len(candidates) or record["caption_id"] != candidates[0]:
                raise ValueError("Paired captions must be a nonempty list of unique candidates")
            if any(caption_id not in text_bank.key_to_row for caption_id in candidates):
                raise ValueError("Paired caption is missing from the text cache")
            if motion_bank is not None and record["sample_id"] not in motion_bank.key_to_row:
                raise ValueError("Paired sample is missing from the JEPA cache")
            if raw_dataset is not None and record["sample_id"] not in self.raw_rows:
                raise ValueError("Paired sample is missing from the raw split index")
            if raw_dataset is not None:
                entry = raw_dataset.entries[self.raw_rows[record["sample_id"]]]
                if (
                    entry.length != record["length"] or entry.fps != record["fps"]
                    or entry.path != raw_dataset.root_path / record["motion_path"]
                ):
                    raise ValueError(f"Paired interval/path disagrees with the raw split index: {record['sample_id']}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        candidates = record["caption_ids"]
        selected = int(torch.randint(len(candidates), ()).item()) if self.sample_captions and len(candidates) > 1 else 0
        caption_id = candidates[selected]
        if self.motion_bank is not None:
            motion = self.motion_bank[record["sample_id"]]
            if self.motion_mean is not None:
                motion = (motion - self.motion_mean) / self.motion_std
        else:
            values, _, length = self.raw_dataset[self.raw_rows[record["sample_id"]]]
            motion = torch.from_numpy(values[:length].copy())
        return {
            "motion_tokens": motion, "text_tokens": self.text_bank[caption_id],
            "caption_id": caption_id, "caption_candidate_ids": candidates,
            **{key: record[key] for key in (
                "sample_id", "source_id", "start_frame", "end_frame"
            )},
        }


def collate_pairs(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    result = {}
    for name in ("motion", "text"):
        tokens = [torch.as_tensor(sample[f"{name}_tokens"], dtype=torch.float32) for sample in samples]
        dim = tokens[0].shape[-1]
        if any(value.ndim != 2 or len(value) < 1 or value.shape[1] != dim for value in tokens):
            raise ValueError(f"{name} token sequences must be nonempty [length,dim]")
        length = max(map(len, tokens))
        padded = torch.zeros((len(tokens), length, dim), dtype=torch.float32)
        mask = torch.zeros((len(tokens), length), dtype=torch.bool)
        for row, value in enumerate(tokens):
            padded[row, :len(value)] = value
            mask[row, :len(value)] = True
        result[f"{name}_tokens"], result[f"{name}_mask"] = padded, mask
    for singular, plural in (("caption_id", "caption_ids"), ("sample_id", "sample_ids"), ("source_id", "source_ids")):
        result[plural] = [sample[singular] for sample in samples]
    result["caption_candidate_ids"] = [sample.get("caption_candidate_ids", [sample["caption_id"]]) for sample in samples]
    for singular, plural in (("start_frame", "start_frames"), ("end_frame", "end_frames")):
        result[plural] = torch.tensor([sample[singular] for sample in samples], dtype=torch.long)
    return result
