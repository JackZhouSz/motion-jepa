"""BONES-SEED v2: retain short motions and end-align the final full window.

NPY arrays contain actual valid frames; MotionDataset pads them when loading.
The original preprocessing module and its complete-window protocol are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_rep import MotionJEPAMotionRep  # noqa: E402
from skeleton import SOMASkeleton30, SOMASkeleton77, parse_bvh_motion  # noqa: E402


TRAIN_SPLIT_FILE = "train_split_paths.txt"
HELDOUT_SPLIT_FILES = (
    "test_content_split_paths.txt",
    "test_repetition_split_paths.txt",
)
SPLITS = ("train", "val", "test")
NPY_FORMAT = "motion_jepa_npy_v1"
PREPROCESSING_VERSION = "bones_seed_v2"
SEGMENTATION_POLICY = "overlapping_full_windows_end_aligned_tail_short_sources"

_DATASET_ROOT: Path | None = None
_NUM_FRAMES = 150
_MIN_FRAMES = 60
_FPS = 30
_STRIDE_FRAMES = 75
_SOURCE_SKELETON: SOMASkeleton77 | None = None
_TARGET_SKELETON: SOMASkeleton30 | None = None
_OUTPUT_ROOT: Path | None = None
_THREAD_CONFIG_PID: int | None = None


@dataclass(frozen=True)
class WorkItem:
    id: str
    split: str
    bvh_path: str
    metadata: dict[str, Any]


def round_fps(fps: float) -> int:
    """Round a positive frame rate using the unambiguous half-up rule."""
    rounded = int(math.floor(float(fps) + 0.5))
    if rounded <= 0:
        raise ValueError(f"Invalid BVH frame rate: {fps}")
    return rounded


def resample_motion_fps(
    local_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    source_fps: int,
    target_fps: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert to the configured FPS using frame-zero-aligned step indexing."""
    source_fps = int(source_fps)
    target_fps = int(target_fps)
    if source_fps < target_fps:
        raise ValueError(
            "Rounded BVH frame rate is below configured FPS: "
            f"{source_fps} < {target_fps}"
        )
    if source_fps % target_fps != 0:
        raise ValueError(
            "Rounded BVH frame rate must be an exact multiple of configured "
            f"FPS {target_fps}, got {source_fps}"
        )
    step = source_fps // target_fps
    if step == 1:
        return local_rotations, root_positions, target_fps
    return local_rotations[::step], root_positions[::step], target_fps


def calculate_stride(num_frames: int, overlap: float) -> int:
    """Return a positive stride using half-up integer rounding."""
    return max(1, int(math.floor(int(num_frames) * (1.0 - float(overlap)) + 0.5)))


def _validate_config(args: argparse.Namespace) -> None:
    if args.num_frames <= 0:
        raise ValueError(f"--num_frames must be positive, got {args.num_frames}")
    if not 1 <= args.min_frames <= args.num_frames:
        raise ValueError(
            "--min_frames must be in [1, --num_frames], "
            f"got {args.min_frames} with num_frames={args.num_frames}"
        )
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError(f"--overlap must be in [0, 1), got {args.overlap}")


def select_windows(
    source_frames: int,
    num_frames: int,
    min_frames: int,
    stride_frames: int,
) -> list[tuple[int, int]]:
    """Keep eligible short sources, plus full windows covering long-source ends."""
    if not 1 <= min_frames <= num_frames or not 1 <= stride_frames <= num_frames:
        raise ValueError("Invalid minimum, maximum, or stride for window selection")
    if source_frames < min_frames:
        return []
    if source_frames <= num_frames:
        return [(0, source_frames)]
    starts = list(range(0, source_frames - num_frames + 1, stride_frames))
    final_start = source_frames - num_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + num_frames) for start in starts]


def _read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        return [line.strip().replace("\\", "/") for line in file if line.strip()]


def _unique_preserving_order(values: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(values))


def build_splits(splits_root: Path, seed: int) -> dict[str, list[str]]:
    train = _unique_preserving_order(_read_ids(splits_root / TRAIN_SPLIT_FILE))
    train_set = set(train)
    heldout: list[str] = []
    for filename in HELDOUT_SPLIT_FILES:
        heldout.extend(_read_ids(splits_root / filename))
    heldout = sorted(set(heldout).difference(train_set))
    random.Random(seed).shuffle(heldout)
    num_val = len(heldout) // 3
    return {"train": train, "val": heldout[:num_val], "test": heldout[num_val:]}


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return by_id
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            raw_path = (row.get("move_soma_uniform_path") or "").replace("\\", "/")
            marker = "soma_uniform/bvh/"
            if marker in raw_path:
                item_id = raw_path.split(marker, 1)[1]
                if item_id.endswith(".bvh"):
                    item_id = item_id[:-4]
                by_id[item_id] = row
    return by_id


def _make_work_items(
    split_ids: dict[str, list[str]],
    metadata_by_id: dict[str, dict[str, str]],
    max_per_split: int | None,
) -> dict[str, list[WorkItem]]:
    output = {split: [] for split in SPLITS}
    metadata_fields = (
        "move_name",
        "filename",
        "package",
        "category",
        "is_neutral",
        "is_mirror",
        "take_actor",
        "take_date",
        "content_name",
    )
    for split in SPLITS:
        ids = split_ids[split]
        selected_ids = ids if max_per_split is None else ids[:max_per_split]
        for item_id in selected_ids:
            row = metadata_by_id.get(item_id, {})
            metadata = {
                key: row[key]
                for key in metadata_fields
                if row.get(key) not in (None, "")
            }
            output[split].append(
                WorkItem(
                    id=item_id,
                    split=split,
                    bvh_path=f"bvh/{item_id}.bvh",
                    metadata=metadata,
                )
            )
    return output


def _init_worker(
    dataset_root: str,
    output_root: str,
    num_frames: int,
    fps: int,
    stride_frames: int,
    min_frames: int = 60,
) -> None:
    global _DATASET_ROOT, _OUTPUT_ROOT, _NUM_FRAMES, _MIN_FRAMES, _FPS, _STRIDE_FRAMES
    global _SOURCE_SKELETON, _TARGET_SKELETON, _THREAD_CONFIG_PID
    _DATASET_ROOT = Path(dataset_root)
    _OUTPUT_ROOT = Path(output_root)
    _NUM_FRAMES = int(num_frames)
    _MIN_FRAMES = int(min_frames)
    _FPS = int(fps)
    _STRIDE_FRAMES = int(stride_frames)
    current_pid = os.getpid()
    if _THREAD_CONFIG_PID != current_pid:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _THREAD_CONFIG_PID = current_pid
    _SOURCE_SKELETON = SOMASkeleton77()
    _TARGET_SKELETON = SOMASkeleton30()


def _segment_identity(item: WorkItem, segment_index: int, segmented: bool) -> str:
    return f"{item.id}_{segment_index:04d}" if segmented else item.id


def _convert_one(item: WorkItem) -> dict[str, Any]:
    """Convert one source motion and return raw clip arrays to the parent."""
    assert _DATASET_ROOT is not None
    assert _SOURCE_SKELETON is not None and _TARGET_SKELETON is not None
    source_path = _DATASET_ROOT / item.bvh_path
    try:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        local_rotations, root_positions, parsed_fps = parse_bvh_motion(source_path)
        source_fps = round_fps(float(parsed_fps))
        local_rotations, root_positions, fps = resample_motion_fps(
            local_rotations, root_positions, source_fps, _FPS
        )
        num_source_frames = len(local_rotations)
        clips = select_windows(
            num_source_frames, _NUM_FRAMES, _MIN_FRAMES, _STRIDE_FRAMES
        )
        if not clips:
            return {
                "ok": True,
                "id": item.id,
                "split": item.split,
                "records": [],
                "source_length": num_source_frames,
                "excluded_reason": "below_min_frames",
            }
        local_rotations, _ = _SOURCE_SKELETON.to_standard_tpose(local_rotations)
        local_rotations = _TARGET_SKELETON.from_soma77(local_rotations)
        motion_rep = MotionJEPAMotionRep(_TARGET_SKELETON, fps)
        segmented = num_source_frames > _NUM_FRAMES
        records: list[dict[str, Any]] = []
        for segment_index, (start_frame, end_frame) in enumerate(clips):
            features = motion_rep(
                local_rotations[start_frame:end_frame],
                root_positions[start_frame:end_frame],
                to_canonicalize=True,
            )
            motion = np.ascontiguousarray(
                features.detach().cpu().numpy(), dtype=np.float32
            )
            expected = (end_frame - start_frame, MotionJEPAMotionRep.FEATURE_DIM)
            if motion.shape != expected:
                raise ValueError(f"Unexpected feature shape {motion.shape}, expected {expected}")
            if not np.isfinite(motion).all():
                raise ValueError("Converted feature array contains non-finite values")
            records.append(
                {
                    "id": _segment_identity(item, segment_index, segmented),
                    "source_id": item.id,
                    "segment_index": segment_index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "split": item.split,
                    "source_path": item.bvh_path,
                    "source_fps": source_fps,
                    "fps": fps,
                    "length": int(len(motion)),
                    "motion_dim": int(motion.shape[1]),
                    "metadata": item.metadata,
                    "motion": motion,
                }
            )
        return {
            "ok": True,
            "id": item.id,
            "split": item.split,
            "records": records,
            "source_length": num_source_frames,
        }
    except Exception as error:
        return {
            "ok": False,
            "id": item.id,
            "split": item.split,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }


def _convert_batch(items: list[WorkItem]) -> list[dict[str, Any]]:
    return [_save_converted_result(_convert_one(item)) for item in items]


def _motion_relative_path(record: dict[str, Any]) -> Path:
    sample_id = Path(str(record["id"]))
    if sample_id.is_absolute() or ".." in sample_id.parts:
        raise ValueError(f"Unsafe motion sample id: {record['id']!r}")
    return Path("motions") / str(record["split"]) / sample_id.parent / f"{sample_id.name}.npy"


def _save_record_motion(record: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Save one converted array and retain only compact statistics in memory."""
    motion = np.ascontiguousarray(record.pop("motion"), dtype=np.float32)
    relative_path = _motion_relative_path(record)
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as file:
        np.save(file, motion, allow_pickle=False)
    record["motion_path"] = relative_path.as_posix()
    if record["split"] == "train":
        motion64 = motion.astype(np.float64, copy=False)
        record["_stats_sum"] = motion64.sum(axis=0)
        record["_stats_sq_sum"] = np.square(motion64).sum(axis=0)
    return record


def _save_converted_result(result: dict[str, Any]) -> dict[str, Any]:
    if not result["ok"]:
        return result
    assert _OUTPUT_ROOT is not None
    try:
        result["records"] = [
            _save_record_motion(record, _OUTPUT_ROOT) for record in result["records"]
        ]
        return result
    except Exception as error:
        return {
            "ok": False,
            "id": result["id"],
            "split": result["split"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }


def _ordered_results(items: list[WorkItem], args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Yield bounded, input-ordered worker results for deterministic manifests."""
    if not items:
        return
    init_args = (
        str(args.dataset_root),
        str(args.output),
        args.num_frames,
        args.fps,
        calculate_stride(args.num_frames, args.overlap),
        args.min_frames,
    )
    worker_count = min(max(1, int(args.workers)), len(items))
    if worker_count == 1:
        _init_worker(*init_args)
        yield from (_save_converted_result(_convert_one(item)) for item in items)
        return
    chunk_size = max(1, int(args.chunksize))
    batches = [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]
    # Keep at most two batches per worker in flight. Pool.imap eagerly queues
    # its whole iterable, which can otherwise accumulate large out-of-order
    # result records behind one slow source motion.
    with Pool(worker_count, initializer=_init_worker, initargs=init_args) as pool:
        pending = []
        next_batch = 0
        while next_batch < len(batches) and len(pending) < worker_count * 2:
            pending.append(pool.apply_async(_convert_batch, (batches[next_batch],)))
            next_batch += 1
        while pending:
            result = pending.pop(0)
            yield from result.get()
            if next_batch < len(batches):
                pending.append(pool.apply_async(_convert_batch, (batches[next_batch],)))
                next_batch += 1


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_split(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            f"{record['id']},{record['motion_path']},{record['fps']},{record['length']}\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _manifest(
    root: Path,
    split: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_path = root / f"{split}.txt"
    return {
        "format": NPY_FORMAT,
        "split": split,
        "motion_root": f"motions/{split}",
        "num_samples": len(records),
        "dtype": "float32",
        "representation": MotionJEPAMotionRep.FORMAT_NAME,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "min_frames": args.min_frames,
        "preprocessing_version": PREPROCESSING_VERSION,
        "segmentation": SEGMENTATION_POLICY,
        "motion_dim": MotionJEPAMotionRep.FEATURE_DIM,
        "values": "npy_valid_frames_row_major",
        "split_index": f"{split}.txt",
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Record inputs and options that determine dataset contents and statistics."""
    return {
        "version": PREPROCESSING_VERSION,
        "segmentation": SEGMENTATION_POLICY,
        "dataset_root": str(args.dataset_root),
        "num_frames": args.num_frames,
        "min_frames": args.min_frames,
        "fps": args.fps,
        "overlap": args.overlap,
        "stride_frames": calculate_stride(args.num_frames, args.overlap),
        "split_seed": args.split_seed,
        "max_per_split": args.max_per_split,
        "metadata_sha256": _file_sha256(args.metadata_csv),
        "split_sha256": {
            name: _file_sha256(args.splits_root / name)
            for name in (TRAIN_SPLIT_FILE, *HELDOUT_SPLIT_FILES)
        },
    }


def _protocol_sha256(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_complete_dataset(output: Path) -> bool:
    """Return whether output is a complete, internally consistent NPY dataset."""
    try:
        metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        if metadata.get("motion_storage") != "npy_float32_v1":
            return False
        if metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
            return False
        if metadata.get("segmentation") != SEGMENTATION_POLICY:
            return False
        if not 1 <= int(metadata.get("min_frames", 0)) <= int(metadata["num_frames"]):
            return False
        protocol = metadata.get("preprocessing_protocol")
        if not isinstance(protocol, dict):
            return False
        if metadata.get("preprocessing_protocol_sha256") != _protocol_sha256(protocol):
            return False
        if protocol.get("version") != metadata["preprocessing_version"]:
            return False
        for key in ("segmentation", "num_frames", "min_frames", "fps", "overlap", "stride_frames", "split_seed"):
            if protocol.get(key) != metadata.get(key):
                return False
        if not (output / "index.json").is_file() or not (output / "errors.jsonl").is_file():
            return False
        mean_path = output / "stats/mean.npy"
        std_path = output / "stats/std.npy"
        if not mean_path.is_file() or not std_path.is_file():
            return False
        mean = np.load(mean_path, allow_pickle=False)
        std = np.load(std_path, allow_pickle=False)
        expected_shape = (int(metadata.get("motion_dim", -1)),)
        if mean.shape != expected_shape or std.shape != expected_shape:
            return False
        if mean.dtype != np.float32 or std.dtype != np.float32:
            return False
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            return False
        if np.any(std <= 0):
            return False
        total = 0
        for split in SPLITS:
            split_path = output / f"{split}.txt"
            manifest_path = output / f"motions/{split}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != NPY_FORMAT:
                return False
            for key in ("num_frames", "min_frames", "preprocessing_version", "segmentation"):
                if manifest.get(key) != metadata.get(key):
                    return False
            if manifest.get("split_sha256") != hashlib.sha256(split_path.read_bytes()).hexdigest():
                return False
            rows = [line for line in split_path.read_text(encoding="utf-8").splitlines() if line]
            if len(rows) != int(manifest["num_samples"]):
                return False
            total += int(manifest["num_samples"])
        return total == int(metadata.get("num_samples", -1))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _prepare_output(output: Path, overwrite: bool) -> None:
    """Create the final output directory before conversion starts."""
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite to rebuild")
        if not output.is_dir():
            raise RuntimeError(f"Refusing to replace non-directory output: {output}")
        # The resolved output must be a named directory, never a filesystem root.
        if output == Path(output.anchor) or output.name in {"", ".", ".."}:
            raise RuntimeError(f"Refusing to remove unsafe output path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "motions").mkdir()


def finalize_processed_dataset(
    args: argparse.Namespace,
    records_by_split: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    source_success_counts: dict[str, int],
    *,
    source_dataset: str,
    segmentation: str,
    num_source_motions: int | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Write manifests, training statistics, and metadata for converted records."""
    for split in SPLITS:
        records = records_by_split.setdefault(split, [])
        _write_split(args.output / f"{split}.txt", records)
        _write_json(
            args.output / f"motions/{split}.json",
            _manifest(args.output, split, records, args),
        )

    train_count = 0
    train_sum: np.ndarray | None = None
    train_sq_sum: np.ndarray | None = None
    for record in records_by_split["train"]:
        clip_sum = record.pop("_stats_sum")
        clip_sq_sum = record.pop("_stats_sq_sum")
        train_count += int(record["length"])
        train_sum = clip_sum if train_sum is None else train_sum + clip_sum
        train_sq_sum = clip_sq_sum if train_sq_sum is None else train_sq_sum + clip_sq_sum

    if train_count == 0 or train_sum is None or train_sq_sum is None:
        raise RuntimeError("No training motions were converted; statistics cannot be computed.")
    mean = train_sum / train_count
    variance = np.maximum(train_sq_sum / train_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)
    stats_root = args.output / "stats"
    stats_root.mkdir(parents=True)
    np.save(stats_root / "mean.npy", mean.astype(np.float32))
    np.save(stats_root / "std.npy", std.astype(np.float32))

    all_records = [record for split in SPLITS for record in records_by_split[split]]
    _write_json(args.output / "index.json", all_records)
    metadata = {
        "source_dataset": source_dataset,
        "representation": MotionJEPAMotionRep.FORMAT_NAME,
        "producer": "motion-jepa",
        "motion_storage": "npy_float32_v1",
        "split_format": "sample_id,relative_npy_path,fps,actual_length",
        "skeleton": "soma30",
        "motion_dim": MotionJEPAMotionRep.FEATURE_DIM,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "min_frames": args.min_frames,
        "min_seconds": args.min_frames / args.fps,
        "preprocessing_version": PREPROCESSING_VERSION,
        "clip_seconds": args.num_frames / args.fps,
        "overlap": args.overlap,
        "stride_frames": calculate_stride(args.num_frames, args.overlap),
        "segmentation": segmentation,
        "window_canonicalization": "independent",
        "downsampling": "fixed_step_from_frame_zero_no_interpolation",
        "fps_validation": "source_fps_must_equal_or_be_divisible_by_configured_fps",
        "resampled": True,
        "canonicalized": True,
        "write_mode": "direct_on_the_fly",
        "split_seed": args.split_seed,
        "npy": {
            "format": NPY_FORMAT,
            "path": "motions",
            "dtype": "float32",
            "lazy_loading": True,
        },
        "statistics": {
            "format": "numpy_float32_pair",
            "mean_path": "stats/mean.npy",
            "std_path": "stats/std.npy",
            "source_split": "train",
            "valid_frames_only": True,
        },
        "split_counts": {split: len(records_by_split[split]) for split in SPLITS},
        "source_split_counts": source_success_counts,
        "num_source_motions": (
            sum(source_success_counts.values())
            if num_source_motions is None
            else int(num_source_motions)
        ),
        "num_samples": len(all_records),
        "num_errors": len(errors),
        "train_stats_frames": int(train_count),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    _write_json(args.output / "meta.json", metadata)
    with (args.output / "errors.jsonl").open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, ensure_ascii=False) + "\n")

    if not _validate_complete_dataset(args.output):
        raise RuntimeError(f"NPY dataset failed final validation: {args.output}")
    print(f"Output: {args.output}")
    print(
        "Splits: "
        + ", ".join(f"{split}={len(records_by_split[split])}" for split in SPLITS)
    )
    print(f"Errors: {len(errors)}")


def preprocess(args: argparse.Namespace) -> None:
    """Write a complete NPY-backed dataset directly to its final directory."""
    _validate_config(args)
    args.dataset_root = Path(args.dataset_root).resolve()
    args.splits_root = Path(args.splits_root).resolve()
    args.metadata_csv = Path(args.metadata_csv).resolve()
    args.output = Path(args.output).resolve()
    protocol = preprocessing_protocol(args)
    if args.output.exists() and _validate_complete_dataset(args.output) and not args.overwrite:
        existing = json.loads((args.output / "meta.json").read_text(encoding="utf-8"))
        if existing["preprocessing_protocol"] != protocol:
            raise ValueError(
                "Existing dataset preprocessing protocol does not match the requested "
                f"configuration or inputs: {args.output}; use --overwrite or a new --output"
            )
        print(f"Reusing complete NPY dataset: {args.output}")
        return
    _prepare_output(args.output, args.overwrite)
    split_ids = build_splits(args.splits_root, args.split_seed)
    metadata = _load_metadata(args.metadata_csv)
    limit = None if args.max_per_split < 0 else args.max_per_split
    items_by_split = _make_work_items(split_ids, metadata, limit)
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    source_success_counts = {split: 0 for split in SPLITS}
    retained_source_counts = {split: 0 for split in SPLITS}
    excluded_short_source_counts = {split: 0 for split in SPLITS}
    excluded_short_frame_counts = {split: 0 for split in SPLITS}

    for split in SPLITS:
        items = items_by_split[split]
        split_records: list[dict[str, Any]] = []
        results = _ordered_results(items, args)
        for result in tqdm(
            results,
            total=len(items),
            desc=f"BVH -> {split}.npy",
            unit="motion",
            leave=True,
            ncols=100,
        ):
            if not result["ok"]:
                errors.append(result)
                continue
            source_success_counts[split] += 1
            if result.get("excluded_reason") == "below_min_frames":
                excluded_short_source_counts[split] += 1
                excluded_short_frame_counts[split] += int(result["source_length"])
            if result["records"]:
                retained_source_counts[split] += 1
            for record in result["records"]:
                # Test and custom result producers may still return an in-memory
                # array; production workers save it before crossing the process pipe.
                if "motion" in record:
                    _save_record_motion(record, args.output)
                split_records.append(record)
        records_by_split[split] = split_records
    finalize_processed_dataset(
        args,
        records_by_split,
        errors,
        source_success_counts,
        source_dataset="BONES-SEED/soma_uniform",
        segmentation=SEGMENTATION_POLICY,
        metadata_extra={
            "preprocessing_protocol": protocol,
            "preprocessing_protocol_sha256": _protocol_sha256(protocol),
            "retained_source_split_counts": retained_source_counts,
            "excluded_short_source_split_counts": excluded_short_source_counts,
            "excluded_short_frame_split_counts": excluded_short_frame_counts,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BONES-SEED SOMA BVHs to variable-length MotionJEPA v2 NPY clips."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=PROJECT_ROOT / "dataset/bones-seed/soma_uniform",
    )
    parser.add_argument(
        "--splits_root",
        type=Path,
        default=PROJECT_ROOT / "dataset/Kimodo-Motion-Gen-Benchmark/splits",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=PROJECT_ROOT / "dataset/bones-seed/metadata/seed_metadata_v004.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset/bones-seed-processed-v2-nframes150-min60",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Worker processes. Each worker uses one PyTorch CPU thread.",
    )
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument(
        "--min_frames", "--min-frames", type=int, default=60,
        help="Minimum valid frames after FPS conversion (60 = 2 seconds at 30 FPS).",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        "--max_per_split",
        dest="max_per_split",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Process at most N source motions from each split. "
            "Negative values process every source motion."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    preprocess(parse_args())
