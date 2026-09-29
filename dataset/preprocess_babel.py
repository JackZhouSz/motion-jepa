"""Build BABEL-60/120 action clips from fixed-identity AMASS SOMA77 BVHs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.preprocess_bones_seed import (  # noqa: E402
    SPLITS,
    _prepare_output,
    _validate_complete_dataset,
    finalize_processed_dataset,
    round_fps,
)
from motion_rep import MotionJEPAMotionRep  # noqa: E402
from skeleton import SOMASkeleton30, parse_bvh_motion  # noqa: E402


FPS = 30
NUM_FRAMES = 150
PREPROCESSING_VERSION = 2
DEFAULT_INPUT = PROJECT_ROOT / "dataset/amass_soma_bvh"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "dataset/babel-annotation"
DEFAULT_LABELS = PROJECT_ROOT / "dataset/babel-60-and-120"
DEFAULT_MANIFEST_NAME = "conversion_manifest.jsonl"
DEFAULT_CONVERSION_ERRORS_NAME = "errors.jsonl"

# BABEL feat_p uses the release name, while the AMASS SMPL-X archives use the
# directory names on the right for these eight collections.
AMASS_SUBSET_ALIASES = {
    "DFaust67": "DFaust",
    "EyesJapanDataset": "Eyes_Japan_Dataset",
    "MPIHDM05": "HDM05",
    "MPILimits": "PosePrior",
    "MPImosh": "MoSh",
    "SSMsynced": "SSM",
    "TCDhandMocap": "TCDHands",
    "Transitionsmocap": "Transitions",
}

_INPUT_DIR: Path | None = None
_OUTPUT_ROOT: Path | None = None
_TARGET_SKELETON: SOMASkeleton30 | None = None
_THREAD_CONFIG_PID: int | None = None
_MIN_FRAMES = FPS


@dataclass(frozen=True)
class LabelRow:
    label: int
    annotator_id: str


@dataclass(frozen=True)
class Chunk:
    segment_id: str
    chunk_n: int
    segment_start: int
    segment_end: int
    start_t: float
    end_t: float
    annotation_type: str
    labels: tuple[LabelRow, ...]


@dataclass(frozen=True)
class SourceWork:
    split: str
    sid: int
    feat_p: str
    source_relpath: str
    bvh_relpath: str
    conversion: dict[str, Any]
    chunks: tuple[Chunk, ...]


def babel_feat_p_to_amass_relpath(feat_p: str) -> str:
    """Map a BABEL feature path to an AMASS SMPL-X stage-II relative path."""
    text = str(feat_p).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        raise ValueError(f"unsafe or malformed BABEL feat_p: {feat_p!r}")
    filename = path.name
    if not filename.endswith("_poses.npz"):
        raise ValueError(f"BABEL feat_p must end in _poses.npz: {feat_p!r}")
    subset = AMASS_SUBSET_ALIASES.get(path.parts[0], path.parts[0])
    tail = list(path.parts[2:])
    # AMASS stage-II archives normalize spaces in filenames to underscores,
    # while BABEL feat_p retains the original spaces (notably in ACCAD).
    tail[-1] = filename[: -len("_poses.npz")].replace(" ", "_") + "_stageii.npz"
    return PurePosixPath(subset, *tail).as_posix()


def _safe_relative_path(value: Any, suffix: str, description: str) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix.lower() != suffix
    ):
        raise ValueError(f"invalid {description}: {value!r}")
    return path.as_posix()


def _safe_component(value: Any, description: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"invalid {description}: {value!r}")
    return text


def load_conversion_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Read successful conversions, keyed by AMASS source relative path."""
    if not path.is_file():
        raise FileNotFoundError(f"conversion manifest does not exist: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"conversion record must be an object at {path}:{line_number}")
        if record.get("status") != "converted":
            continue
        source = _safe_relative_path(record.get("source_relpath"), ".npz", "source_relpath")
        output = _safe_relative_path(record.get("output_relpath"), ".bvh", "output_relpath")
        if source in records:
            raise ValueError(f"duplicate converted source_relpath: {source}")
        records[source] = {**record, "source_relpath": source, "output_relpath": output}
    if not records:
        raise ValueError(f"conversion manifest has no converted records: {path}")
    return records


def load_conversion_errors(path: Path) -> dict[str, dict[str, Any]]:
    """Load optional upstream discard/failure details for clearer skip reports."""
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"conversion error must be an object at {path}:{line_number}")
        source = _safe_relative_path(record.get("source_relpath"), ".npz", "source_relpath")
        records[source] = record
    return records


def load_action_labels(path: Path, subset: int) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"BABEL action label map does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) != 150:
        raise ValueError("action label map must contain the official 150 BABEL classes")
    by_index: dict[int, str] = {}
    for name, index in raw.items():
        if not isinstance(name, str) or not name or isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"invalid action label mapping: {name!r}: {index!r}")
        if index in by_index:
            raise ValueError(f"duplicate action label index: {index}")
        by_index[index] = name
    if set(by_index) != set(range(150)):
        raise ValueError("action label indices must be exactly 0..149")
    return tuple(by_index[index] for index in range(subset))


def _load_json_annotations(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"BABEL annotation does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"BABEL annotation must be an object: {path}")
    result: dict[int, dict[str, Any]] = {}
    for value in raw.values():
        if not isinstance(value, dict) or "babel_sid" not in value:
            raise ValueError(f"malformed BABEL annotation entry in {path}")
        sid = int(value["babel_sid"])
        if sid in result:
            raise ValueError(f"duplicate babel_sid {sid} in {path}")
        result[sid] = value
    return result


def _load_label_rows(path: Path, subset: int) -> list[tuple[str, LabelRow, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"BABEL label pickle does not exist: {path}")
    # These are official local BABEL files. Pickle must never be loaded from an
    # untrusted source.
    with path.open("rb") as file:
        payload = pickle.load(file)
    try:
        segment_ids, (labels, sids, chunk_ns, annotator_ids) = payload
    except (TypeError, ValueError) as error:
        raise ValueError(f"malformed BABEL label pickle: {path}") from error
    lengths = {len(segment_ids), len(labels), len(sids), len(chunk_ns), len(annotator_ids)}
    if len(lengths) != 1:
        raise ValueError(f"BABEL label columns have unequal lengths: {path}")
    rows: list[tuple[str, LabelRow, int, int]] = []
    seen: set[tuple[str, int, int, int, str]] = set()
    for segment_id, label, sid, chunk_n, annotator_id in zip(
        segment_ids, labels, sids, chunk_ns, annotator_ids
    ):
        label = int(label)
        sid = int(sid)
        chunk_n = int(chunk_n)
        if not 0 <= label < subset:
            raise ValueError(f"label {label} is outside BABEL-{subset} in {path}")
        if chunk_n < 0:
            raise ValueError(f"negative chunk index for sid={sid}: {chunk_n}")
        segment_id = _safe_component(segment_id, "segment ID")
        key = (segment_id, label, sid, chunk_n, str(annotator_id))
        if key in seen:
            continue
        seen.add(key)
        rows.append((segment_id, LabelRow(label, str(annotator_id)), sid, chunk_n))
    return rows


def _segments_for_motion(annotation: dict[str, Any]) -> tuple[str, str, dict[str, dict[str, Any]]]:
    frame_ann = annotation.get("frame_ann")
    annotation_type = "frame_ann" if frame_ann is not None else "seq_ann"
    container = frame_ann if frame_ann is not None else annotation.get("seq_ann")
    if not isinstance(container, dict) or not isinstance(container.get("labels"), list):
        raise ValueError(f"missing usable {annotation_type} for sid={annotation.get('babel_sid')}")
    annotator_id = str(container.get("anntr_id", ""))
    duration = float(annotation.get("dur"))
    segments: dict[str, dict[str, Any]] = {}
    for label in container["labels"]:
        if not isinstance(label, dict) or not label.get("seg_id"):
            raise ValueError(f"malformed segment for sid={annotation.get('babel_sid')}")
        segment_id = _safe_component(label["seg_id"], "segment ID")
        start_t = float(label.get("start_t", 0.0))
        end_t = float(label.get("end_t", duration))
        if segment_id in segments:
            raise ValueError(f"duplicate segment id for sid={annotation.get('babel_sid')}: {segment_id}")
        segments[segment_id] = {"start_t": start_t, "end_t": end_t}
    return annotation_type, annotator_id, segments


def build_work_items(
    *,
    subset: int,
    annotations_dir: Path,
    labels_dir: Path,
    conversions: dict[str, dict[str, Any]],
    conversion_errors: dict[str, dict[str, Any]],
    input_dir: Path,
    limit: int | None,
) -> tuple[list[SourceWork], list[dict[str, Any]], dict[str, int]]:
    """Join official label rows, BABEL JSON, and the SOMA conversion manifest."""
    works: list[SourceWork] = []
    skips: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for split in ("train", "val"):
        annotations = _load_json_annotations(annotations_dir / f"{split}.json")
        rows = _load_label_rows(labels_dir / f"{split}_label_{subset}.pkl", subset)
        counts[f"{split}_label_rows"] = len(rows)
        by_sid: dict[int, list[tuple[str, LabelRow, int]]] = defaultdict(list)
        for segment_id, label_row, sid, chunk_n in rows:
            by_sid[sid].append((segment_id, label_row, chunk_n))
        for sid in sorted(by_sid):
            source_rows = by_sid[sid]
            annotation = annotations.get(sid)
            if annotation is None:
                raise ValueError(f"sid={sid} from {split} PKL is absent from {split}.json")
            feat_p = str(annotation.get("feat_p", ""))
            source_relpath = babel_feat_p_to_amass_relpath(feat_p)
            conversion = conversions.get(source_relpath)
            if conversion is None:
                upstream = conversion_errors.get(source_relpath)
                status = str(upstream.get("status")) if upstream else "missing"
                reason = str(upstream.get("error")) if upstream else "source absent from conversion manifest"
                skips.append({
                    "ok": False,
                    "kind": f"upstream_{status}",
                    "split": split,
                    "sid": sid,
                    "feat_p": feat_p,
                    "source_amass_relpath": source_relpath,
                    "error": reason,
                    "skipped_label_rows": len(source_rows),
                })
                counts[f"upstream_{status}_motions"] += 1
                counts[f"upstream_{status}_label_rows"] += len(source_rows)
                continue
            bvh_relpath = str(conversion["output_relpath"])
            if not (input_dir / bvh_relpath).is_file():
                skips.append({
                    "ok": False,
                    "kind": "missing_bvh",
                    "split": split,
                    "sid": sid,
                    "feat_p": feat_p,
                    "source_amass_relpath": source_relpath,
                    "source_bvh_relpath": bvh_relpath,
                    "error": "converted BVH file does not exist",
                    "skipped_label_rows": len(source_rows),
                })
                counts["missing_bvh_motions"] += 1
                counts["missing_bvh_label_rows"] += len(source_rows)
                continue
            annotation_type, annotation_annotator, segments = _segments_for_motion(annotation)
            grouped: dict[tuple[str, int], list[LabelRow]] = defaultdict(list)
            chunk_info: dict[tuple[str, int], tuple[int, int, float, float]] = {}
            for segment_id, label_row, chunk_n in source_rows:
                segment = segments.get(segment_id)
                if segment is None:
                    raise ValueError(f"unknown segment {segment_id} for sid={sid}")
                if annotation_annotator != label_row.annotator_id:
                    raise ValueError(f"annotator mismatch for sid={sid}, segment={segment_id}")
                start_t, end_t = segment["start_t"], segment["end_t"]
                if not 0.0 <= start_t <= end_t:
                    raise ValueError(
                        f"invalid segment times for sid={sid}, segment={segment_id}"
                    )
                start_frame, end_frame = int(FPS * start_t), int(FPS * end_t)
                key = (segment_id, chunk_n)
                info = (start_frame, end_frame, start_t, end_t)
                if key in chunk_info and chunk_info[key] != info:
                    raise ValueError(f"conflicting chunk metadata for sid={sid}, segment={segment_id}")
                chunk_info[key] = info
                grouped[key].append(label_row)
            chunks = []
            for (segment_id, chunk_n), label_rows in sorted(grouped.items()):
                start_frame, end_frame, start_t, end_t = chunk_info[(segment_id, chunk_n)]
                chunks.append(Chunk(
                    segment_id=segment_id,
                    chunk_n=chunk_n,
                    segment_start=start_frame,
                    segment_end=end_frame,
                    start_t=start_t,
                    end_t=end_t,
                    annotation_type=annotation_type,
                    labels=tuple(sorted(label_rows, key=lambda row: row.label)),
                ))
            works.append(SourceWork(
                split=split,
                sid=sid,
                feat_p=feat_p,
                source_relpath=source_relpath,
                bvh_relpath=bvh_relpath,
                conversion=conversion,
                chunks=tuple(chunks),
            ))
    works.sort(key=lambda work: (SPLITS.index(work.split), work.sid))
    if limit is not None:
        works = works[:limit]
    return works, skips, dict(counts)


def _init_worker(input_dir: str, output_root: str, min_frames: int = FPS) -> None:
    global _INPUT_DIR, _OUTPUT_ROOT, _TARGET_SKELETON, _THREAD_CONFIG_PID, _MIN_FRAMES
    _INPUT_DIR = Path(input_dir)
    _OUTPUT_ROOT = Path(output_root)
    _MIN_FRAMES = min_frames
    current_pid = os.getpid()
    if _THREAD_CONFIG_PID != current_pid:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _THREAD_CONFIG_PID = current_pid
    _TARGET_SKELETON = SOMASkeleton30()


def _chunk_motion_relpath(work: SourceWork, chunk: Chunk) -> str:
    return PurePosixPath(
        "motions", work.split, f"{work.sid:05d}", chunk.segment_id,
        f"chunk_{chunk.chunk_n:03d}.npy",
    ).as_posix()


def _record_id(work: SourceWork, chunk: Chunk, label: int) -> str:
    return PurePosixPath(
        work.split, f"{work.sid:05d}", chunk.segment_id, f"chunk_{chunk.chunk_n:03d}",
        f"label_{label:03d}",
    ).as_posix()


def _conversion_metadata(conversion: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_fps_raw", "source_fps", "source_num_frames", "frame_step",
        "output_num_frames", "target_fps", "mean_vertex_error_m",
        "max_vertex_error_m", "motion_correction", "motion_correction_settings",
        "smpl_model_sha256", "skeleton_sha256", "target_identity",
        "resampling_method",
    )
    return {key: conversion[key] for key in keys if key in conversion}


def _convert_one(work: SourceWork, class_names: tuple[str, ...]) -> dict[str, Any]:
    assert _INPUT_DIR is not None
    assert _OUTPUT_ROOT is not None
    assert _TARGET_SKELETON is not None
    try:
        conversion = work.conversion
        target_fps = int(conversion.get("target_fps", -1))
        source_fps = int(conversion.get("source_fps", -1))
        frame_step = conversion.get("frame_step")
        method = conversion.get("resampling_method", "fixed_step")
        if target_fps != FPS:
            raise ValueError(f"manifest target_fps must be {FPS}, got {target_fps}")
        if source_fps <= 0:
            raise ValueError(f"manifest source_fps must be positive, got {source_fps}")
        if method == "fixed_step":
            if source_fps % FPS or frame_step != source_fps // FPS:
                raise ValueError(
                    f"invalid fixed-step manifest FPS: source={source_fps}, step={frame_step}"
                )
        elif method == "lerp_slerp":
            if source_fps % FPS == 0 or frame_step is not None:
                raise ValueError(
                    f"invalid interpolation manifest FPS: source={source_fps}, step={frame_step}"
                )
        else:
            raise ValueError(f"unknown resampling_method: {method!r}")
        rotations, roots, parsed_fps = parse_bvh_motion(_INPUT_DIR / work.bvh_relpath)
        if round_fps(float(parsed_fps)) != FPS:
            raise ValueError(f"SOMA BVH must be {FPS} FPS, got {parsed_fps}")
        expected = (len(roots), 77, 3, 3)
        if tuple(rotations.shape) != expected:
            raise ValueError(f"expected SOMA77 rotations {expected}, got {tuple(rotations.shape)}")
        if tuple(roots.shape) != (len(rotations), 3):
            raise ValueError(f"unexpected root shape: {tuple(roots.shape)}")
        if not torch.isfinite(rotations).all() or not torch.isfinite(roots).all():
            raise ValueError("BVH contains NaN or infinity")
        manifest_frames = conversion.get("output_num_frames")
        if manifest_frames is not None and int(manifest_frames) != len(rotations):
            raise ValueError(
                f"manifest has {manifest_frames} frames but BVH has {len(rotations)}"
            )
        rotations = _TARGET_SKELETON.from_soma77(rotations)
        representation = MotionJEPAMotionRep(_TARGET_SKELETON, FPS)
        records: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        common_metadata = _conversion_metadata(conversion)
        for chunk in work.chunks:
            start = chunk.segment_start + chunk.chunk_n * NUM_FRAMES
            end = min(start + NUM_FRAMES, chunk.segment_end, len(rotations))
            if start >= len(rotations) or end <= start:
                errors.append({
                    "ok": False,
                    "kind": "empty_chunk",
                    "split": work.split,
                    "sid": work.sid,
                    "feat_p": work.feat_p,
                    "segment_id": chunk.segment_id,
                    "chunk_n": chunk.chunk_n,
                    "valid_length": max(0, end - start),
                    "min_frames": _MIN_FRAMES,
                    "error": f"chunk [{start}, {end}) is outside BVH with {len(rotations)} frames",
                    "skipped_label_rows": len(chunk.labels),
                })
                continue
            if end - start < _MIN_FRAMES:
                errors.append({
                    "ok": False,
                    "kind": "short_chunk",
                    "split": work.split,
                    "sid": work.sid,
                    "feat_p": work.feat_p,
                    "segment_id": chunk.segment_id,
                    "chunk_n": chunk.chunk_n,
                    "valid_length": end - start,
                    "min_frames": _MIN_FRAMES,
                    "error": f"chunk has {end - start} frames; minimum is {_MIN_FRAMES}",
                    "skipped_label_rows": len(chunk.labels),
                })
                continue
            features = representation(
                rotations[start:end], roots[start:end], to_canonicalize=True
            )
            motion = np.ascontiguousarray(features.detach().cpu().numpy(), dtype=np.float32)
            if motion.shape != (end - start, MotionJEPAMotionRep.FEATURE_DIM):
                raise ValueError(f"unexpected encoded shape: {motion.shape}")
            if not np.isfinite(motion).all():
                raise ValueError("encoded representation contains NaN or infinity")
            relative = _chunk_motion_relpath(work, chunk)
            destination = _OUTPUT_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as file:
                np.save(file, motion, allow_pickle=False)
            motion64 = motion.astype(np.float64, copy=False)
            stats_sum = motion64.sum(axis=0)
            stats_sq_sum = np.square(motion64).sum(axis=0)
            for label_row in chunk.labels:
                label_name = class_names[label_row.label]
                metadata = {
                    **common_metadata,
                    "label": label_row.label,
                    "label_name": label_name,
                    "babel_sid": work.sid,
                    "segment_id": chunk.segment_id,
                    "chunk_n": chunk.chunk_n,
                    "annotator_id": label_row.annotator_id,
                    "annotation_type": chunk.annotation_type,
                    "feat_p": work.feat_p,
                    "source_amass_relpath": work.source_relpath,
                    "source_bvh_relpath": work.bvh_relpath,
                    "segment_start_t": chunk.start_t,
                    "segment_end_t": chunk.end_t,
                    "segment_start_frame": chunk.segment_start,
                    "segment_end_frame": chunk.segment_end,
                    "chunk_start_frame": start,
                    "chunk_end_frame": end,
                    "valid_length": len(motion),
                }
                record = {
                    "id": _record_id(work, chunk, label_row.label),
                    "source_id": str(work.sid),
                    "split": work.split,
                    "source_path": work.bvh_relpath,
                    "source_bvh_relpath": work.bvh_relpath,
                    "source_amass_relpath": work.source_relpath,
                    "start_frame": start,
                    "end_frame": end,
                    "fps": FPS,
                    "length": len(motion),
                    "motion_dim": motion.shape[1],
                    "motion_path": relative,
                    "captions": [label_name],
                    "metadata": metadata,
                }
                if work.split == "train":
                    # Arrays are deliberately shared across labels for this chunk.
                    # finalize_processed_dataset counts each label row, matching
                    # the classifier's effective sampling distribution.
                    record["_stats_sum"] = stats_sum
                    record["_stats_sq_sum"] = stats_sq_sum
                records.append(record)
        return {"ok": True, "sid": work.sid, "split": work.split, "records": records, "errors": errors}
    except Exception as error:
        return {
            "ok": False,
            "kind": "conversion_error",
            "split": work.split,
            "sid": work.sid,
            "feat_p": work.feat_p,
            "source_amass_relpath": work.source_relpath,
            "source_bvh_relpath": work.bvh_relpath,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
            "skipped_label_rows": sum(len(chunk.labels) for chunk in work.chunks),
        }


def _convert_batch(batch: tuple[Sequence[SourceWork], tuple[str, ...]]) -> list[dict[str, Any]]:
    works, class_names = batch
    return [_convert_one(work, class_names) for work in works]


def _ordered_results(
    works: Sequence[SourceWork], args: argparse.Namespace, class_names: tuple[str, ...]
) -> Iterator[dict[str, Any]]:
    if not works:
        return
    init_args = (str(args.input_dir), str(args.output), args.min_frames)
    workers = min(max(1, int(args.workers)), len(works))
    if workers == 1:
        _init_worker(*init_args)
        yield from (_convert_one(work, class_names) for work in works)
        return
    batches = [
        (works[start : start + args.chunksize], class_names)
        for start in range(0, len(works), args.chunksize)
    ]
    with Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
        for results in pool.imap(_convert_batch, batches, chunksize=1):
            yield from results


def _validate_args(args: argparse.Namespace) -> None:
    if args.subset not in (60, 120):
        raise ValueError("--subset must be 60 or 120")
    if args.input_dir.resolve() == args.output.resolve():
        raise ValueError("input and output directories must differ")
    if args.workers <= 0 or args.chunksize <= 0:
        raise ValueError("workers and chunksize must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not 1 <= args.min_frames <= NUM_FRAMES:
        raise ValueError(f"--min-frames must be between 1 and {NUM_FRAMES}")


def preprocess(args: argparse.Namespace) -> None:
    args.input_dir = Path(args.input_dir).resolve()
    args.annotations_dir = Path(args.annotations_dir).resolve()
    args.labels_dir = Path(args.labels_dir).resolve()
    args.output = Path(args.output).resolve() if args.output else (
        PROJECT_ROOT / f"dataset/babel-{args.subset}-processed"
    ).resolve()
    args.manifest = Path(args.manifest).resolve() if args.manifest else (
        args.input_dir / DEFAULT_MANIFEST_NAME
    )
    args.action_label_map = Path(args.action_label_map).resolve() if args.action_label_map else (
        args.labels_dir / "action_label_2_idx.json"
    )
    args.fps = FPS
    args.num_frames = NUM_FRAMES
    args.overlap = 0.0
    args.split_seed = None
    args.min_frames = getattr(args, "min_frames", FPS)
    _validate_args(args)
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    reuse_metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "min_frames": args.min_frames,
        "conversion_manifest_sha256": manifest_sha256,
        "subset": args.subset,
    }
    if (
        args.output.exists()
        and _validate_complete_dataset(args.output)
        and (args.output / "class-index.json").is_file()
        and not args.overwrite
    ):
        existing_metadata = json.loads((args.output / "meta.json").read_text(encoding="utf-8"))
        mismatches = [
            key for key, value in reuse_metadata.items()
            if existing_metadata.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Existing BABEL dataset has incompatible metadata ({', '.join(mismatches)}); "
                "use a new --output directory or explicit --overwrite"
            )
        print(f"Reusing complete NPY dataset: {args.output}")
        return
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"SOMA BVH input directory does not exist: {args.input_dir}")
    class_names = load_action_labels(args.action_label_map, args.subset)
    conversions = load_conversion_manifest(args.manifest)
    conversion_errors_path = args.manifest.parent / DEFAULT_CONVERSION_ERRORS_NAME
    conversion_errors = load_conversion_errors(conversion_errors_path)
    works, errors, discovery_counts = build_work_items(
        subset=args.subset,
        annotations_dir=args.annotations_dir,
        labels_dir=args.labels_dir,
        conversions=conversions,
        conversion_errors=conversion_errors,
        input_dir=args.input_dir,
        limit=args.limit,
    )
    if not works:
        raise RuntimeError("no BABEL motions have corresponding converted SOMA BVHs")
    _prepare_output(args.output, args.overwrite)
    records_by_split = {split: [] for split in SPLITS}
    source_success_counts = {split: 0 for split in SPLITS}
    converted_chunks: set[tuple[str, int, str, int]] = set()
    for result in tqdm(
        _ordered_results(works, args, class_names),
        total=len(works),
        desc=f"BABEL-{args.subset} SOMA BVH -> NPY",
        unit="motion",
    ):
        if not result["ok"]:
            errors.append(result)
            continue
        errors.extend(result["errors"])
        if result["records"]:
            source_success_counts[result["split"]] += 1
        for record in result["records"]:
            metadata = record["metadata"]
            converted_chunks.add((
                result["split"], result["sid"], metadata["segment_id"], metadata["chunk_n"]
            ))
            records_by_split[result["split"]].append(record)
    if not records_by_split["train"]:
        with (args.output / "errors.jsonl").open("w", encoding="utf-8") as file:
            for error in errors:
                file.write(json.dumps(error, ensure_ascii=False) + "\n")
        first = errors[0].get("error", "no valid chunks") if errors else "no valid chunks"
        raise RuntimeError(f"no BABEL training samples were produced: {first}")
    all_ids: set[str] = set()
    for split in SPLITS:
        for record in records_by_split[split]:
            if record["id"] in all_ids:
                raise ValueError(f"duplicate or conflicting sample ID: {record['id']}")
            all_ids.add(record["id"])
    error_counts = dict(Counter(str(error.get("kind", "unknown")) for error in errors))
    skipped_label_rows = sum(int(error.get("skipped_label_rows", 0)) for error in errors)
    finalize_processed_dataset(
        args,
        records_by_split,
        errors,
        source_success_counts,
        source_dataset=f"BABEL-{args.subset}_fixed_identity_soma77",
        segmentation="official_babel_action_segments_150_frame_chunks",
        metadata_extra={
            **reuse_metadata,
            "class_names": list(class_names),
            "num_classes": args.subset,
            "split_policy": "official_babel_train_val_test_empty",
            "tail_policy": "store_valid_length_loader_zero_pads_to_150",
            "source_standard_tpose": True,
            "source_rotations_already_tpose_relative": True,
            "resampled": False,
            "downsampling": "performed_upstream_fixed_step_or_lerp_slerp",
            "fps_validation": "positive_rounded_source_fps_target_30",
            "resampling_stage": "amass_to_soma",
            "conversion_manifest": str(args.manifest),
            "conversion_errors": str(conversion_errors_path),
            "annotations_dir": str(args.annotations_dir),
            "labels_dir": str(args.labels_dir),
            "action_label_map": str(args.action_label_map),
            "num_selected_source_motions": len(works),
            "num_unique_motion_chunks": len(converted_chunks),
            "discovery_counts": discovery_counts,
            "error_counts": error_counts,
            "num_skipped_label_rows": skipped_label_rows,
        },
    )
    class_index = {
        "class_names": list(class_names),
        "class_to_index": {name: index for index, name in enumerate(class_names)},
    }
    (args.output / "class-index.json").write_text(
        json.dumps(class_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Unique motion chunks: {len(converted_chunks)}")
    print(f"Skipped/error records: {len(errors)}")
    print(json.dumps({
        "subset": args.subset,
        "selected_source_motions": len(works),
        "converted_source_motions": sum(source_success_counts.values()),
        "train_label_samples": len(records_by_split["train"]),
        "val_label_samples": len(records_by_split["val"]),
        "test_label_samples": 0,
        "unique_motion_chunks": len(converted_chunks),
        "skipped_label_rows": skipped_label_rows,
        "error_counts": error_counts,
    }, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, choices=(60, 120), required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, help="Defaults to <input-dir>/conversion_manifest.jsonl.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--action-label-map", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument(
        "--min-frames", type=int, default=FPS,
        help="Minimum valid frames per final 30 FPS chunk (1–150; default: 30).",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of joined source motions.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preprocess(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
