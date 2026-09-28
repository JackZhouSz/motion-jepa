"""Preprocess fixed-identity AMASS SOMA77 BVHs into Motion-JEPA NPY windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
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
    _save_record_motion,
    _validate_complete_dataset,
    calculate_stride,
    finalize_processed_dataset,
    round_fps,
)
from motion_rep import MotionJEPAMotionRep  # noqa: E402
from skeleton import SOMASkeleton30, parse_bvh_motion  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "dataset/amass_soma_bvh"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset/amass-soma-processed"
DEFAULT_MANIFEST_NAME = "conversion_manifest.jsonl"

_INPUT_DIR: Path | None = None
_OUTPUT_ROOT: Path | None = None
_NUM_FRAMES = 150
_FPS = 30
_STRIDE_FRAMES = 75
_TARGET_SKELETON: SOMASkeleton30 | None = None
_THREAD_CONFIG_PID: int | None = None


@dataclass(frozen=True)
class SourceMotion:
    id: str
    bvh_relpath: str
    conversion: dict[str, Any]


def _safe_bvh_relpath(value: Any, *, line_number: int) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix.lower() != ".bvh"
    ):
        raise ValueError(
            f"invalid output_relpath on manifest line {line_number}: {value!r}"
        )
    return path.as_posix()


def load_conversion_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load successful conversion records keyed by their BVH relative path."""
    if not path.is_file():
        raise FileNotFoundError(f"conversion manifest does not exist: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON on manifest line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        if record.get("status") != "converted":
            continue
        if "output_relpath" not in record:
            raise ValueError(
                f"converted manifest line {line_number} has no output_relpath"
            )
        relative = _safe_bvh_relpath(
            record["output_relpath"], line_number=line_number
        )
        if relative in records:
            raise ValueError(f"duplicate converted output_relpath: {relative}")
        records[relative] = dict(record)
    if not records:
        raise ValueError(f"manifest has no converted records: {path}")
    return records


def discover_sources(
    input_dir: Path,
    manifest_path: Path,
    limit: int | None = None,
) -> list[SourceMotion]:
    """Validate the manifest/BVH set and return deterministic work items."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"AMASS BVH input directory does not exist: {input_dir}")
    records = load_conversion_manifest(manifest_path)
    discovered = {
        path.relative_to(input_dir).as_posix()
        for path in input_dir.rglob("*.bvh")
        if path.is_file()
    }
    expected = set(records)
    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing BVHs: {missing[:5]}")
        if unexpected:
            details.append(f"BVHs absent from manifest: {unexpected[:5]}")
        raise ValueError("manifest/BVH mismatch; " + "; ".join(details))

    selected = sorted(expected)
    if limit is not None:
        selected = selected[:limit]
    return [
        SourceMotion(
            id=PurePosixPath(relative).with_suffix("").as_posix(),
            bvh_relpath=relative,
            conversion=records[relative],
        )
        for relative in selected
    ]


def complete_windows(
    num_frames: int,
    window_frames: int,
    stride_frames: int,
) -> tuple[tuple[int, int], ...]:
    if num_frames < window_frames:
        return ()
    return tuple(
        (start, start + window_frames)
        for start in range(0, num_frames - window_frames + 1, stride_frames)
    )


def _init_worker(
    input_dir: str,
    output_root: str,
    num_frames: int,
    fps: int,
    stride_frames: int,
) -> None:
    global _INPUT_DIR, _OUTPUT_ROOT, _NUM_FRAMES, _FPS, _STRIDE_FRAMES
    global _TARGET_SKELETON, _THREAD_CONFIG_PID
    _INPUT_DIR = Path(input_dir)
    _OUTPUT_ROOT = Path(output_root)
    _NUM_FRAMES = int(num_frames)
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
    _TARGET_SKELETON = SOMASkeleton30()


def _segment_id(source_id: str, segment_index: int, num_windows: int) -> str:
    return (
        f"{source_id}_{segment_index:04d}"
        if num_windows > 1
        else source_id
    )


def _record_metadata(source: SourceMotion) -> dict[str, Any]:
    conversion = source.conversion
    keys = (
        "source_relpath",
        "source_fps_raw",
        "source_fps",
        "source_num_frames",
        "frame_step",
        "output_num_frames",
        "source_gender",
        "surface_model_type",
        "target_identity",
        "smpl_model_sha256",
        "skeleton_sha256",
        "mean_vertex_error_m",
        "max_vertex_error_m",
        "motion_correction",
        "motion_correction_settings",
    )
    return {key: conversion[key] for key in keys if key in conversion}


def _convert_one(source: SourceMotion) -> dict[str, Any]:
    assert _INPUT_DIR is not None
    assert _OUTPUT_ROOT is not None
    assert _TARGET_SKELETON is not None
    source_path = _INPUT_DIR / source.bvh_relpath
    try:
        local_rotations, root_positions, parsed_fps = parse_bvh_motion(source_path)
        source_fps = round_fps(float(parsed_fps))
        if source_fps != _FPS:
            raise ValueError(
                f"AMASS SOMA BVH must be {_FPS} FPS, got {parsed_fps} "
                f"(rounded to {source_fps})"
            )
        expected_rotations = (len(root_positions), 77, 3, 3)
        if tuple(local_rotations.shape) != expected_rotations:
            raise ValueError(
                f"expected SOMA77 rotations {expected_rotations}, "
                f"got {tuple(local_rotations.shape)}"
            )
        if tuple(root_positions.shape) != (len(local_rotations), 3):
            raise ValueError(
                f"expected root positions ({len(local_rotations)}, 3), "
                f"got {tuple(root_positions.shape)}"
            )
        if not torch.isfinite(local_rotations).all() or not torch.isfinite(
            root_positions
        ).all():
            raise ValueError("BVH motion contains NaN or infinity")
        manifest_frames = source.conversion.get("output_num_frames")
        if manifest_frames is not None and int(manifest_frames) != len(local_rotations):
            raise ValueError(
                f"manifest reports {manifest_frames} output frames, "
                f"but BVH contains {len(local_rotations)}"
            )
        manifest_fps = source.conversion.get("target_fps")
        if manifest_fps is not None and int(manifest_fps) != _FPS:
            raise ValueError(
                f"manifest target FPS {manifest_fps} does not match {_FPS}"
            )

        windows = complete_windows(
            len(local_rotations), _NUM_FRAMES, _STRIDE_FRAMES
        )
        if not windows:
            return {
                "ok": True,
                "id": source.id,
                "discarded": True,
                "source_frames": len(local_rotations),
                "records": [],
            }

        # convert_amass_to_soma.py already emits T-pose-relative rotations.
        # Applying SOMASkeleton77.to_standard_tpose() here would rotate them a
        # second time and produce an invalid pose.
        local_rotations = _TARGET_SKELETON.from_soma77(local_rotations)
        representation = MotionJEPAMotionRep(_TARGET_SKELETON, _FPS)
        metadata = _record_metadata(source)
        records: list[dict[str, Any]] = []
        for segment_index, (start_frame, end_frame) in enumerate(windows):
            features = representation(
                local_rotations[start_frame:end_frame],
                root_positions[start_frame:end_frame],
                to_canonicalize=True,
            )
            motion = np.ascontiguousarray(
                features.detach().cpu().numpy(), dtype=np.float32
            )
            expected = (_NUM_FRAMES, MotionJEPAMotionRep.FEATURE_DIM)
            if motion.shape != expected:
                raise ValueError(
                    f"unexpected feature shape {motion.shape}, expected {expected}"
                )
            if not np.isfinite(motion).all():
                raise ValueError("encoded representation contains NaN or infinity")
            record = {
                "id": _segment_id(source.id, segment_index, len(windows)),
                "source_id": source.id,
                "segment_index": segment_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "split": "train",
                "source_path": source.bvh_relpath,
                "source_bvh_relpath": source.bvh_relpath,
                "source_amass_relpath": source.conversion.get("source_relpath"),
                "source_fps": source_fps,
                "fps": _FPS,
                "length": len(motion),
                "motion_dim": motion.shape[1],
                "captions": [],
                "metadata": metadata,
                "motion": motion,
            }
            records.append(_save_record_motion(record, _OUTPUT_ROOT))
        return {
            "ok": True,
            "id": source.id,
            "discarded": False,
            "source_frames": len(local_rotations),
            "records": records,
        }
    except Exception as error:
        return {
            "ok": False,
            "id": source.id,
            "source_path": source.bvh_relpath,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }


def _convert_batch(sources: Sequence[SourceMotion]) -> list[dict[str, Any]]:
    return [_convert_one(source) for source in sources]


def _ordered_results(
    sources: Sequence[SourceMotion],
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
    if not sources:
        return
    init_args = (
        str(args.input_dir),
        str(args.output),
        args.num_frames,
        args.fps,
        calculate_stride(args.num_frames, args.overlap),
    )
    worker_count = min(max(1, int(args.workers)), len(sources))
    if worker_count == 1:
        _init_worker(*init_args)
        yield from (_convert_one(source) for source in sources)
        return

    batches = [
        sources[start : start + args.chunksize]
        for start in range(0, len(sources), args.chunksize)
    ]
    with Pool(worker_count, initializer=_init_worker, initargs=init_args) as pool:
        for results in pool.imap(_convert_batch, batches, chunksize=1):
            yield from results


def _validate_args(args: argparse.Namespace) -> None:
    if args.input_dir.resolve() == args.output.resolve():
        raise ValueError("input and output directories must differ")
    if args.num_frames <= 0 or args.fps <= 0:
        raise ValueError("num-frames and fps must be positive")
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if args.workers <= 0 or args.chunksize <= 0:
        raise ValueError("workers and chunksize must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")


def preprocess(args: argparse.Namespace) -> None:
    args.input_dir = Path(args.input_dir).resolve()
    args.output = Path(args.output).resolve()
    args.manifest = (
        Path(args.manifest).resolve()
        if args.manifest is not None
        else args.input_dir / DEFAULT_MANIFEST_NAME
    )
    args.split_seed = None
    _validate_args(args)
    if args.output.exists() and _validate_complete_dataset(args.output) and not args.overwrite:
        print(f"Reusing complete NPY dataset: {args.output}")
        return

    sources = discover_sources(args.input_dir, args.manifest, args.limit)
    if not sources:
        raise RuntimeError("no AMASS SOMA BVH sources selected")
    _prepare_output(args.output, args.overwrite)

    records_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    errors: list[dict[str, Any]] = []
    discarded_sources: list[dict[str, Any]] = []
    successful_sources = 0
    for result in tqdm(
        _ordered_results(sources, args),
        total=len(sources),
        desc="AMASS SOMA BVH -> NPY",
        unit="motion",
    ):
        if not result["ok"]:
            errors.append(result)
            continue
        if result["discarded"]:
            discarded_sources.append(
                {
                    "id": result["id"],
                    "source_frames": result["source_frames"],
                    "reason": (
                        f"shorter than the required complete "
                        f"{args.num_frames}-frame window"
                    ),
                }
            )
            continue
        successful_sources += 1
        records_by_split["train"].extend(result["records"])

    if not records_by_split["train"]:
        first_error = errors[0]["error"] if errors else "all sources were too short"
        raise RuntimeError(f"no complete AMASS windows were produced: {first_error}")

    finalize_processed_dataset(
        args,
        records_by_split,
        errors,
        {"train": successful_sources, "val": 0, "test": 0},
        source_dataset="AMASS_fixed_identity_soma77",
        segmentation="overlapping_complete_windows",
        num_source_motions=successful_sources,
        metadata_extra={
            "split_policy": "all_train_without_babel",
            "split_unit": "source_motion",
            "tail_policy": "discard_incomplete",
            "source_standard_tpose": True,
            "source_rotations_already_tpose_relative": True,
            "resampled": False,
            "downsampling": "none_source_bvh_already_at_target_fps",
            "conversion_manifest": str(args.manifest),
            "num_discovered_source_motions": len(sources),
            "num_discarded_short_sources": len(discarded_sources),
            "discarded_short_sources": discarded_sources,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to <input-dir>/conversion_manifest.jsonl.",
    )
    parser.add_argument("--num-frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preprocess(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
