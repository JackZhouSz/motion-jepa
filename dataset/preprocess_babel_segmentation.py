"""Convert continuous BABEL timelines and strong frame labels to NPY windows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.babel_segmentation import (  # noqa: E402
    BabelSegmentationDataset, FORMAT, SUPERVISION_POLICY, file_sha256,
)
from dataset.preprocess_babel import (  # noqa: E402
    _conversion_metadata, _load_json_annotations, babel_feat_p_to_amass_relpath,
    load_action_labels, load_conversion_manifest,
)
from dataset.preprocess_bones_seed import (  # noqa: E402
    SPLITS, _prepare_output, _save_record_motion, _validate_complete_dataset,
    finalize_processed_dataset, round_fps,
)
from motion_rep import MotionJEPAMotionRep  # noqa: E402
from skeleton import SOMASkeleton30, parse_bvh_motion  # noqa: E402

FPS = 30
NUM_FRAMES = 150
MIN_FRAMES = 30
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset/babel-segmentation-120-processed-nframes30-150"
_INPUT: Path | None = None
_OUTPUT: Path | None = None
_SKELETON: SOMASkeleton30 | None = None
_CLASSES: tuple[str, ...] = ()


def rasterize_frame_labels(annotation: dict, num_frames: int, class_names: tuple[str, ...]):
    """Union strong action intervals; seq_ann never supplies temporal labels.

    Match the existing BABEL conversion convention: floor(seconds * 30),
    start-inclusive/end-exclusive, clipped to the converted source timeline.
    """
    container = annotation.get("frame_ann")
    if not isinstance(container, dict) or not isinstance(container.get("labels"), list):
        raise ValueError("Frame segmentation requires frame_ann labels")
    labels = np.zeros((num_frames, len(class_names)), dtype=np.bool_)
    coverage = np.zeros(num_frames, dtype=np.bool_)
    transition = np.zeros(num_frames, dtype=np.bool_)
    out_of_vocabulary = np.zeros(num_frames, dtype=np.bool_)
    mapping = {name: index for index, name in enumerate(class_names)}
    for segment in container["labels"]:
        start_t, end_t = float(segment["start_t"]), float(segment["end_t"])
        categories = segment.get("act_cat")
        if (not math.isfinite(start_t) or not math.isfinite(end_t)
                or not 0 <= start_t <= end_t or not isinstance(categories, list)
                or not categories or any(not isinstance(name, str) for name in categories)):
            raise ValueError(f"Malformed frame annotation: {segment}")
        start = min(num_frames, int(FPS * start_t))
        end = min(num_frames, int(FPS * end_t))
        coverage[start:end] = True
        transition[start:end] |= "transition" in categories
        out_of_vocabulary[start:end] |= any(name not in mapping for name in categories)
        for name in categories:
            if name in mapping:
                labels[start:end, mapping[name]] = True
    supervision = labels.any(axis=1)
    flags = {"annotated": coverage, "transition": transition, "oov": out_of_vocabulary}
    return labels, supervision, flags


def build_work_items(input_dir: Path, annotations_dir: Path, conversions: dict, limit: int | None = None):
    works, errors = [], []
    counts = Counter()
    source_splits = {}
    for split in ("train", "val"):
        selected = 0
        for sid, annotation in sorted(_load_json_annotations(annotations_dir / f"{split}.json").items()):
            if annotation.get("frame_ann") is None:
                counts[f"{split}_seq_ann_only_excluded"] += 1
                continue
            source = babel_feat_p_to_amass_relpath(annotation["feat_p"])
            if source in source_splits:
                raise ValueError(f"Duplicate or split-overlapping frame source: {source}")
            source_splits[source] = split
            counts[f"{split}_frame_sources"] += 1
            conversion = conversions.get(source)
            if conversion is None or not (input_dir / conversion["output_relpath"]).is_file():
                errors.append({"kind": "missing_converted_source", "split": split,
                               "babel_sid": sid, "source_amass_relpath": source})
                continue
            if limit is not None and selected >= limit:
                continue
            works.append({"split": split, "sid": sid, "annotation": annotation,
                          "source": source, "conversion": conversion})
            selected += 1
    return works, errors, dict(counts)


def _init_worker(input_dir: Path, output: Path, class_names: tuple[str, ...]):
    global _INPUT, _OUTPUT, _SKELETON, _CLASSES
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _INPUT, _OUTPUT = input_dir, output
    _SKELETON, _CLASSES = SOMASkeleton30(), class_names


def _convert_source(work: dict):
    assert _INPUT is not None and _OUTPUT is not None and _SKELETON is not None
    conversion = work["conversion"]
    try:
        source_fps = int(conversion.get("source_fps", -1))
        method = conversion.get("resampling_method", "fixed_step")
        if int(conversion.get("target_fps", -1)) != FPS or source_fps <= 0:
            raise ValueError("Conversion manifest FPS mismatch")
        if method == "fixed_step":
            if source_fps % FPS or conversion.get("frame_step") != source_fps // FPS:
                raise ValueError("Invalid fixed-step conversion provenance")
        elif method == "lerp_slerp":
            if source_fps % FPS == 0 or conversion.get("frame_step") is not None:
                raise ValueError("Invalid interpolation conversion provenance")
        else:
            raise ValueError(f"Unsupported source resampling: {method}")
        try:
            labels, supervision, flags = rasterize_frame_labels(
                work["annotation"], int(conversion["output_num_frames"]), _CLASSES,
            )
        except ValueError as error:
            # Some official annotations contain reversed temporal bounds. Do
            # not guess corrected times or silently label their whole source.
            source_frames = int(conversion["output_num_frames"])
            full_windows, remainder = divmod(source_frames, NUM_FRAMES)
            return {"ok": False, "kind": "invalid_frame_annotation", "split": work["split"],
                    "babel_sid": work["sid"], "source_amass_relpath": work["source"],
                    "source_frames": source_frames,
                    "excluded_windows": full_windows + int(remainder >= MIN_FRAMES),
                    "excluded_window_frames": full_windows * NUM_FRAMES + (remainder if remainder >= MIN_FRAMES else 0),
                    "error": str(error)}
        rotations, roots, parsed_fps = parse_bvh_motion(_INPUT / conversion["output_relpath"])
        if round_fps(float(parsed_fps)) != FPS:
            raise ValueError("Converted BVH is not 30 FPS")
        if rotations.shape != (len(roots), 77, 3, 3) or roots.shape != (len(rotations), 3):
            raise ValueError("Converted BVH is not SOMA77")
        if not torch.isfinite(rotations).all() or not torch.isfinite(roots).all():
            raise ValueError("Nonfinite BVH values")
        if int(conversion["output_num_frames"]) != len(rotations):
            raise ValueError("BVH frame count differs from conversion manifest")
        rotations = _SKELETON.from_soma77(rotations)
        representation = MotionJEPAMotionRep(_SKELETON, FPS)
        records = []
        dropped_tail_frames = 0
        for window_index, start in enumerate(range(0, len(rotations), NUM_FRAMES)):
            end = min(start + NUM_FRAMES, len(rotations))
            if end - start < MIN_FRAMES:
                dropped_tail_frames += end - start
                continue
            sample_id = f"{work['sid']:05d}/window_{window_index:04d}"
            motion = representation(rotations[start:end], roots[start:end], to_canonicalize=True)
            if tuple(motion.shape) != (end - start, 366) or not torch.isfinite(motion).all():
                raise ValueError("Invalid encoded motion")
            label_relative = Path("labels") / work["split"] / f"{sample_id}.npz"
            label_path = _OUTPUT / label_relative
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("xb") as file:
                np.savez_compressed(file, labels=labels[start:end], supervision=supervision[start:end])
            record = {
                "id": sample_id, "source_id": str(work["sid"]), "babel_sid": work["sid"],
                "split": work["split"], "source_amass_relpath": work["source"],
                "source_path": conversion["output_relpath"], "start_frame": start, "end_frame": end,
                "fps": FPS, "length": end - start, "motion_dim": 366,
                "labels_path": label_relative.as_posix(), "labels_sha256": file_sha256(label_path),
                "supervised_frames": int(supervision[start:end].sum()),
                "frame_counts": {name: int(value[start:end].sum()) for name, value in flags.items()},
                "metadata": {"annotation_type": "frame_ann", **_conversion_metadata(conversion)},
                "motion": motion.cpu().numpy(),
            }
            record = _save_record_motion(record, _OUTPUT)
            record["motion_sha256"] = file_sha256(_OUTPUT / record["motion_path"])
            records.append(record)
        return {"ok": True, "split": work["split"], "records": records,
                "dropped_tail_frames": dropped_tail_frames}
    except Exception as error:
        return {"ok": False, "kind": "conversion_failure", "split": work["split"],
                "babel_sid": work["sid"], "source_amass_relpath": work["source"],
                "error": f"{type(error).__name__}: {error}"}


def preprocess(args: argparse.Namespace):
    args.input_dir = Path(args.input_dir).resolve()
    args.annotations_dir = Path(args.annotations_dir).resolve()
    args.output = Path(args.output).resolve()
    args.manifest = Path(args.manifest).resolve() if args.manifest else args.input_dir / "conversion_manifest.jsonl"
    args.action_label_map = Path(args.action_label_map).resolve()
    if not 1 <= args.workers <= 8 or args.chunksize < 1:
        raise ValueError("Use 1..8 workers and a positive chunksize")
    class_names = load_action_labels(args.action_label_map, 120)
    class_names60 = load_action_labels(args.action_label_map, 60)
    provenance = {
        "segmentation_format": FORMAT, "supervision_policy": SUPERVISION_POLICY,
        "preprocessing_version": 1, "min_frames": MIN_FRAMES,
        "num_frames": NUM_FRAMES, "fps": FPS,
        "class_names": list(class_names), "class_names_60": list(class_names60),
        "class_indices_60": [class_names.index(name) for name in class_names60],
        "num_classes": len(class_names), "limit_per_split": args.limit,
        "conversion_manifest_sha256": file_sha256(args.manifest),
        "action_label_map_sha256": file_sha256(args.action_label_map),
        "annotation_sha256": {split: file_sha256(args.annotations_dir / f"{split}.json")
                              for split in ("train", "val")},
    }
    if args.output.exists() and not args.overwrite and _validate_complete_dataset(args.output):
        previous = json.loads((args.output / "meta.json").read_text())
        if any(previous.get(key) != value for key, value in provenance.items()):
            raise ValueError("Existing segmentation provenance mismatch; use a new output directory")
        for split in SPLITS:
            dataset = BabelSegmentationDataset(args.output, split, normalize=False)
            for index in range(len(dataset)):
                dataset[index]
        print(f"Reusing verified BABEL segmentation dataset: {args.output}")
        return previous
    conversions = load_conversion_manifest(args.manifest)
    works, errors, discovery = build_work_items(args.input_dir, args.annotations_dir, conversions, args.limit)
    if not works:
        raise RuntimeError("No converted BABEL sources with frame annotations")
    _prepare_output(args.output, args.overwrite)
    records = {split: [] for split in SPLITS}
    success = {split: 0 for split in SPLITS}
    dropped = Counter()
    init_args = (args.input_dir, args.output, class_names)
    if args.workers == 1:
        _init_worker(*init_args)
        results = map(_convert_source, works)
        pool = None
    else:
        pool = Pool(args.workers, initializer=_init_worker, initargs=init_args)
        results = pool.imap(_convert_source, works, chunksize=args.chunksize)
    try:
        for result in tqdm(results, total=len(works), desc="BABEL frame segmentation", unit="source"):
            if not result["ok"]:
                errors.append(result)
                continue
            split = result["split"]
            records[split].extend(result["records"])
            success[split] += bool(result["records"])
            dropped[split] += result["dropped_tail_frames"]
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    failures = [error for error in errors if error["kind"] == "conversion_failure"]
    if failures:
        (args.output / "errors.jsonl").write_text("".join(json.dumps(error) + "\n" for error in errors))
        raise RuntimeError(f"{len(failures)} unexpected BABEL conversion failures: {failures[0]}")
    args.num_frames, args.fps, args.overlap, args.split_seed = NUM_FRAMES, FPS, 0.0, None
    frame_counts = {}
    for split in SPLITS:
        frame_counts[split] = {
            "total": sum(record["length"] for record in records[split]),
            "supervised": sum(record["supervised_frames"] for record in records[split]),
            **{name: sum(record["frame_counts"][name] for record in records[split])
               for name in ("annotated", "transition", "oov")},
            "dropped_short_tail_frames": dropped[split],
        }
    finalize_processed_dataset(
        args, records, errors, success, source_dataset="BABEL-120_frame_segmentation_fixed_identity_soma77",
        segmentation="continuous_nonoverlapping_source_windows",
        metadata_extra={
            **provenance, "split_policy": "official_babel_train_val_test_empty",
            "frame_interval": "floor_seconds_times_30_start_inclusive_end_exclusive",
            "tail_policy": "retain_at_least_30_frames_then_zero_pad_in_loader",
            "annotation_policy": "frame_ann_only_no_sequence_label_broadcast",
            "invalid_annotation_policy": "exclude_entire_source_and_record_error",
            "unsupervised_policy": "retain_input_ignore_loss_metrics_for_no_120_positive",
            "resampled": False, "resampling_stage": "amass_to_soma",
            "downsampling": "reuse_converted_30fps_without_resampling",
            "fps_validation": "converted_bvh_must_be_30fps",
            "discovery_counts": discovery, "frame_counts": frame_counts,
            "error_counts": dict(Counter(error["kind"] for error in errors)),
            "invalid_annotation_sources": [error for error in errors if error["kind"] == "invalid_frame_annotation"],
            "conversion_manifest": str(args.manifest), "annotations_dir": str(args.annotations_dir),
        },
    )
    metadata_path = args.output / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["index_sha256"] = file_sha256(args.output / "index.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    for split in SPLITS:
        dataset = BabelSegmentationDataset(args.output, split, normalize=False)
        for index in range(len(dataset)):
            dataset[index]
    print(json.dumps({"frame_counts": frame_counts, "index_sha256": metadata["index_sha256"]}, indent=2))
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "dataset/amass-soma77")
    parser.add_argument("--annotations-dir", type=Path, default=PROJECT_ROOT / "dataset/babel-annotation")
    parser.add_argument("--action-label-map", type=Path,
                        default=PROJECT_ROOT / "dataset/babel-60-and-120/action_label_2_idx.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Maximum converted sources per split for fixtures/previews")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
