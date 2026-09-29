from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import MotionDataset, preprocess_babel
from dataset.convert_amass_to_soma import write_bvh_atomic
from experiment.linear_probe.dataset import load_classification_label_index
from skeleton import SOMASkeleton77
from visualization.dataset_viewer import discover_entries, load_motion


def _write_bvh(path: Path, frames: int) -> None:
    skeleton = SOMASkeleton77()
    rotations = skeleton.relaxed_hands.expand(frames, -1, -1, -1).clone()
    roots = torch.zeros(frames, 3)
    roots[:, 0] = torch.linspace(0.0, 1.0, frames)
    write_bvh_atomic(path, rotations, roots, skeleton)


def _write_labels(path: Path, rows: list[tuple[str, int, int, int, str]]) -> None:
    columns = list(zip(*rows)) if rows else [(), (), (), (), ()]
    with path.open("wb") as file:
        pickle.dump((list(columns[0]), tuple(list(column) for column in columns[1:])), file)


def _write_fixture(root: Path) -> argparse.Namespace:
    input_dir = root / "bvh"
    annotations = root / "annotations"
    labels = root / "labels"
    output = root / "processed"
    input_dir.mkdir()
    annotations.mkdir()
    labels.mkdir()
    segment_id = "11111111-1111-1111-1111-111111111111"
    annotator = "22222222-2222-2222-2222-222222222222"
    annotation = {
        "1": {
            "babel_sid": 1,
            "feat_p": "ACCAD/ACCAD/Female1General_c3d/A1 - Stand_poses.npz",
            "dur": 6.0,
            "frame_ann": {
                "anntr_id": annotator,
                "labels": [{
                    "seg_id": segment_id,
                    "start_t": 0.0,
                    "end_t": 6.0,
                    "act_cat": ["walk", "stand"],
                }],
            },
            "seq_ann": None,
        }
    }
    (annotations / "train.json").write_text(json.dumps(annotation), encoding="utf-8")
    (annotations / "val.json").write_text("{}", encoding="utf-8")
    _write_labels(
        labels / "train_label_60.pkl",
        [
            (segment_id, 0, 1, 1, annotator),
            (segment_id, 1, 1, 1, annotator),
            (segment_id, 1, 1, 1, annotator),
        ],
    )
    _write_labels(labels / "val_label_60.pkl", [])
    source_relpath = "ACCAD/Female1General_c3d/A1_-_Stand_stageii.npz"
    bvh_relpath = source_relpath.replace(".npz", ".bvh")
    _write_bvh(input_dir / bvh_relpath, 180)
    conversion = {
        "status": "converted",
        "source_relpath": source_relpath,
        "output_relpath": bvh_relpath,
        "source_fps_raw": 120.0,
        "source_fps": 120,
        "target_fps": 30,
        "frame_step": 4,
        "source_num_frames": 720,
        "output_num_frames": 180,
        "mean_vertex_error_m": 0.004,
        "motion_correction": "applied",
    }
    (input_dir / "conversion_manifest.jsonl").write_text(
        json.dumps(conversion) + "\n", encoding="utf-8"
    )
    (input_dir / "errors.jsonl").write_text("", encoding="utf-8")
    source_map = preprocess_babel.DEFAULT_LABELS / "action_label_2_idx.json"
    (labels / "action_label_2_idx.json").write_text(
        source_map.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return argparse.Namespace(
        subset=60,
        input_dir=input_dir,
        manifest=None,
        annotations_dir=annotations,
        labels_dir=labels,
        action_label_map=None,
        output=output,
        workers=1,
        chunksize=2,
        limit=None,
        overwrite=False,
    )


def test_feat_p_mapping_uses_archive_aliases_and_filename_normalization() -> None:
    assert preprocess_babel.babel_feat_p_to_amass_relpath(
        "ACCAD/ACCAD/Female1General_c3d/A1 - Stand_poses.npz"
    ) == "ACCAD/Female1General_c3d/A1_-_Stand_stageii.npz"
    assert preprocess_babel.babel_feat_p_to_amass_relpath(
        "Transitionsmocap/Transitions_mocap/mazen_c3d/devil1_poses.npz"
    ) == "Transitions/mazen_c3d/devil1_stageii.npz"
    with pytest.raises(ValueError, match="_poses"):
        preprocess_babel.babel_feat_p_to_amass_relpath("CMU/CMU/01/motion.npz")


def test_action_label_map_selects_official_numeric_prefix() -> None:
    names = preprocess_babel.load_action_labels(
        preprocess_babel.DEFAULT_LABELS / "action_label_2_idx.json", 60
    )
    assert len(names) == 60
    assert names[:4] == ("walk", "stand", "hand movements", "turn")
    assert names[-1] == "martial art"


def test_preprocess_shares_multilabel_motion_and_zero_pads_tail(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    preprocess_babel.preprocess(args)

    index = json.loads((args.output / "index.json").read_text(encoding="utf-8"))
    assert len(index) == 2
    assert {record["metadata"]["label"] for record in index} == {0, 1}
    assert index[0]["motion_path"] == index[1]["motion_path"]
    assert index[0]["length"] == index[1]["length"] == 30
    motion_files = list((args.output / "motions/train").rglob("*.npy"))
    assert len(motion_files) == 1
    stored = np.load(motion_files[0], allow_pickle=False)
    assert stored.shape == (30, 366)
    assert stored.dtype == np.float32
    assert np.isfinite(stored).all()
    assert not (args.output / "test.txt").read_text(encoding="utf-8")

    dataset = MotionDataset(args.output, "train.txt", num_frames=150, fps=30, motion_dim=366)
    padded, fps, valid_length = dataset[0]
    assert padded.shape == (150, 366)
    assert fps == 30
    assert valid_length == 30
    assert np.array_equal(padded[:30], stored)
    assert np.count_nonzero(padded[30:]) == 0

    label_index = load_classification_label_index(args.output)
    assert label_index.num_classes == 60
    assert label_index.class_names[:2] == ("walk", "stand")
    labels = label_index.label_for_sample(index[0]["motion_path"])
    assert torch.equal(labels[:2], torch.tensor([1.0, 1.0]))
    assert int(labels.sum()) == 2
    viewer_entries = discover_entries(args.output, "train", limit=0)
    decoded, decoded_fps = load_motion(viewer_entries[0], fps=30)
    assert decoded_fps == 30
    assert decoded["local_rot_mats"].shape == (30, 77, 3, 3)


def test_missing_conversion_is_skipped_with_label_count(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    labels = tmp_path / "labels"
    input_dir = tmp_path / "input"
    for path in (annotations, labels, input_dir):
        path.mkdir()
    segment_id = "segment"
    annotator = "annotator"
    item = {
        "babel_sid": 7,
        "feat_p": "CMU/CMU/01/01_01_poses.npz",
        "dur": 1.0,
        "frame_ann": {
            "anntr_id": annotator,
            "labels": [{"seg_id": segment_id, "start_t": 0.0, "end_t": 1.0}],
        },
    }
    (annotations / "train.json").write_text(json.dumps({"7": item}), encoding="utf-8")
    (annotations / "val.json").write_text("{}", encoding="utf-8")
    _write_labels(labels / "train_label_60.pkl", [(segment_id, 0, 7, 0, annotator)])
    _write_labels(labels / "val_label_60.pkl", [])
    works, errors, counts = preprocess_babel.build_work_items(
        subset=60,
        annotations_dir=annotations,
        labels_dir=labels,
        conversions={},
        conversion_errors={},
        input_dir=input_dir,
        limit=None,
    )
    assert works == []
    assert errors[0]["kind"] == "upstream_missing"
    assert errors[0]["skipped_label_rows"] == 1
    assert counts["upstream_missing_motions"] == 1


def _resize_fixture(args: argparse.Namespace, frames: int, *, include_first: bool = True) -> None:
    manifest_path = args.input_dir / "conversion_manifest.jsonl"
    conversion = json.loads(manifest_path.read_text())
    conversion["source_num_frames"] = frames * 4
    conversion["output_num_frames"] = frames
    manifest_path.write_text(json.dumps(conversion) + "\n")
    _write_bvh(args.input_dir / conversion["output_relpath"], frames)
    annotation_path = args.annotations_dir / "train.json"
    annotation = json.loads(annotation_path.read_text())
    item = annotation["1"]
    item["dur"] = frames / 30
    segment = item["frame_ann"]["labels"][0]
    segment["end_t"] = frames / 30
    annotation_path.write_text(json.dumps(annotation))
    _write_labels(args.labels_dir / "train_label_60.pkl", [
        (segment["seg_id"], label, 1, chunk, item["frame_ann"]["anntr_id"])
        for chunk in ((0, 1) if include_first else (1,)) for label in (0, 1)
    ])


@pytest.mark.parametrize("tail,min_frames,kept", [(29, 30, False), (30, 30, True),
    (31, 30, True), (15, 30, False), (29, 29, True), (30, 31, False),
    (1, 1, True), (149, 150, False), (150, 150, True)])
def test_minimum_chunk_length_filters_files_labels_and_stats(
    tmp_path: Path, tail: int, min_frames: int, kept: bool,
) -> None:
    args = _write_fixture(tmp_path)
    _resize_fixture(args, 150 + tail)
    args.min_frames = min_frames
    preprocess_babel.preprocess(args)
    index = json.loads((args.output / "index.json").read_text())
    meta = json.loads((args.output / "meta.json").read_text())
    errors = [json.loads(line) for line in (args.output / "errors.jsonl").read_text().splitlines()]
    assert len(index) == (4 if kept else 2)
    assert len(list((args.output / "motions").rglob("*.npy"))) == (2 if kept else 1)
    assert all(row["length"] >= min_frames for row in index)
    assert meta["train_stats_frames"] == 2 * (150 + (tail if kept else 0))
    assert meta["min_frames"] == min_frames
    assert meta["num_skipped_label_rows"] == (0 if kept else 2)
    expected = np.concatenate([np.load(args.output / row["motion_path"]) for row in index])
    np.testing.assert_allclose(np.load(args.output / "stats/mean.npy"), expected.mean(axis=0, dtype=np.float64), atol=1e-6)
    if not kept:
        assert errors[0]["kind"] == "short_chunk"
        assert errors[0]["valid_length"] == tail
        assert errors[0]["skipped_label_rows"] == 2
        assert errors[0]["min_frames"] == min_frames


def test_short_chunk_after_clamping_to_bvh_is_skipped(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    _resize_fixture(args, 179)
    path = args.annotations_dir / "train.json"
    annotation = json.loads(path.read_text())
    annotation["1"]["frame_ann"]["labels"][0]["end_t"] = 10.0
    path.write_text(json.dumps(annotation))
    preprocess_babel.preprocess(args)
    assert len(json.loads((args.output / "index.json").read_text())) == 2
    errors = [json.loads(line) for line in (args.output / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["valid_length"] == 29


def test_all_short_training_chunks_fail_without_stats(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    _resize_fixture(args, 165, include_first=False)
    with pytest.raises(RuntimeError, match="no BABEL training samples.*minimum is 30"):
        preprocess_babel.preprocess(args)
    assert not list(args.output.rglob("*.npy"))
    error = json.loads((args.output / "errors.jsonl").read_text())
    assert error["kind"] == "short_chunk"
    assert error["valid_length"] == 15
    assert error["skipped_label_rows"] == 2


def test_subframe_segment_is_logged_as_empty_chunk(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    path = args.annotations_dir / "train.json"
    annotation = json.loads(path.read_text())
    item = annotation["1"]
    item["frame_ann"]["labels"].append({
        "seg_id": "short-segment", "start_t": 0.0, "end_t": 0.01,
    })
    path.write_text(json.dumps(annotation))
    annotator = item["frame_ann"]["anntr_id"]
    _write_labels(args.labels_dir / "train_label_60.pkl", [
        (segment, label, 1, chunk, annotator)
        for segment, chunk in ((item["frame_ann"]["labels"][0]["seg_id"], 1), ("short-segment", 0))
        for label in (0, 1)
    ])
    preprocess_babel.preprocess(args)
    assert len(json.loads((args.output / "index.json").read_text())) == 2
    error = json.loads((args.output / "errors.jsonl").read_text())
    assert error["kind"] == "empty_chunk"
    assert error["valid_length"] == 0
    assert error["skipped_label_rows"] == 2


@pytest.mark.parametrize("method,fps,step", [(None, 120, 4), ("fixed_step", 60, 2),
    ("lerp_slerp", 100, None), ("lerp_slerp", 20, None)])
def test_legacy_and_new_manifest_policies(
    tmp_path: Path, method: str | None, fps: int, step: int | None,
) -> None:
    args = _write_fixture(tmp_path)
    path = args.input_dir / "conversion_manifest.jsonl"
    manifest = json.loads(path.read_text())
    manifest.update(source_fps=fps, source_fps_raw=float(fps), frame_step=step, source_num_frames=fps * 6)
    if method is not None:
        manifest["resampling_method"] = method
    path.write_text(json.dumps(manifest) + "\n")
    preprocess_babel.preprocess(args)
    metadata = json.loads((args.output / "index.json").read_text())[0]["metadata"]
    assert metadata["frame_step"] == step
    assert metadata.get("resampling_method") == method


@pytest.mark.parametrize("method,fps,step", [("unknown", 100, None),
    ("lerp_slerp", 120, None), ("lerp_slerp", 100, 3), ("fixed_step", 100, 3)])
def test_invalid_resampling_manifests_are_rejected(tmp_path: Path, method: str, fps: int, step: int | None) -> None:
    args = _write_fixture(tmp_path)
    path = args.input_dir / "conversion_manifest.jsonl"
    manifest = json.loads(path.read_text())
    manifest.update(resampling_method=method, source_fps=fps, frame_step=step)
    path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(RuntimeError, match="no BABEL training samples"):
        preprocess_babel.preprocess(args)


@pytest.mark.parametrize("change", ["min_frames", "version", "legacy", "manifest", "subset"])
def test_complete_dataset_reuse_requires_matching_policy(tmp_path: Path, change: str) -> None:
    args = _write_fixture(tmp_path)
    preprocess_babel.preprocess(args)
    index_path = args.output / "index.json"
    original_mtime = index_path.stat().st_mtime_ns
    preprocess_babel.preprocess(args)
    assert index_path.stat().st_mtime_ns == original_mtime
    if change == "min_frames":
        args.min_frames = 31
    elif change == "subset":
        args.subset = 120
    elif change == "manifest":
        with args.manifest.open("a") as file:
            file.write("\n")
    else:
        path = args.output / "meta.json"
        meta = json.loads(path.read_text())
        if change == "legacy":
            del meta["preprocessing_version"]
        else:
            meta["preprocessing_version"] = -1
        path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="new --output directory or explicit --overwrite"):
        preprocess_babel.preprocess(args)
    assert index_path.stat().st_mtime_ns == original_mtime


def test_explicit_overwrite_applies_new_min_frames(tmp_path: Path) -> None:
    args = _write_fixture(tmp_path)
    _resize_fixture(args, 180)
    preprocess_babel.preprocess(args)
    args.min_frames = 31
    args.overwrite = True
    preprocess_babel.preprocess(args)
    assert len(json.loads((args.output / "index.json").read_text())) == 2
    assert json.loads((args.output / "meta.json").read_text())["min_frames"] == 31


@pytest.mark.parametrize("minimum", [0, -1, 151])
def test_invalid_min_frames_is_rejected(tmp_path: Path, minimum: int) -> None:
    args = _write_fixture(tmp_path)
    args.min_frames = minimum
    with pytest.raises(ValueError, match="--min-frames must be between"):
        preprocess_babel.preprocess(args)


def test_min_frames_cli_default() -> None:
    assert preprocess_babel.build_parser().parse_args(["--subset", "60"]).min_frames == 30
