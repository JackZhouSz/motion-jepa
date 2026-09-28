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
from experiment.linear_probe.dataset import load_style_label_index
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

    label_index = load_style_label_index(args.output)
    assert label_index.num_classes == 60
    assert label_index.class_names[:2] == ("walk", "stand")
    assert [label_index.label_for_sample(record["id"]) for record in index] == [0, 1]
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
