from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dataset.babel_segmentation import BabelSegmentationDataset, file_sha256
from dataset.convert_amass_to_soma import write_bvh_atomic
from dataset.preprocess_babel_segmentation import (
    build_work_items, preprocess, rasterize_frame_labels,
)
from skeleton import SOMASkeleton77


CLASSES = tuple(f"action_{index:03d}" for index in range(120))


def _annotation(sid, name, frames):
    return {
        "babel_sid": sid, "feat_p": f"KIT/KIT/subject/{name}_poses.npz",
        "dur": frames / 30,
        "seq_ann": {"labels": [{"act_cat": [CLASSES[2]]}]},
        "frame_ann": {"labels": [
            {"start_t": 0, "end_t": 1, "act_cat": [CLASSES[0]]},
            {"start_t": .5, "end_t": 1.5, "act_cat": [CLASSES[1], CLASSES[0]]},
            {"start_t": 1.5, "end_t": 2, "act_cat": ["transition"]},
            {"start_t": 3, "end_t": 4, "act_cat": ["unknown activity"]},
            {"start_t": 5, "end_t": frames / 30, "act_cat": [CLASSES[70]]},
        ]},
    }


def _fixture(tmp_path):
    torch.set_num_threads(1)
    input_dir = tmp_path / "bvh"
    annotations = tmp_path / "annotations"
    input_dir.mkdir()
    annotations.mkdir()
    label_map = tmp_path / "action_label_2_idx.json"
    label_map.write_text(json.dumps({f"action_{index:03d}": index for index in range(150)}))
    skeleton = SOMASkeleton77()
    conversions = []
    for sid, split, name, frames in [(1, "train", "train_motion", 365), (2, "val", "val_motion", 165)]:
        relative = f"KIT/subject/{name}_stageii.bvh"
        rotations = skeleton.relaxed_hands.expand(frames, -1, -1, -1).clone()
        roots = torch.zeros(frames, 3)
        roots[:, 0] = torch.linspace(0, 1, frames)
        write_bvh_atomic(input_dir / relative, rotations, roots, skeleton)
        annotation = _annotation(sid, name, frames)
        records = {str(sid): annotation}
        if split == "train":
            sequence_only = _annotation(3, "sequence_only", 150)
            sequence_only["frame_ann"] = None
            records["3"] = sequence_only
        (annotations / f"{split}.json").write_text(json.dumps(records))
        conversions.append({
            "status": "converted", "source_relpath": relative.replace(".bvh", ".npz"),
            "output_relpath": relative, "target_fps": 30, "source_fps": 120,
            "frame_step": 4, "output_num_frames": frames,
        })
    (input_dir / "conversion_manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in conversions))
    return argparse.Namespace(
        input_dir=input_dir, annotations_dir=annotations, manifest=None,
        action_label_map=label_map, output=tmp_path / "processed", workers=1,
        chunksize=1, limit=None, overwrite=False,
    )


def test_rasterization_unions_overlaps_and_excludes_unlabelled_frames():
    labels, valid, flags = rasterize_frame_labels(_annotation(1, "motion", 365), 365, CLASSES)
    assert labels.shape == (365, 120)
    assert labels[15:30, :2].all()
    assert not valid[45:150].any()  # transition, OOV and annotation gaps
    assert flags["transition"][45:60].all()
    assert flags["oov"][90:120].all()
    assert not flags["annotated"][60:90].any()
    assert valid[150:].all() and labels[150:, 70].all()
    assert not labels[:, 2].any()  # sequence label was never broadcast


def test_rasterization_uses_floor_half_open_intervals():
    annotation = {"frame_ann": {"labels": [{"start_t": .049, "end_t": .099, "act_cat": [CLASSES[0]]}]}}
    labels, valid, _ = rasterize_frame_labels(annotation, 9, CLASSES)
    assert np.flatnonzero(valid).tolist() == [1]
    assert labels[1, 0]
    with pytest.raises(ValueError, match="frame_ann"):
        rasterize_frame_labels({"seq_ann": annotation["frame_ann"]}, 9, CLASSES)
    annotation["frame_ann"]["labels"][0]["end_t"] = 0
    with pytest.raises(ValueError, match="Malformed frame annotation"):
        rasterize_frame_labels(annotation, 9, CLASSES)


def test_continuous_windows_loader_padding_hashes_and_external_stats(tmp_path):
    args = _fixture(tmp_path)
    metadata = preprocess(args)
    assert metadata["split_counts"] == {"train": 3, "val": 1, "test": 0}
    assert metadata["frame_counts"]["val"]["dropped_short_tail_frames"] == 15
    assert metadata["discovery_counts"]["train_seq_ann_only_excluded"] == 1
    assert metadata["index_sha256"] == file_sha256(args.output / "index.json")
    stats = tmp_path / "pretrain_stats"
    stats.mkdir()
    np.save(stats / "mean.npy", np.ones(366, dtype=np.float32))
    np.save(stats / "std.npy", np.full(366, 2, dtype=np.float32))
    dataset = BabelSegmentationDataset(args.output, "train", stats_root=stats)
    assert dataset.class_names == CLASSES
    assert dataset.class_indices_60 == tuple(range(60))
    assert [record["start_frame"] for record in dataset.records] == [0, 150, 300]
    first = dataset[0]
    assert first[0].shape == (150, 366) and first[3].shape == (150, 120)
    assert not first[4][45:150].any()
    final = dataset[2]
    assert final[2] == 65 and final[4][:65].all() and not final[4][65:].any()
    assert not final[3][:65, :60].any() and final[3][:65, 70].all()
    assert not final[0][65:].any() and not final[3][65:].any()
    raw = np.load(dataset.entries[2].path)
    np.testing.assert_allclose(final[0][:65], (raw - 1) / 2)
    assert len(BabelSegmentationDataset(args.output, "test", normalize=False)) == 0
    assert preprocess(args)["index_sha256"] == metadata["index_sha256"]
    labels_path = args.output / dataset.records[0]["labels_path"]
    with labels_path.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="label hash mismatch"):
        dataset[0]


def test_stale_index_and_annotation_provenance_are_rejected(tmp_path):
    args = _fixture(tmp_path)
    preprocess(args)
    annotation_path = args.annotations_dir / "train.json"
    annotation_path.write_text(annotation_path.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance mismatch"):
        preprocess(args)
    index = args.output / "index.json"
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="index hash mismatch"):
        BabelSegmentationDataset(args.output, "train", normalize=False)


def test_duplicate_source_across_splits_is_rejected(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    for split, sid in [("train", 1), ("val", 2)]:
        (annotations / f"{split}.json").write_text(json.dumps({str(sid): _annotation(sid, "same", 150)}))
    with pytest.raises(ValueError, match="split-overlapping"):
        build_work_items(tmp_path, annotations, {})


def test_malformed_official_interval_excludes_source_with_audit_record(tmp_path):
    args = _fixture(tmp_path)
    path = args.annotations_dir / "val.json"
    annotation = json.loads(path.read_text())
    annotation["2"]["frame_ann"]["labels"][0].update(start_t=2, end_t=1)
    path.write_text(json.dumps(annotation))
    metadata = preprocess(args)
    assert metadata["split_counts"]["val"] == 0
    assert metadata["error_counts"] == {"invalid_frame_annotation": 1}
    errors = [json.loads(line) for line in (args.output / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["babel_sid"] == 2 and errors[0]["kind"] == "invalid_frame_annotation"
    assert errors[0]["excluded_windows"] == 1 and errors[0]["excluded_window_frames"] == 150
    assert metadata["invalid_annotation_sources"] == errors
