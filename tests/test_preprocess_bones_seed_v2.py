"""Valid-length BONES-SEED v2 preprocessing and protocol compatibility tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from dataset import preprocess_bones_seed_v2 as preprocessing
from dataset.convert_amass_to_soma import write_bvh_atomic
from dataset.motion_dataset import MotionDataset
from skeleton import SOMASkeleton77


@pytest.mark.parametrize(
    "length,expected",
    [
        (59, []),
        (60, [(0, 60)]),
        (149, [(0, 149)]),
        (150, [(0, 150)]),
        (151, [(0, 150), (1, 151)]),
        (224, [(0, 150), (74, 224)]),
        (225, [(0, 150), (75, 225)]),
        (226, [(0, 150), (75, 225), (76, 226)]),
    ],
)
def test_short_sources_and_end_aligned_windows(length, expected):
    windows = preprocessing.select_windows(length, 150, 60, 75)
    assert windows == expected
    assert len(windows) == len(set(windows))
    if windows:
        covered = np.zeros(length, dtype=bool)
        for start, end in windows:
            covered[start:end] = True
            assert 60 <= end - start <= 150
        assert covered.all()


def test_non_overlapping_stride_recovers_tail_with_a_full_window():
    assert preprocessing.select_windows(310, 150, 60, 150) == [
        (0, 150), (150, 300), (160, 310),
    ]
    with pytest.raises(ValueError, match="Invalid minimum"):
        preprocessing.select_windows(150, 150, 151, 75)


def _fixture(tmp_path: Path):
    source_root = tmp_path / "source"
    split_root = tmp_path / "splits"
    split_root.mkdir()
    skeleton = SOMASkeleton77()
    names = []
    for frames in (59, 60, 149, 151):
        name = f"motion{frames}"
        names.append(name)
        rotations = skeleton.relaxed_hands.expand(frames, -1, -1, -1).clone()
        roots = torch.zeros(frames, 3)
        roots[:, 0] = torch.arange(frames, dtype=torch.float32) / 30
        write_bvh_atomic(source_root / f"bvh/{name}.bvh", rotations, roots, skeleton)
    (split_root / preprocessing.TRAIN_SPLIT_FILE).write_text("\n".join(names) + "\n")
    for name in preprocessing.HELDOUT_SPLIT_FILES:
        (split_root / name).write_text("")
    return argparse.Namespace(
        dataset_root=source_root,
        splits_root=split_root,
        metadata_csv=tmp_path / "metadata.csv",
        output=tmp_path / "processed-v2",
        workers=1,
        chunksize=1,
        num_frames=150,
        min_frames=60,
        fps=30,
        overlap=0.5,
        split_seed=42,
        max_per_split=-1,
        overwrite=False,
    )


def test_real_bvh_conversion_stats_and_loader_padding(tmp_path):
    args = _fixture(tmp_path)
    preprocessing.preprocess(args)
    metadata = json.loads((args.output / "meta.json").read_text())
    assert metadata["min_frames"] == 60
    assert metadata["min_seconds"] == 2
    assert metadata["preprocessing_version"] == preprocessing.PREPROCESSING_VERSION
    assert metadata["segmentation"] == preprocessing.SEGMENTATION_POLICY
    assert metadata["split_counts"] == {"train": 4, "val": 0, "test": 0}
    assert metadata["retained_source_split_counts"]["train"] == 3
    assert metadata["excluded_short_source_split_counts"]["train"] == 1
    assert metadata["excluded_short_frame_split_counts"]["train"] == 59
    assert metadata["num_errors"] == 0
    assert metadata["train_stats_frames"] == 60 + 149 + 150 + 150
    assert preprocessing._validate_complete_dataset(args.output)

    records = json.loads((args.output / "index.json").read_text())
    assert [r["length"] for r in records] == [60, 149, 150, 150]
    assert [(r["start_frame"], r["end_frame"]) for r in records[-2:]] == [(0, 150), (1, 151)]
    arrays = [np.load(args.output / record["motion_path"]) for record in records]
    assert all(array.dtype == np.float32 and np.isfinite(array).all() for array in arrays)
    expected_mean = np.concatenate(arrays).astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(np.load(args.output / "stats/mean.npy"), expected_mean, atol=1e-6)

    dataset = MotionDataset(args.output, "train.txt", num_frames=150, fps=30, normalize=True)
    motion, fps, length = dataset[0]
    assert length == 60
    assert fps == 30
    assert motion.shape == (150, 366)
    assert not motion[60:].any()
    mean = np.load(args.output / "stats/mean.npy")
    std = np.load(args.output / "stats/std.npy")
    np.testing.assert_allclose(motion[:60], (arrays[0] - mean) / std, atol=1e-5)


def test_complete_reuse_requires_matching_protocol_and_input_hashes(tmp_path):
    args = _fixture(tmp_path)
    preprocessing.preprocess(args)
    with mock.patch.object(preprocessing, "_ordered_results", side_effect=AssertionError("re-extraction")):
        preprocessing.preprocess(args)
    for key, changed in (("min_frames", 61), ("num_frames", 180), ("fps", 60), ("overlap", 0.25), ("split_seed", 7), ("max_per_split", 2)):
        mismatch = argparse.Namespace(**vars(args))
        setattr(mismatch, key, changed)
        with pytest.raises(ValueError, match="protocol does not match"):
            preprocessing.preprocess(mismatch)
    split_path = args.splits_root / preprocessing.TRAIN_SPLIT_FILE
    split_path.write_text(split_path.read_text() + "\n")
    with pytest.raises(ValueError, match="protocol does not match"):
        preprocessing.preprocess(args)


def test_old_preprocessing_output_is_not_silently_reused(tmp_path):
    args = _fixture(tmp_path)
    preprocessing.preprocess(args)
    path = args.output / "meta.json"
    metadata = json.loads(path.read_text())
    metadata.pop("preprocessing_version")
    path.write_text(json.dumps(metadata))
    with pytest.raises(FileExistsError, match="use --overwrite"):
        preprocessing.preprocess(args)


def test_parallel_workers_have_identical_order_values_and_statistics(tmp_path):
    args = _fixture(tmp_path)
    preprocessing.preprocess(args)
    first_output = args.output
    args.output = tmp_path / "processed-parallel"
    args.workers = 2
    preprocessing.preprocess(args)
    assert (first_output / "train.txt").read_bytes() == (args.output / "train.txt").read_bytes()
    assert (first_output / "index.json").read_bytes() == (args.output / "index.json").read_bytes()
    assert (first_output / "meta.json").read_bytes() == (args.output / "meta.json").read_bytes()
    for record in json.loads((first_output / "index.json").read_text()):
        np.testing.assert_array_equal(
            np.load(first_output / record["motion_path"]),
            np.load(args.output / record["motion_path"]),
        )
    for name in ("mean.npy", "std.npy"):
        np.testing.assert_array_equal(
            np.load(first_output / "stats" / name),
            np.load(args.output / "stats" / name),
        )


def test_minimum_is_checked_after_fps_downsampling(tmp_path):
    args = _fixture(tmp_path)
    preprocessing._init_worker(str(args.dataset_root), str(args.output), 150, 30, 75, 60)
    skeleton = SOMASkeleton77()
    item = preprocessing.WorkItem("motion60", "train", "bvh/motion60.bvh", {})
    for source_frames, expected in ((236, []), (240, [60])):
        rotations = skeleton.relaxed_hands.expand(source_frames, -1, -1, -1).clone()
        roots = torch.zeros(source_frames, 3)
        with mock.patch.object(preprocessing, "parse_bvh_motion", return_value=(rotations, roots, 120)):
            result = preprocessing._convert_one(item)
        assert result["ok"]
        assert [record["length"] for record in result["records"]] == expected


def test_cli_defaults_and_minimum_validation():
    args = preprocessing.parse_args([])
    assert args.min_frames == 60 and args.fps == 30 and args.num_frames == 150
    assert args.output.name == "bones-seed-processed-v2-nframes150-min60"
    assert preprocessing.parse_args(["--min-frames", "90"]).min_frames == 90
    for minimum in (0, 151):
        args.min_frames = minimum
        with pytest.raises(ValueError, match="--min_frames"):
            preprocessing._validate_config(args)
