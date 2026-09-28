"""Regression tests for NPY-backed BONES-SEED preprocessing."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dataset import preprocess_bones_seed as preprocessing


class MotionFPSTest(unittest.TestCase):
    @staticmethod
    def _motion(frames: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
        rotations = torch.arange(frames).view(frames, 1, 1, 1).expand(-1, 1, 3, 3)
        positions = torch.arange(frames * 3).view(frames, 3)
        return rotations, positions

    def test_matching_and_exact_multiple_fps(self):
        rotations, positions = self._motion()
        actual_rotations, actual_positions, fps = preprocessing.resample_motion_fps(
            rotations, positions, 60, 60
        )
        self.assertIs(actual_rotations, rotations)
        self.assertIs(actual_positions, positions)
        self.assertEqual(fps, 60)
        down_rotations, down_positions, _ = preprocessing.resample_motion_fps(
            rotations, positions, 180, 60
        )
        torch.testing.assert_close(down_rotations, rotations[::3])
        torch.testing.assert_close(down_positions, positions[::3])

    def test_lower_and_non_multiple_fps_are_rejected(self):
        rotations, positions = self._motion()
        with self.assertRaisesRegex(ValueError, "below configured FPS"):
            preprocessing.resample_motion_fps(rotations, positions, 30, 60)
        with self.assertRaisesRegex(ValueError, "exact multiple"):
            preprocessing.resample_motion_fps(rotations, positions, 90, 60)

    def test_stride_uses_half_up_rounding(self):
        self.assertEqual(preprocessing.calculate_stride(300, 0.5), 150)
        self.assertEqual(preprocessing.calculate_stride(3, 0.5), 2)


class _SourceSkeleton:
    def to_standard_tpose(self, rotations):
        return rotations, None


class _TargetSkeleton:
    def from_soma77(self, rotations):
        return rotations


class _RecordingRepresentation:
    FEATURE_DIM = 366
    FORMAT_NAME = "motion_jepa_366_v1"
    calls: list[tuple[int, torch.Tensor]] = []

    def __init__(self, skeleton, fps):
        self.fps = fps

    def __call__(self, rotations, positions, to_canonicalize):
        self.calls.append((self.fps, positions.clone()))
        return positions[:, :1].expand(-1, self.FEATURE_DIM).to(torch.float32)


class ConvertOneClipTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temporary.name)
        (self.dataset_root / "bvh").mkdir()
        (self.dataset_root / "bvh/example.bvh").write_text("fixture", encoding="utf-8")
        self.item = preprocessing.WorkItem(
            id="example",
            split="train",
            bvh_path="bvh/example.bvh",
            captions=("caption",),
            metadata={"category": "test"},
        )
        preprocessing._DATASET_ROOT = self.dataset_root
        preprocessing._NUM_FRAMES = 4
        preprocessing._MIN_FRAMES = 2
        preprocessing._FPS = 60
        preprocessing._STRIDE_FRAMES = 2
        preprocessing._SOURCE_SKELETON = _SourceSkeleton()
        preprocessing._TARGET_SKELETON = _TargetSkeleton()
        _RecordingRepresentation.calls.clear()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _parsed_motion(frames: int, fps: float = 60):
        rotations = torch.arange(frames).view(frames, 1, 1, 1).expand(-1, 1, 3, 3)
        positions = torch.arange(frames * 3).view(frames, 3)
        return rotations, positions, fps

    def _convert(self, frames: int, fps: float = 60):
        with (
            mock.patch.object(
                preprocessing,
                "parse_bvh_motion",
                return_value=self._parsed_motion(frames, fps),
            ),
            mock.patch.object(preprocessing, "MotionJEPAMotionRep", _RecordingRepresentation),
        ):
            return preprocessing._convert_one(self.item)

    def test_complete_overlapping_windows_are_independently_encoded(self):
        result = self._convert(8)
        self.assertTrue(result["ok"])
        self.assertEqual(
            [(record["start_frame"], record["end_frame"]) for record in result["records"]],
            [(0, 4), (2, 6), (4, 8)],
        )
        self.assertEqual(
            [record["id"] for record in result["records"]],
            ["example_0000", "example_0001", "example_0002"],
        )
        self.assertEqual(len(_RecordingRepresentation.calls), 3)
        for record in result["records"]:
            self.assertEqual(record["motion"].dtype, np.float32)
            self.assertEqual(record["motion"].shape, (4, 366))

    def test_variable_tail_and_short_motion_rules(self):
        preprocessing._STRIDE_FRAMES = 3
        result = self._convert(6)
        self.assertEqual(
            [(record["start_frame"], record["end_frame"]) for record in result["records"]],
            [(0, 4), (4, 6)],
        )
        short = self._convert(3)
        self.assertEqual(short["records"][0]["id"], "example")
        self.assertEqual(short["records"][0]["length"], 3)
        below = self._convert(1)
        self.assertFalse(below["ok"])
        self.assertIn("Too few frames", below["error"])


def _args(root: Path, output: Path, *, overwrite: bool = False):
    return argparse.Namespace(
        dataset_root=root,
        splits_root=root,
        metadata_csv=root / "metadata.csv",
        output=output,
        workers=1,
        chunksize=16,
        num_frames=4,
        min_frames=2,
        fps=60,
        overlap=0.5,
        split_seed=42,
        max_per_split=-1,
        overwrite=overwrite,
    )


def _fake_results(items, _args):
    for item in items:
        value = float(ord(item.id[-1]))
        motion = np.full((2, 366), value, dtype=np.float32)
        yield {
            "ok": True,
            "id": item.id,
            "split": item.split,
            "records": [
                {
                    "id": item.id,
                    "source_id": item.id,
                    "segment_index": 0,
                    "start_frame": 0,
                    "end_frame": 2,
                    "split": item.split,
                    "source_path": item.bvh_path,
                    "source_fps": 60,
                    "fps": 60,
                    "length": 2,
                    "motion_dim": 366,
                    "captions": list(item.captions),
                    "metadata": item.metadata,
                    "motion": motion,
                }
            ],
        }


class DirectNPYPreprocessTest(unittest.TestCase):
    def _build(self, root: Path, output: Path, *, overwrite: bool = False):
        with (
            mock.patch.object(
                preprocessing,
                "build_splits",
                return_value={"train": ["source-a", "source-b", "source-c"], "val": [], "test": []},
            ),
            mock.patch.object(preprocessing, "_ordered_results", side_effect=_fake_results),
        ):
            preprocessing.preprocess(_args(root, output, overwrite=overwrite))

    def test_direct_output_values_manifests_metadata_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "processed"
            self._build(root, output)
            self.assertTrue((output / "motions").is_dir())
            self.assertFalse((output / "stats/stats.json").exists())
            mean = np.load(output / "stats/mean.npy", allow_pickle=False)
            std = np.load(output / "stats/std.npy", allow_pickle=False)
            self.assertEqual(mean.shape, (366,))
            self.assertEqual(std.shape, (366,))
            self.assertEqual(mean.dtype, np.float32)
            self.assertEqual(std.dtype, np.float32)
            rows = (output / "train.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            ids = []
            for row in rows:
                sample_id, path, fps, length = row.split(",")
                ids.append(sample_id)
                self.assertEqual(path, f"motions/train/{sample_id}.npy")
                self.assertEqual(fps, "60")
                self.assertEqual(length, "2")
                value = np.load(output / path, allow_pickle=False)
                self.assertEqual(value.shape, (2, 366))
                self.assertEqual(value.dtype, np.float32)
            manifest = json.loads((output / "motions/train.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "motion_jepa_npy_v1")
            self.assertEqual(manifest["num_samples"], 3)
            metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["motion_storage"], "npy_float32_v1")
            self.assertEqual(
                metadata["split_format"],
                "sample_id,relative_npy_path,fps,actual_length",
            )
            self.assertEqual(metadata["train_stats_frames"], 6)
            index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            self.assertNotIn("motion", index[0])
            self.assertEqual(index[0]["motion_path"], "motions/train/source-a.npy")
            self.assertEqual((output / "errors.jsonl").read_text(encoding="utf-8"), "")

    def test_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            self._build(root, first)
            self._build(root, second)
            self.assertEqual((first / "train.txt").read_bytes(), (second / "train.txt").read_bytes())
            self.assertEqual((first / "index.json").read_bytes(), (second / "index.json").read_bytes())
            np.testing.assert_array_equal(
                np.load(first / "motions/train/source-a.npy"),
                np.load(second / "motions/train/source-a.npy"),
            )

    def test_complete_reuse_direct_partial_handling_and_old_storage_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "processed"
            self._build(root, output)
            self.assertFalse((root / ".processed.npy-staging").exists())
            self.assertFalse((output / preprocessing.BUILD_MARKER).exists())
            original = (output / "index.json").read_bytes()
            self._build(root, output)
            self.assertEqual((output / "index.json").read_bytes(), original)
            partial = root / "partial"
            partial.mkdir()
            (partial / preprocessing.BUILD_MARKER).write_text("recognized", encoding="utf-8")
            (partial / "visible-during-build").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "incomplete"):
                self._build(root, partial)
            self._build(root, partial, overwrite=True)
            self.assertTrue((partial / "motions/train/source-a.npy").is_file())
            self.assertFalse((partial / preprocessing.BUILD_MARKER).exists())
            old = root / "old"
            (old / "lmdb").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "unrecognized"):
                self._build(root, old)
            self._build(root, old, overwrite=True)
            self.assertTrue((old / "motions/train/source-a.npy").is_file())
            self.assertFalse((old / "lmdb").exists())


class NPYRecordSaveTest(unittest.TestCase):
    def test_nested_sample_path_and_statistics_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            motion = np.full((3, 366), 2.0, dtype=np.float32)
            record = preprocessing._save_record_motion(
                {
                    "id": "actor/walk_0000",
                    "split": "train",
                    "length": 3,
                    "motion": motion,
                },
                Path(directory),
            )
            self.assertEqual(record["motion_path"], "motions/train/actor/walk_0000.npy")
            self.assertNotIn("motion", record)
            np.testing.assert_array_equal(
                np.load(Path(directory) / record["motion_path"]), motion
            )
            np.testing.assert_allclose(record["_stats_sum"], motion.sum(axis=0))


if __name__ == "__main__":
    unittest.main()
