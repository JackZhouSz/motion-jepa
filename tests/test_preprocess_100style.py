"""Tests for official-cut, content-disjoint 100STYLE preprocessing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dataset.motion_dataset import MotionDataset
from dataset import preprocess_100style as preprocessing


def _write_timing_header(path: Path, frames: int = 1800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"HIERARCHY\nMOTION\nFrames: {frames}\nFrame Time: 0.0166666667\n",
        encoding="utf-8",
    )


def _write_frame_cuts(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["STYLE_NAME"] + [
        f"{content}_{endpoint}"
        for content in preprocessing.KNOWN_CONTENTS
        for endpoint in ("START", "STOP")
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for partial in rows:
            row = {field: "N/A" for field in fields}
            row.update(partial)
            writer.writerow(row)


class FrameCutAndWindowPlanningTest(unittest.TestCase):
    def test_frame_cut_parser_handles_na_and_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.csv"
            _write_frame_cuts(valid, [{"STYLE_NAME": "Style", "BR_START": 10, "BR_STOP": 110}])
            self.assertEqual(
                preprocessing.load_frame_cuts(valid),
                {"Style_BR": preprocessing.FrameCut(10, 110)},
            )
            cases = (
                ("Duplicate style", [
                    {"STYLE_NAME": "Style", "BR_START": 10, "BR_STOP": 110},
                    {"STYLE_NAME": "Style", "BR_START": 20, "BR_STOP": 120},
                ]),
                ("Incomplete", [{"STYLE_NAME": "Style", "BR_START": 10, "BR_STOP": "N/A"}]),
                ("Invalid", [{"STYLE_NAME": "Style", "BR_START": 110, "BR_STOP": 10}]),
            )
            for index, (message, rows) in enumerate(cases):
                path = root / f"bad-{index}.csv"
                _write_frame_cuts(path, rows)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    preprocessing.load_frame_cuts(path)

    def test_selected_content_limit_and_non_overlapping_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for content in ("BR", "FW", "ID"):
                _write_timing_header(root / f"bvh/Aeroplane_{content}_soma77.bvh")
            sources = preprocessing.discover_sources(root, 1, ("FW",))
            self.assertEqual([source.id for source in sources], ["Aeroplane_FW"])
            selected = preprocessing.discover_sources(root, -1, ("BR", "FW"))
            cuts = {
                "Aeroplane_BR": preprocessing.FrameCut(100, 1000),
                "Aeroplane_FW": preprocessing.FrameCut(200, 1100),
            }
            plan, errors = preprocessing.build_window_plan(selected, root, 90, 30, cuts, "FW")
            self.assertFalse(errors)
            self.assertEqual([len(item.windows) for item in plan], [5, 5])
            self.assertEqual({w.split for w in plan[0].windows}, {"train"})
            self.assertEqual({w.split for w in plan[1].windows}, {"test"})
            self.assertEqual(
                [(w.start_frame, w.end_frame) for w in plan[0].windows],
                [(0, 90), (90, 180), (180, 270), (270, 360), (360, 450)],
            )

    def test_trim_occurs_before_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            item = preprocessing.PlannedSource(
                preprocessing.SourceMotion("Style_BR", "bvh/Style_BR_soma77.bvh", "Style", "BR"),
                60, 20, 4, 16, 6,
                (preprocessing.WindowDescriptor("Style_BR", 0, 0, 6, "train"),),
            )
            values = torch.arange(20.0)[:, None]

            class Skeleton:
                def from_soma77(self, value):
                    return value

            class Representation:
                FEATURE_DIM = 366
                def __init__(self, skeleton, fps):
                    del skeleton, fps
                def __call__(self, local, roots, to_canonicalize):
                    del roots, to_canonicalize
                    return torch.zeros((len(local), 366))

            def resample(local, roots, source_fps, target_fps):
                self.assertEqual(local[:, 0].tolist(), list(range(4, 16)))
                self.assertEqual((source_fps, target_fps), (60, 30))
                return local[::2], roots[::2], 30

            preprocessing._DATASET_ROOT = root
            preprocessing._OUTPUT_ROOT = output
            preprocessing._TARGET_SKELETON = Skeleton()
            with mock.patch.object(preprocessing, "parse_bvh_motion", return_value=(values, values, 60)), \
                 mock.patch.object(preprocessing, "resample_motion_fps", side_effect=resample), \
                 mock.patch.object(preprocessing, "MotionJEPAMotionRep", Representation):
                result = preprocessing._convert_source(item)
            self.assertTrue(result["ok"], result)
            metadata = result["records"][0]["metadata"]
            self.assertEqual((metadata["source_start_frame"], metadata["source_stop_frame"]), (4, 16))


class Preprocess100StyleOutputTest(unittest.TestCase):
    def test_output_has_fw_test_and_empty_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root, output = root / "source", root / "processed"
            cuts_path = root / "Frame_Cuts.csv"
            for content in ("BR", "FW"):
                _write_timing_header(dataset_root / f"bvh/Aeroplane_{content}_soma77.bvh")
            _write_frame_cuts(cuts_path, [{
                "STYLE_NAME": "Aeroplane", "BR_START": 100, "BR_STOP": 1000,
                "FW_START": 200, "FW_STOP": 1100,
            }])
            args = argparse.Namespace(
                dataset_root=dataset_root, frame_cuts=cuts_path, output=output,
                workers=1, chunksize=1, num_frames=90, fps=30,
                contents=["BR", "FW"], test_content="FW", limit=-1, overwrite=False,
            )

            def fake_results(items, _args):
                for item in items:
                    records = []
                    for window in item.windows:
                        records.append({
                            "id": f"{item.source.id}_{window.segment_index:04d}",
                            "source_id": item.source.id, "segment_index": window.segment_index,
                            "start_frame": window.start_frame, "end_frame": window.end_frame,
                            "split": window.split, "source_path": item.source.bvh_path,
                            "source_fps": 60, "fps": 30, "length": 90, "motion_dim": 366,
                            "metadata": {"style": item.source.style,
                                         "motion_code": item.source.motion_code,
                                         "source_id": item.source.id},
                            "motion": np.full((90, 366), window.segment_index + 1, np.float32),
                        })
                    yield {"ok": True, "id": item.source.id, "records": records}

            with mock.patch.object(preprocessing, "_ordered_results", side_effect=fake_results):
                preprocessing.preprocess(args)
            metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["train_contents"], ["BR"])
            self.assertEqual(metadata["test_contents"], ["FW"])
            self.assertFalse(metadata["validation_enabled"])
            self.assertEqual(metadata["split_unit"], "source_content")
            self.assertEqual(metadata["split_counts"], {"train": 5, "val": 0, "test": 5})
            self.assertEqual(metadata["frame_cuts_sha256"], hashlib.sha256(cuts_path.read_bytes()).hexdigest())
            self.assertEqual((output / "val.txt").read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads((output / "motions/val.json").read_text())["num_samples"], 0)
            index = json.loads((output / "index.json").read_text(encoding="utf-8"))
            by_source = {}
            for record in index:
                by_source.setdefault(record["source_id"], set()).add(record["split"])
            self.assertEqual(by_source, {"Aeroplane_BR": {"train"}, "Aeroplane_FW": {"test"}})
            self.assertEqual(len(MotionDataset(output, "train.txt", 90, 30, motion_dim=366)), 5)


if __name__ == "__main__":
    unittest.main()
