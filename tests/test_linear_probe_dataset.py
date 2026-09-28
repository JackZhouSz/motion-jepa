"""Tests for shared 100STYLE style labels and datasets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment.linear_probe.dataset import (
    StyleMotionDataset,
    build_style_datasets,
    load_style_label_index,
)
from test_linear_probe import _write_dataset


class StyleLabelIndexTest(unittest.TestCase):
    def test_style_names_are_sorted_into_stable_class_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"id": "z", "metadata": {"style": "Zesty"}},
                {"id": "a", "metadata": {"style": "Angry"}},
                {"id": "m", "metadata": {"style": "Angry"}},
            ]
            (root / "index.json").write_text(json.dumps(records), encoding="utf-8")
            index = load_style_label_index(root)
            self.assertEqual(index.class_names, ("Angry", "Zesty"))
            self.assertEqual(index.class_to_index, {"Angry": 0, "Zesty": 1})
            self.assertEqual(index.label_for_sample("m"), 0)
            self.assertEqual(index.label_for_sample("z"), 1)

    def test_duplicate_and_missing_style_records_are_rejected(self):
        cases = (
            [
                {"id": "same", "metadata": {"style": "A"}},
                {"id": "same", "metadata": {"style": "B"}},
            ],
            [{"id": "sample", "metadata": {}}],
            [{"metadata": {"style": "A"}}],
            ["not-an-object"],
        )
        for records in cases:
            with self.subTest(records=records), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "index.json").write_text(json.dumps(records), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_style_label_index(root)

    def test_explicit_numeric_labels_preserve_declared_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"id": "z", "metadata": {"label": 0, "label_name": "Zebra"}},
                {"id": "a", "metadata": {"label": 1, "label_name": "Apple"}},
            ]
            (root / "index.json").write_text(json.dumps(records), encoding="utf-8")
            (root / "meta.json").write_text(
                json.dumps({"class_names": ["Zebra", "Apple"]}), encoding="utf-8"
            )
            index = load_style_label_index(root)
            self.assertEqual(index.class_names, ("Zebra", "Apple"))
            self.assertEqual(index.class_to_index, {"Zebra": 0, "Apple": 1})
            self.assertEqual(index.label_for_sample("z"), 0)
            self.assertEqual(index.label_for_sample("a"), 1)


class StyleMotionDatasetTest(unittest.TestCase):
    def test_builder_shares_labels_and_returns_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_dataset(root)
            stats = root / "pretrain-stats"
            stats.mkdir()
            np.save(stats / "mean.npy", np.zeros(6, dtype=np.float32))
            np.save(stats / "std.npy", np.ones(6, dtype=np.float32))
            datasets, index = build_style_datasets(
                root,
                num_frames=4,
                fps=30,
                motion_dim=6,
                stats_root=stats,
            )
            self.assertEqual(index.class_to_index, {"A": 0, "B": 1})
            self.assertTrue(all(isinstance(value, StyleMotionDataset) for value in datasets.values()))
            motion, fps, length, label, sample_id = datasets["train"][0]
            self.assertEqual(motion.shape, (4, 6))
            self.assertEqual((fps, length, label, sample_id), (30, 4, 0, "train-a0"))


if __name__ == "__main__":
    unittest.main()
