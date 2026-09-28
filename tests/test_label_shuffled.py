"""Tests for the removable raw label-shuffled negative control."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from experiment.linear_probe.label_shuffled import (
    CONTROL_NAME,
    LabelShuffledDataset,
    make_label_derangement,
    run,
)
from test_linear_probe import _write_dataset


class _ToyDataset:
    labels = [0, 0, 1, 2, 2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, 30, 1, self.labels[index], f"sample-{index}"


class LabelShuffledControlTest(unittest.TestCase):
    def test_derangement_is_reproducible_and_has_no_fixed_classes(self):
        first = make_label_derangement(100, 42)
        second = make_label_derangement(100, 42)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(range(100)))
        self.assertTrue(all(source != target for source, target in enumerate(first)))

    def test_all_windows_of_a_style_receive_the_same_fake_label(self):
        wrapped = LabelShuffledDataset(_ToyDataset(), (2, 0, 1))
        self.assertEqual(wrapped.labels, [2, 2, 0, 1, 1])
        self.assertEqual(wrapped[0][3], wrapped[1][3])
        self.assertEqual(wrapped[0][4], "sample-0")

    def test_control_trains_with_original_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            output_root = root / "output"
            _write_dataset(dataset_root, validation_enabled=False)
            args = argparse.Namespace(
                model="linear",
                input_source="raw",
                dataset_root=dataset_root,
                output_root=output_root,
                findings_root=None,
                device="cpu",
                seed=7,
                label_shuffle_seed=11,
                epochs=2,
                warmup_epochs=0,
                batch_size=4,
                num_workers=0,
                lr=3.0e-4,
                final_lr=1.0e-6,
                weight_decay=0.05,
                gradient_clip=1.0,
                use_bfloat16=False,
                resume=False,
                overwrite=False,
            )
            summary = run(args)
            self.assertEqual(summary["control"], CONTROL_NAME)
            self.assertEqual(summary["label_shuffle_seed"], 11)
            self.assertFalse(summary["test_labels_shuffled"])
            self.assertEqual(summary["train_label_mapping"], {"A": "B", "B": "A"})
            self.assertIn("seed=11", summary["signature"]["input_source"])
            self.assertTrue(
                (output_root / "linear/seed-7/classifier-final.pth.tar").is_file()
            )


if __name__ == "__main__":
    unittest.main()
