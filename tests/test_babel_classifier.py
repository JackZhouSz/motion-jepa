"""BABEL multi-label data and classifier integration checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiment.linear_probe import train_classifier
from experiment.linear_probe.dataset import (
    BabelLabelIndex,
    build_classification_datasets,
)
from test_linear_probe import _write_checkpoint, _write_dataset


def _write_babel_dataset(root: Path) -> None:
    _write_dataset(root, num_frames=4, motion_dim=6)
    names = [f"action-{index}" for index in range(60)]
    rows = {}
    for split in ("train", "val", "test"):
        rows.update({line.split(",", 1)[0]: line.split(",")[1]
                     for line in (root / f"{split}.txt").read_text().splitlines()})
    records = json.loads((root / "index.json").read_text())
    for record in records:
        label = 0 if "-a" in record["id"] else 1
        if record["id"] == "val-b":
            label = 2  # A validation-positive class absent from training.
        record["metadata"] = {"label": label, "label_name": names[label]}
        record["motion_path"] = rows[record["id"]]
    records = [record for record in records if record["split"] != "test"]
    duplicate = dict(next(record for record in records if record["id"] == "train-a1"))
    duplicate["id"] = "train-a1-second-label"
    duplicate["metadata"] = {"label": 1, "label_name": names[1]}
    records.append(duplicate)
    (root / "index.json").write_text(json.dumps(records), encoding="utf-8")
    meta = json.loads((root / "meta.json").read_text())
    meta.update(
        source_dataset="BABEL-60_fixed_identity_soma77",
        subset=60,
        num_classes=60,
        class_names=names,
    )
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "class-index.json").write_text(json.dumps({
        "class_names": names,
        "class_to_index": {name: index for index, name in enumerate(names)},
    }), encoding="utf-8")
    for split, extra in (("train", "train-a1-second-label"), ("test", None)):
        path = root / f"{split}.txt"
        if extra is None:
            path.write_text("", encoding="utf-8")
        else:
            path.write_text(path.read_text() + f"{extra},{rows['train-a1']},30,4\n")
        manifest_path = root / "motions" / f"{split}.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["num_samples"] = len(path.read_text().splitlines())
        manifest["split_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _args(root: Path, output: Path, *, source: str = "raw", checkpoint: Path | None = None):
    return argparse.Namespace(
        model="linear" if source == "raw" else "cnn",
        input_source=source,
        jepa_checkpoint=checkpoint,
        checkpoint_key="target_encoder",
        stats_path=None,
        feature_batch_size=4,
        feature_cache_root=None,
        recompute_features=False,
        dataset_root=root,
        output_root=output,
        findings_root=None,
        device="cpu", seed=42, epochs=2, warmup_epochs=0,
        batch_size=4, num_workers=0, lr=3e-4, final_lr=1e-6,
        weight_decay=0.05, gradient_clip=1.0, use_bfloat16=False,
        resume=False, overwrite=False,
    )


class BabelClassifierTest(unittest.TestCase):
    def test_unique_chunk_and_official_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_babel_dataset(root)
            datasets, index = build_classification_datasets(
                root, num_frames=4, fps=30, motion_dim=6, stats_root=root / "stats",
            )
            self.assertIsInstance(index, BabelLabelIndex)
            self.assertEqual((len(datasets["train"]), len(datasets["val"]), len(datasets["test"])), (4, 2, 0))
            self.assertEqual(index.num_classes, 60)
            targets = {sample_id: label for label, sample_id in zip(
                datasets["train"].labels, datasets["train"].sample_ids,
            )}
            torch.testing.assert_close(targets["motions/train/train-a1.npy"][:3], torch.tensor([1., 1., 0.]))
            self.assertEqual(datasets["train"][1][0].shape, (4, 6))

    def test_absent_test_files_are_an_empty_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_babel_dataset(root)
            (root / "test.txt").unlink()
            (root / "motions/test.json").unlink()
            datasets, _ = build_classification_datasets(
                root, num_frames=4, fps=30, motion_dim=6, stats_root=root / "stats",
            )
            self.assertEqual(len(datasets["test"]), 0)

    def test_ap_excludes_absent_classes_and_hits_use_any_positive(self):
        metric = train_classifier.MultiLabelMetricAccumulator(3)
        metric.update(
            torch.tensor([[.9, .1, 0.], [.8, .9, 0.], [.1, .8, 0.]]),
            torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]),
            3.0,
        )
        result = metric.compute()
        self.assertAlmostEqual(result.mean_average_precision, (5/6 + 1)/2)
        self.assertAlmostEqual(result.top1_hit, 2/3)
        self.assertEqual(result.top5_hit, 1.0)
        self.assertEqual(result.classes_without_positives, 1)

    def test_raw_training_empty_test_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            output = Path(directory) / "out"
            _write_babel_dataset(root)
            args = _args(root, output)
            original_save = train_classifier._atomic_torch_save

            def interrupt(value, path):
                original_save(value, path)
                if path.name == "classifier-latest.pth.tar" and value.get("next_epoch") == 1:
                    raise RuntimeError("interrupted")

            with mock.patch.object(train_classifier, "_atomic_torch_save", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    train_classifier.run(args)
            args.resume = True
            summary = train_classifier.run(args)["linear"]
            self.assertEqual(summary["selection_metric"], "val_mean_average_precision")
            self.assertIsNone(summary["test"])
            self.assertEqual(summary["split_counts"], {"train": 4, "val": 2, "test": 0})
            self.assertIn(2, summary["missing_train_class_ids"])
            self.assertEqual(summary["best_val"]["classes_with_positives"], 2)

    def test_jepa_token_cache_has_multihot_labels_and_no_test_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            output = Path(directory) / "out"
            _write_babel_dataset(root)
            checkpoint = Path(directory) / "pretrain" / "latest.pth.tar"
            checkpoint.parent.mkdir()
            _write_checkpoint(checkpoint, root / "stats", num_frames=4, motion_dim=6)
            args = _args(root, output, source="jepa", checkpoint=checkpoint)
            summary = train_classifier.run(args)["cnn"]
            cache = checkpoint.parent / "linear-probe/token-features/babel-60"
            payload = torch.load(cache / "train.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["labels"].shape, (4, 60))
            self.assertEqual(payload["labels"].dtype, torch.float32)
            self.assertFalse((cache / "test.pt").exists())
            self.assertIsNone(summary["test"])

    def test_both_networks_write_report_without_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            output = Path(directory) / "out"
            findings = Path(directory) / "findings"
            _write_babel_dataset(root)
            args = _args(root, output)
            args.model = "all"
            args.epochs = 1
            args.findings_root = findings
            summaries = train_classifier.run(args)
            self.assertEqual(set(summaries), {"cnn", "transformer"})
            self.assertTrue(all(summary["test"] is None for summary in summaries.values()))
            report = (findings / "README.md").read_text()
            self.assertIn("BABEL-60 multi-label", report)
            self.assertIn("Val mean AP", report)
            self.assertTrue((findings / "training-curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
