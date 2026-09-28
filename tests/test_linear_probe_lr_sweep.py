"""Tests for the resumable all-checkpoint linear-probe sweep."""

from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiment.linear_probe import lr_sweep as sweep
from test_linear_probe import _write_checkpoint, _write_dataset


class SweepUnitTest(unittest.TestCase):
    def test_validation_free_dataset_rejects_lr_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_dataset(root, validation_enabled=False)
            with self.assertRaisesRegex(ValueError, "validation-free"):
                sweep.require_validation_dataset(root)

    def test_model_size_ordering_uses_encoder_then_predictor(self):
        run_names = [
            "mot_small-base_1d-bs.512-ep.300",
            "mot_huge-large_1d-bs.512-ep.300",
            "mot_huge-giant_1d-bs.512-ep.300",
            "mot_huge_1d-bs.512-ep.300",
        ]
        self.assertEqual(
            sorted(run_names, key=sweep.run_architecture_sort_key),
            [
                "mot_huge-giant_1d-bs.512-ep.300",
                "mot_huge_1d-bs.512-ep.300",
                "mot_huge-large_1d-bs.512-ep.300",
                "mot_small-base_1d-bs.512-ep.300",
            ],
        )

    def test_default_findings_root_is_unified_component(self):
        default = sweep.build_parser().get_default("findings_root")
        self.assertEqual(
            default,
            sweep.PROJECT_ROOT
            / "findings/000-100style-classification/linear-probe",
        )

    def test_discovers_only_direct_child_latest_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("run-b", "run-a"):
                run = root / name
                run.mkdir()
                (run / "model-latest.pth.tar").touch()
            nested = root / "run-a" / "linear-probe"
            nested.mkdir()
            (nested / "ignored-latest.pth.tar").touch()
            (root / "no-checkpoint").mkdir()
            discovered = sweep.discover_latest_checkpoints(root)
            self.assertEqual([name for name, _ in discovered], ["run-a", "run-b"])

    def test_adaptive_extraction_halves_batch_size_after_oom(self):
        expected = {"features": torch.zeros(1, 2)}
        with mock.patch.object(
            sweep.features,
            "load_or_extract_split",
            side_effect=[RuntimeError("CUDA out of memory"), expected],
        ) as extract:
            payload, used = sweep.load_or_extract_adaptive(
                split="train",
                cache_path=Path("missing.pt"),
                metadata={},
                dataset=mock.Mock(),
                encoder=mock.Mock(),
                device=torch.device("cuda"),
                initial_batch_size=8,
                num_workers=0,
                use_bfloat16=True,
                recompute=False,
            )
        self.assertIs(payload, expected)
        self.assertEqual(used, 4)
        self.assertEqual(
            [call.kwargs["batch_size"] for call in extract.call_args_list], [8, 4]
        )


class SweepEndToEndTest(unittest.TestCase):
    def test_explicit_checkpoint_diagnostic_without_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            dataset_root = root / "dataset"
            findings_root = root / "findings"
            run_root = output_root / "run-a"
            run_root.mkdir(parents=True)
            _write_dataset(dataset_root, validation_enabled=False)
            pretrain_stats = root / "pretrain-stats"
            pretrain_stats.mkdir()
            np.save(pretrain_stats / "mean.npy", np.zeros(6, dtype=np.float32))
            np.save(pretrain_stats / "std.npy", np.ones(6, dtype=np.float32))
            checkpoint = run_root / "motion-jepa-ep50.pth.tar"
            _write_checkpoint(checkpoint, pretrain_stats)

            args = argparse.Namespace(
                output_root=output_root,
                checkpoints=[checkpoint],
                dataset_root=dataset_root,
                findings_root=findings_root,
                lrs=[0.1],
                seeds=[42],
                pooling="valid_token_mean",
                diagnostic_no_validation=True,
                epochs=2,
                batch_size=2,
                feature_batch_size=4,
                num_workers=0,
                momentum=0.9,
                weight_decay=0.0,
                device="cpu",
                recompute_features=False,
                overwrite_runs=False,
            )
            result = sweep.run(args)
            self.assertEqual(result["num_completed"], 1)
            self.assertTrue((findings_root / "test-top1-heatmap.png").is_file())
            self.assertFalse(
                (findings_root / "validation-top1-heatmap.png").exists()
            )
            with (findings_root / "sweep-results.csv").open() as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["val_top1_accuracy"], "nan")
            final_heads = list(
                (findings_root / "runs").rglob("linear-probe-final.pth.tar")
            )
            self.assertEqual(len(final_heads), 1)

    def test_two_checkpoint_sweep_reports_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            dataset_root = root / "dataset"
            findings_root = root / "findings"
            output_root.mkdir()
            _write_dataset(dataset_root)
            pretrain_stats = root / "pretrain-stats"
            pretrain_stats.mkdir()
            np.save(pretrain_stats / "mean.npy", np.zeros(6, dtype=np.float32))
            np.save(pretrain_stats / "std.npy", np.ones(6, dtype=np.float32))
            for run_name in ("run-a", "run-b"):
                run_root = output_root / run_name
                run_root.mkdir()
                _write_checkpoint(
                    run_root / "motion-jepa-latest.pth.tar", pretrain_stats
                )

            args = argparse.Namespace(
                output_root=output_root,
                dataset_root=dataset_root,
                findings_root=findings_root,
                lrs=[0.03, 0.1],
                seeds=[0, 1],
                epochs=2,
                batch_size=2,
                feature_batch_size=4,
                num_workers=0,
                momentum=0.9,
                weight_decay=0.0,
                device="cpu",
                recompute_features=False,
                overwrite_runs=False,
            )
            result = sweep.run(args)
            self.assertEqual(result["num_checkpoints"], 2)
            self.assertEqual(result["num_completed"], 8)
            expected_counts = {
                "sweep-results.csv": 8,
                "aggregate-results.csv": 4,
                "epoch-metrics.csv": 16,
            }
            for filename, expected in expected_counts.items():
                with (findings_root / filename).open(encoding="utf-8") as file:
                    self.assertEqual(len(list(csv.DictReader(file))), expected)
            for filename in (
                "README.md",
                "sweep-config.json",
                "validation-top1-heatmap.png",
                "test-top1-heatmap.png",
                "plots/run-a.png",
                "plots/run-b.png",
            ):
                self.assertTrue((findings_root / filename).is_file(), filename)
            metrics = findings_root / "runs/run-a/lr-0p03/seed-0/metrics.csv"
            modified = metrics.stat().st_mtime_ns
            resumed = sweep.run(args)
            self.assertEqual(resumed["num_completed"], 8)
            self.assertEqual(metrics.stat().st_mtime_ns, modified)


if __name__ == "__main__":
    unittest.main()
