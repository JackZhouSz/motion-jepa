"""Tests for the unified 100STYLE findings report."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiment.linear_probe import report


def _aggregate(run: str, lr: float, val_top1: float, test_top1: float) -> dict[str, str]:
    row = {
        "run_name": run,
        "model_name": "mot_tiny_1d",
        "lr": str(lr),
        "num_seeds": "1",
        "best_epoch_mean": "4",
        "best_epoch_std": "0",
    }
    for split, top1 in (("val", val_top1), ("test", test_top1)):
        values = {
            "loss": 0.5,
            "top1_accuracy": top1,
            "macro_accuracy": top1 - 0.01,
            "top5_accuracy": min(1.0, top1 + 0.1),
        }
        for metric, value in values.items():
            row[f"{split}_{metric}_mean"] = str(value)
            row[f"{split}_{metric}_std"] = "0"
    return row


def _summary(name: str, val_top1: float, test_top1: float) -> dict:
    metrics = lambda top1: {
        "loss": 0.2,
        "top1_accuracy": top1,
        "macro_accuracy": top1 - 0.01,
        "top5_accuracy": min(1.0, top1 + 0.04),
    }
    return {
        "model": name,
        "num_parameters": 10,
        "best_epoch": 7,
        "best_val": metrics(val_top1),
        "test": metrics(test_top1),
        "signature": {"architecture": {"name": f"{name.title()}Model"}},
    }


class UnifiedReportTest(unittest.TestCase):
    def test_validation_free_sweep_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep_root = root / "linear-probe"
            sweep_root.mkdir()
            (sweep_root / "sweep-config.json").write_text(
                json.dumps({"checkpoints": [{"split_counts": {"val": 0}}]})
            )
            with self.assertRaisesRegex(ValueError, "validation-free"):
                report.run(root)

    def test_percentage_formatting(self):
        self.assertEqual(report._percent(0.948355), "94.84")

    def test_classifier_summary_ingestion_and_ordering(self):
        rows = report.load_classifier_rows(
            {
                "seed": 42,
                "summaries": {
                    "cnn": _summary("cnn", 0.90, 0.89),
                    "transformer": _summary("transformer", 0.91, 0.90),
                },
            }
        )
        self.assertEqual([row["name"] for row in rows], ["transformer", "cnn"])
        self.assertEqual(rows[0]["seed"], 42)
        self.assertEqual(rows[0]["test_top1_accuracy"], 0.90)

    def test_validation_selection_prefers_lower_lr_on_tie(self):
        rows = [
            _aggregate("run", 0.7, 0.8, 0.9),
            _aggregate("run", 0.3, 0.8, 0.7),
            _aggregate("other", 0.5, 0.6, 0.6),
        ]
        selected = report.select_probe_rows(rows, seed=42)
        by_name = {row["name"]: row for row in selected}
        self.assertEqual(by_name["run"]["selected_lr"], 0.3)
        self.assertEqual(by_name["run"]["test_top1_accuracy"], 0.7)

    def test_probes_are_ordered_by_encoder_then_predictor_size(self):
        run_names = (
            "mot_small-base_1d-bs.512-ep.300",
            "mot_huge-large_1d-bs.512-ep.300",
            "mot_huge-giant_1d-bs.512-ep.300",
            "mot_tiny-small_1d-bs.512-ep.300",
            "mot_huge_1d-bs.512-ep.300",
        )
        rows = [_aggregate(name, 0.3, 0.5, 0.5) for name in run_names]
        selected = report.select_probe_rows(rows)
        self.assertEqual(
            [row["name"] for row in selected],
            [
                "mot_huge-giant_1d-bs.512-ep.300",
                "mot_huge_1d-bs.512-ep.300",
                "mot_huge-large_1d-bs.512-ep.300",
                "mot_small-base_1d-bs.512-ep.300",
                "mot_tiny-small_1d-bs.512-ep.300",
            ],
        )

    def test_builds_single_seed_report_and_resolves_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep_root = root / "linear-probe"
            classifier_root = root / "classifiers"
            (sweep_root / "plots").mkdir(parents=True)
            classifier_root.mkdir()
            aggregates = [
                _aggregate("run-a", 0.1, 0.75, 0.74),
                _aggregate("run-a", 0.3, 0.80, 0.79),
            ]
            with (sweep_root / "aggregate-results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(aggregates[0]))
                writer.writeheader()
                writer.writerows(aggregates)
            config = {
                "seeds": [42],
                "lrs": [0.1, 0.3],
                "epochs": 50,
                "momentum": 0.9,
                "batch_size": 256,
                "checkpoints": [
                    {
                        "run_name": "run-a",
                        "model_name": "mot_tiny_1d",
                        "feature_dim": 192,
                        "checkpoint_sha256": "abc",
                        "split_counts": {"train": 8, "val": 1, "test": 1},
                    }
                ],
            }
            (sweep_root / "sweep-config.json").write_text(json.dumps(config))
            classifier_results = {
                "format_version": 1,
                "input_source": "raw",
                "seed": 42,
                "summaries": {
                    "cnn": _summary("cnn", 0.9, 0.89),
                    "transformer": _summary("transformer", 0.91, 0.90),
                },
            }
            (classifier_root / "results.json").write_text(
                json.dumps(classifier_results)
            )
            for path in (
                sweep_root / "validation-top1-heatmap.png",
                sweep_root / "test-top1-heatmap.png",
                sweep_root / "plots/run-a.png",
                sweep_root / "sweep-results.csv",
                sweep_root / "epoch-metrics.csv",
                classifier_root / "training-curves.png",
                classifier_root / "cnn-metrics.csv",
                classifier_root / "transformer-metrics.csv",
            ):
                path.touch()

            result = report.run(root)
            self.assertEqual(result["best_probe"], "run-a")
            self.assertTrue(
                (sweep_root / "lr-0p3-model-comparison.png").is_file()
            )
            text = (root / "README.md").read_text()
            self.assertNotIn("+/-", text)
            self.assertNotIn("standard deviation", text.lower())
            self.assertIn("Validation-selected LR: `0.3`", text)
            self.assertEqual(report.validate_local_links(root / "README.md"), [])
            with (root / "selected-results.csv").open() as file:
                selected = list(csv.DictReader(file))
            self.assertEqual(
                [row["name"] for row in selected],
                ["transformer", "cnn", "run-a"],
            )


if __name__ == "__main__":
    unittest.main()
