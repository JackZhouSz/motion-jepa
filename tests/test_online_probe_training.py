"""Online linear-probe scheduling and best-checkpoint integration tests."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from experiment.linear_probe.online import OnlineLinearProbe
from model import MotionTransformer1D
from train import _evaluate_online_probe_preserving_rng, main as train_main
from test_linear_probe import _write_dataset


def _summary(val_top1: float, test_top1: float, probe_epoch: int) -> dict:
    metrics = lambda top1: {
        "loss": 1.0,
        "top1_accuracy": top1,
        "macro_accuracy": top1,
        "top5_accuracy": top1,
    }
    return {
        "best_epoch": probe_epoch,
        "best_val": metrics(val_top1),
        "test": metrics(test_top1),
        "feature_dim": 4,
        "num_classes": 2,
        "split_counts": {"train": 2, "val": 2, "test": 2},
    }


def _fixed_summary(test_top1: float, probe_epoch: int) -> dict:
    result = _summary(0.0, test_top1, probe_epoch)
    result.update(
        selection="fixed_last_epoch",
        validation_used=False,
        best_val=None,
        split_counts={"train": 2, "val": 0, "test": 2},
        test_content="FW",
    )
    return result


class _Writer:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))

    def flush(self):
        pass

    def close(self):
        pass


class _FakeOnlineProbe:
    summaries = [
        _summary(0.6, 0.65, 1),
        _summary(0.8, 0.7, 2),
        _summary(0.75, 0.95, 3),
        _summary(0.7, 0.9, 4),
    ]
    calls = 0

    def __init__(self, training_config, probe_config, *, device):
        del training_config, probe_config, device
        self.dataset_root = Path("fake-100style")
        self.epochs = 2
        self.learning_rate = 0.3

    def evaluate(self, encoder):
        if encoder.training or any(
            parameter.requires_grad for parameter in encoder.parameters()
        ):
            raise AssertionError("Online probe received an unfrozen encoder")
        result = copy.deepcopy(self.summaries[type(self).calls])
        type(self).calls += 1
        return result


class _FakeFixedOnlineProbe(_FakeOnlineProbe):
    summaries = [_fixed_summary(0.55, 2), _fixed_summary(0.75, 2)]
    calls = 0


class _FakeOnlineMetrics:
    calls = 0

    def __init__(self, training_config, metric_config, *, device, collator):
        del training_config, metric_config, device, collator
        self.indices = [0, 1]

    def evaluate(self, encoder, predictor):
        if encoder.training or predictor.training:
            raise AssertionError("Online metrics received training-mode models")
        type(self).calls += 1
        return {
            "num_samples": 2,
            "representation": {"body": {"rankme": 2.0, "mean_std": 0.2}},
            "heldout_jepa": {
                "prediction_gain": 0.3,
                "trajectory_ablation": {"trajectory_reliance": 0.1},
            },
        }


class OnlineProbeTrainingTest(unittest.TestCase):
    def test_validation_free_online_probe_never_creates_best_accuracy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "output"
            write_npy_dataset(
                dataset,
                [np.zeros((4, 6), dtype=np.float32), np.ones((4, 6), dtype=np.float32)],
            )
            config = {
                "data": {
                    "batch_size": 2, "root_path": str(dataset), "meta_files": ["train.txt"],
                    "num_workers": 0, "pin_mem": False, "persistent_workers": False,
                    "drop_last": True, "num_frames": 4, "fps": 60, "motion_dim": 6,
                    "num_joints": 30, "normalize": False, "stats_path": None,
                },
                "logging": {
                    "folder": str(output), "write_tag": "fixed-probe", "log_freq": 1,
                    "checkpoint_freq": 1, "tensorboard": True,
                },
                "mask": {
                    "allow_overlap": False, "num_enc_masks": 1, "num_pred_masks": 1,
                    "enc_frame_mask_ratio": [0.75, 0.75],
                    "pred_frame_mask_ratio": [0.25, 0.25],
                },
                "meta": {
                    "seed": 0, "load_checkpoint": False, "read_checkpoint": None,
                    "model_name": "mot_tiny_1d", "predictor_name": "mot_predictor_tiny_1d",
                    "use_bfloat16": False, "use_float16": False,
                },
                "optimization": {
                    "ema": [0.9, 1.0], "epochs": 1, "final_lr": 1.0e-5,
                    "final_weight_decay": 0.4, "ipe_scale": 1.0, "lr": 1.0e-3,
                    "start_lr": 1.0e-4, "warmup": 0, "weight_decay": 0.04,
                },
                "linear_probe": {"enabled": True, "frequency": 1},
                "online_metrics": {"enabled": True},
            }
            writer = _Writer()
            _FakeFixedOnlineProbe.calls = 0
            _FakeOnlineMetrics.calls = 0
            with patch(
                "experiment.linear_probe.online.OnlineLinearProbe", _FakeFixedOnlineProbe
            ), patch(
                "experiment.online_metrics.OnlineRepresentationMetrics",
                _FakeOnlineMetrics,
            ), patch("train._make_tensorboard_writer", return_value=writer):
                train_main(config, device="cpu")
            self.assertEqual(_FakeFixedOnlineProbe.calls, 2)
            self.assertEqual(_FakeOnlineMetrics.calls, 2)
            self.assertFalse((output / "fixed-probe-best-accuracy.pth.tar").exists())
            latest = torch.load(
                output / "fixed-probe-latest.pth.tar", map_location="cpu", weights_only=False
            )
            self.assertEqual(latest["linear_probe_latest"]["selection"], "fixed_last_epoch")
            self.assertIsNone(latest["best_probe_epoch"])
            self.assertEqual(latest["online_metrics_latest"]["pretrain_epoch"], 1)
            self.assertEqual(
                len((output / "online-metrics.jsonl").read_text().splitlines()), 2
            )
            self.assertEqual(
                [event for event in writer.scalars if event[0] == "linear_probe/test_top1_accuracy"],
                [("linear_probe/test_top1_accuracy", 0.55, 0),
                 ("linear_probe/test_top1_accuracy", 0.75, 1)],
            )

    def test_real_in_memory_probe_evaluates_without_writing_feature_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "style-dataset"
            pretrain = root / "pretrain"
            stats = pretrain / "stats"
            _write_dataset(dataset)
            stats.mkdir(parents=True)
            np.save(stats / "mean.npy", np.zeros(6, dtype=np.float32))
            np.save(stats / "std.npy", np.ones(6, dtype=np.float32))
            evaluator = OnlineLinearProbe(
                {
                    "data": {
                        "root_path": str(pretrain),
                        "stats_path": "stats",
                        "num_frames": 4,
                        "fps": 30,
                        "motion_dim": 6,
                    },
                    "meta": {"use_bfloat16": False},
                },
                {
                    "dataset_root": str(dataset),
                    "epochs": 2,
                    "feature_batch_size": 4,
                    "batch_size": 2,
                    "num_workers": 0,
                    "lr": 0.1,
                    "seed": 3,
                },
                device=torch.device("cpu"),
            )
            encoder = MotionTransformer1D(
                6, 4, embed_dim=12, depth=1, num_heads=3
            ).eval().requires_grad_(False)
            summary = evaluator.evaluate(encoder)
            self.assertEqual(summary["num_classes"], 2)
            self.assertEqual(
                summary["split_counts"], {"train": 4, "val": 2, "test": 2}
            )
            self.assertFalse(any(dataset.rglob("features/*.pt")))

    def test_snapshot_and_final_probe_use_validation_for_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            write_npy_dataset(
                dataset,
                [np.zeros((4, 6), dtype=np.float32), np.ones((4, 6), dtype=np.float32)],
            )
            config = {
                "data": {
                    "batch_size": 2,
                    "root_path": str(dataset),
                    "meta_files": ["train.txt"],
                    "num_workers": 0,
                    "pin_mem": False,
                    "persistent_workers": False,
                    "drop_last": True,
                    "num_frames": 4,
                    "fps": 60,
                    "motion_dim": 6,
                    "num_joints": 30,
                    "normalize": False,
                    "stats_path": None,
                },
                "logging": {
                    "folder": str(output),
                    "write_tag": "online-probe",
                    "log_freq": 1,
                    "checkpoint_freq": 2,
                    "tensorboard": True,
                },
                "mask": {
                    "allow_overlap": False,
                    "num_enc_masks": 1,
                    "num_pred_masks": 1,
                    "enc_frame_mask_ratio": [0.75, 0.75],
                    "pred_frame_mask_ratio": [0.25, 0.25],
                },
                "meta": {
                    "seed": 0,
                    "load_checkpoint": False,
                    "read_checkpoint": None,
                    "model_name": "mot_tiny_1d",
                    "predictor_name": "mot_predictor_tiny_1d",
                    "use_bfloat16": False,
                    "use_float16": False,
                },
                "optimization": {
                    "ema": [0.9, 1.0],
                    "epochs": 3,
                    "final_lr": 1.0e-5,
                    "final_weight_decay": 0.4,
                    "ipe_scale": 1.0,
                    "lr": 1.0e-3,
                    "start_lr": 1.0e-4,
                    "warmup": 0,
                    "weight_decay": 0.04,
                },
                "linear_probe": {"enabled": True, "frequency": 1},
            }
            writer = _Writer()
            _FakeOnlineProbe.calls = 0
            with patch(
                "experiment.linear_probe.online.OnlineLinearProbe", _FakeOnlineProbe
            ), patch("train._make_tensorboard_writer", return_value=writer):
                train_main(config, device="cpu")

            self.assertEqual(_FakeOnlineProbe.calls, 4)
            latest = torch.load(
                output / "online-probe-latest.pth.tar",
                map_location="cpu",
                weights_only=False,
            )
            best = torch.load(
                output / "online-probe-best-accuracy.pth.tar",
                map_location="cpu",
                weights_only=False,
            )
            snapshot = torch.load(
                output / "online-probe-ep2.pth.tar",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(latest["linear_probe_latest"]["pretrain_epoch"], 3)
            self.assertEqual(latest["best_probe_epoch"], 1)
            self.assertEqual(latest["best_probe_val_top1"], 0.8)
            self.assertEqual(best["linear_probe_latest"]["pretrain_epoch"], 1)
            self.assertEqual(snapshot["linear_probe_latest"]["pretrain_epoch"], 2)
            test_events = [
                event
                for event in writer.scalars
                if event[0] == "linear_probe/test_top1_accuracy"
            ]
            self.assertEqual(
                test_events,
                [
                    ("linear_probe/test_top1_accuracy", 0.65, 0),
                    ("linear_probe/test_top1_accuracy", 0.7, 1),
                    ("linear_probe/test_top1_accuracy", 0.95, 2),
                    ("linear_probe/test_top1_accuracy", 0.9, 3),
                ],
            )

    def test_probe_evaluation_restores_all_rng_streams(self):
        class MutatingEvaluator:
            def evaluate(self, encoder):
                del encoder
                random.random()
                np.random.rand()
                torch.rand(())
                return {"ok": True}

        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        expected = (random.random(), np.random.rand(), torch.rand(()))
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        self.assertEqual(
            _evaluate_online_probe_preserving_rng(MutatingEvaluator(), object()),
            {"ok": True},
        )
        actual = (random.random(), np.random.rand(), torch.rand(()))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])


if __name__ == "__main__":
    unittest.main()
