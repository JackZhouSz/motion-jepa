"""BABEL online probing and independent pretraining checkpoint selection."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from experiment.linear_probe.features import extract_features
from experiment.linear_probe.online import OnlineBabelProbes
from model import MotionPatchTransformer1D
from test_babel_classifier import _write_babel_dataset
from train import main as train_main
from utils.distributed import init_distributed


def _write_babel_subset(root: Path, subset: int) -> None:
    _write_babel_dataset(root)
    if subset == 60:
        return
    names = [f"action-{index}" for index in range(subset)]
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    metadata.update(
        source_dataset=f"BABEL-{subset}_fixture",
        subset=subset,
        num_classes=subset,
        class_names=names,
    )
    (root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "class-index.json").write_text(json.dumps({
        "class_names": names,
        "class_to_index": {name: index for index, name in enumerate(names)},
    }), encoding="utf-8")


def _probe_summary(score: float) -> dict:
    return {
        "best_epoch": 2,
        "best_val": {
            "loss": 0.5,
            "mean_average_precision": score,
            "top1_hit": 0.4,
            "top1_label_row_accuracy": 0.3,
            "top5_hit": 0.9,
            "classes_with_positives": 2,
            "classes_without_positives": 58,
        },
        "test": None,
        "selection": "validation_best",
        "validation_used": True,
    }


class _FakeBabelProbes:
    calls = 0
    scores = ((0.4, 0.5), (0.7, 0.4), (0.6, 0.8), (0.65, 0.7))

    def __init__(self, training_config, probe_config, *, device):
        del training_config, probe_config, device
        self.epochs = 2
        self.learning_rate = 0.3

    def evaluate(self, encoder):
        assert not encoder.training
        assert not any(parameter.requires_grad for parameter in encoder.parameters())
        scores = self.scores[type(self).calls]
        type(self).calls += 1
        return {
            "babel-60": _probe_summary(scores[0]),
            "babel-120": _probe_summary(scores[1]),
        }


class _Writer:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))

    def flush(self):
        pass

    def close(self):
        pass


class OnlineBabelProbeTest(unittest.TestCase):
    def test_real_babel_60_and_120_probes_use_multilabel_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain = root / "pretrain"
            stats = pretrain / "stats"
            stats.mkdir(parents=True)
            np.save(stats / "mean.npy", np.zeros(6, dtype=np.float32))
            np.save(stats / "std.npy", np.ones(6, dtype=np.float32))
            paths = {}
            for subset in (60, 120):
                dataset = root / f"babel-{subset}"
                _write_babel_subset(dataset, subset)
                paths[f"babel-{subset}"] = str(dataset)
            evaluator = OnlineBabelProbes(
                {
                    "data": {
                        "root_path": str(pretrain), "stats_path": "stats",
                        "num_frames": 4, "fps": 30, "motion_dim": 6,
                    },
                    "meta": {"use_bfloat16": False},
                },
                {
                    "datasets": paths, "epochs": 2, "feature_batch_size": 2,
                    "batch_size": 2, "num_workers": 0, "lr": 0.1, "seed": 3,
                },
                device=torch.device("cpu"),
            )
            encoder = MotionPatchTransformer1D(
                6, 4, temporal_patch_size=2, embed_dim=12, depth=1, num_heads=3
            ).eval().requires_grad_(False)
            summaries = evaluator.evaluate(encoder)
            for subset in (60, 120):
                summary = summaries[f"babel-{subset}"]
                self.assertEqual(summary["num_classes"], subset)
                self.assertEqual(summary["split_counts"], {"train": 4, "val": 2, "test": 0})
                self.assertIsNone(summary["test"])
                self.assertEqual(summary["best_val"]["classes_with_positives"], 2)
                self.assertEqual(
                    summary["best_val"]["classes_without_positives"], subset - 2
                )
                self.assertGreaterEqual(summary["best_val"]["mean_average_precision"], 0)
            self.assertFalse(any(root.rglob("features/*.pt")))
            self.assertFalse(any(parameter.grad is not None for parameter in encoder.parameters()))

            short = extract_features(
                encoder,
                [(torch.zeros(4, 6), 30, 1, torch.tensor([1.0, 0.0]), "short")],
                device=torch.device("cpu"), batch_size=1, num_workers=0,
                use_bfloat16=False, show_progress=False,
            )
            self.assertTrue(torch.isfinite(short["features"]).all())
            self.assertEqual(float(short["features"].abs().sum()), 0.0)
            self.assertEqual(short["labels"].dtype, torch.float32)

    def test_checkpoint_selection_logging_and_resume_are_per_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "output"
            write_npy_dataset(
                dataset,
                [np.random.default_rng(i).normal(size=(6, 6)).astype(np.float32)
                 for i in range(2)],
            )
            config = {
                "data": {
                    "batch_size": 2, "root_path": str(dataset), "meta_files": ["train.txt"],
                    "num_workers": 0, "pin_mem": False, "persistent_workers": False,
                    "drop_last": True, "num_frames": 6, "fps": 60, "motion_dim": 6,
                    "num_joints": 30, "normalize": False, "stats_path": None,
                },
                "patch": {"temporal_patch_size": 3},
                "logging": {
                    "folder": str(output), "write_tag": "babel-probe", "log_freq": 1,
                    "checkpoint_freq": 2, "tensorboard": True,
                },
                "mask": {
                    "allow_overlap": False, "num_enc_masks": 1, "num_pred_masks": 1,
                    "enc_frame_mask_ratio": [0.5, 0.5],
                    "pred_frame_mask_ratio": [0.5, 0.5],
                },
                "meta": {
                    "seed": 0, "load_checkpoint": False, "read_checkpoint": None,
                    "model_name": "mot_patch_tiny_1d",
                    "predictor_name": "mot_predictor_patch_tiny_1d",
                    "use_bfloat16": False, "use_float16": False,
                },
                "optimization": {
                    "ema": [0.9, 1.0], "epochs": 1, "final_lr": 1.0e-5,
                    "final_weight_decay": 0.4, "ipe_scale": 1.0, "lr": 1.0e-3,
                    "start_lr": 1.0e-4, "warmup": 0, "weight_decay": 0.04,
                },
                "linear_probe": {
                    "enabled": True, "frequency": 1,
                    "datasets": {"babel-60": "unused", "babel-120": "unused"},
                },
            }
            writer = _Writer()
            _FakeBabelProbes.calls = 0
            with patch(
                "experiment.linear_probe.online.OnlineBabelProbes", _FakeBabelProbes
            ), patch("train._make_tensorboard_writer", return_value=writer), patch(
                "train.init_distributed", wraps=init_distributed
            ) as distributed_init:
                train_main(config, device="cpu")
                resumed = copy.deepcopy(config)
                resumed["meta"]["load_checkpoint"] = True
                resumed["optimization"]["epochs"] = 3
                train_main(resumed, device="cpu")

            self.assertEqual(
                [call.kwargs["timeout_seconds"] for call in distributed_init.call_args_list],
                [4 * 60 * 60, 4 * 60 * 60],
            )
            self.assertEqual(_FakeBabelProbes.calls, 4)
            latest = torch.load(output / "babel-probe-latest.pth.tar", weights_only=False)
            best_60 = torch.load(output / "babel-probe-best-babel-60-map.pth.tar", weights_only=False)
            best_120 = torch.load(output / "babel-probe-best-babel-120-map.pth.tar", weights_only=False)
            self.assertEqual(latest["babel_probe_state"]["babel-60"]["latest"]["pretrain_epoch"], 3)
            self.assertEqual(latest["babel_probe_state"]["babel-120"]["latest"]["pretrain_epoch"], 3)
            self.assertEqual(latest["babel_probe_state"]["babel-60"]["best_epoch"], 1)
            self.assertEqual(latest["babel_probe_state"]["babel-120"]["best_epoch"], 2)
            self.assertEqual(best_60["next_epoch"], 1)
            self.assertEqual(best_120["next_epoch"], 2)
            self.assertFalse((output / "babel-probe-best-accuracy.pth.tar").exists())
            self.assertEqual(
                [step for name, _, step in writer.scalars
                 if name == "linear_probe/babel-60/val_mean_average_precision"],
                [0, 1, 2, 3],
            )
            self.assertEqual(
                [value for name, value, _ in writer.scalars
                 if name == "linear_probe/babel-120/best_val_mean_average_precision"],
                [0.5, 0.5, 0.8, 0.8],
            )


if __name__ == "__main__":
    unittest.main()
