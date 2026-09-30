"""Integration coverage for independent segmentation and motion evaluations."""
from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from test_patch_training_smoke import PatchTrainingSmokeTest
from test_online_probe_training import _Writer
from train import (
    _evaluate_frozen_encoder_preserving_rng,
    _run_rank_zero,
    _write_tensorboard_babel_probes,
    _write_tensorboard_online_metrics,
    main as train_main,
)


class _Segmentation:
    calls = []

    def __init__(self, config, options, *, device):
        self.protocol_hash = options.get("test_protocol", "seg-v1")
        random.random()
        torch.rand(1)

    def evaluate(self, encoder, *, pretrain_epoch):
        assert not encoder.training
        assert not any(p.requires_grad for p in encoder.parameters())
        self.calls.append(pretrain_epoch)
        random.random()
        np.random.rand()
        torch.rand(7)
        score = {0: .2, 2: .8, 3: .7}[pretrain_epoch]
        metrics = {"frame_map": score, "bce": .4, "micro_f1": .3}
        return {"best_epoch": 2, "best_val": {"babel-120": metrics, "babel-60": metrics},
                "timings": {"feature_extraction_seconds": .01, "head_training_seconds": .02,
                            "total_seconds": .03}, "protocol_hash": self.protocol_hash}


class _Motion:
    calls = 0

    def __init__(self, config, options, *, device):
        self.protocol_hash = options.get("test_protocol", "motion-v1")
        random.random()
        torch.rand(1)

    def evaluate(self, encoder):
        assert not encoder.training
        assert not any(p.requires_grad for p in encoder.parameters())
        type(self).calls += 1
        random.random()
        np.random.rand()
        torch.rand(7)
        return {"retrieval": {"recall_at_1": .2}, "elapsed_seconds": .01,
                "protocol_hash": self.protocol_hash}


class SegmentationTrainingTest(unittest.TestCase):
    def _run(self, config, writer):
        with patch("experiment.segmentation_probe.OnlineSegmentationProbe", _Segmentation), \
             patch("experiment.motion_online_metrics.MotionOnlineMetrics", _Motion), \
             patch("experiment.motion_online_metrics.tensorboard_metrics",
                   return_value={"retrieval/recall_at_1": .2}), \
             patch("train._make_tensorboard_writer", return_value=writer):
            return train_main(config, device="cpu")

    def test_tensorboard_resume_retains_committed_probe_and_purges_future_events(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            self.skipTest("TensorBoard is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "data", root / "output"
            write_npy_dataset(dataset, [np.random.default_rng(i).normal(size=(6, 6)).astype(np.float32)
                                        for i in range(2)])
            config = PatchTrainingSmokeTest()._config(dataset, output)
            config["optimization"]["epochs"] = 2
            config["logging"].update(tensorboard=True, log_freq=1)
            config["linear_probe"] = {"enabled": False}
            config["attentive_probe"] = {"enabled": False}
            config["online_metrics"] = {"enabled": False}
            config["segmentation_probe"] = {"enabled": True, "frequency": 2}
            _Segmentation.calls = []
            with patch("experiment.segmentation_probe.OnlineSegmentationProbe", _Segmentation):
                result = train_main(config, device="cpu")
            checkpoint = torch.load(result["checkpoint"], weights_only=False)
            committed_step = checkpoint["global_step"]
            tag = "segmentation_probe/babel-120/val_frame_map"
            log_dir = output / "tensorboard"
            # Simulate events emitted after the most recent successful save.
            stale_writer = SummaryWriter(log_dir=str(log_dir))
            stale_writer.add_scalar(tag, 0.99, committed_step + 1)
            stale_writer.add_scalar("train/loss", 123.0, committed_step + 1)
            stale_writer.close()
            before = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
            self.assertEqual([event.step for event in before.Scalars(tag)], [0, committed_step, committed_step + 1])

            config["meta"].update(load_checkpoint=True, read_checkpoint=None)
            _Segmentation.calls = []
            with patch("experiment.segmentation_probe.OnlineSegmentationProbe", _Segmentation):
                train_main(config, device="cpu")
            self.assertEqual(_Segmentation.calls, [])  # Already-completed probe is not repeated.
            after = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
            probe_events = after.Scalars(tag)
            self.assertEqual([event.step for event in probe_events], [0, committed_step])
            self.assertAlmostEqual(probe_events[-1].value, 0.8)
            self.assertTrue(all(event.step <= committed_step for event in after.Scalars("train/loss")))

    def test_failed_epoch_zero_probe_has_resumable_initial_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "data", root / "interrupted"
            write_npy_dataset(dataset, [np.random.default_rng(i).normal(size=(6, 6)).astype(np.float32)
                                        for i in range(2)])
            config = PatchTrainingSmokeTest()._config(dataset, output)
            config["optimization"]["epochs"] = 3
            config["linear_probe"] = {"enabled": False}
            config["attentive_probe"] = {"enabled": False}
            config["online_metrics"] = {"enabled": False}
            config["segmentation_probe"] = {"enabled": True, "frequency": 2}

            def fail_initial(evaluator, encoder, *, pretrain_epoch):
                self.assertEqual(pretrain_epoch, 0)
                self.assertFalse(encoder.training)
                torch.rand(13)
                np.random.rand()
                random.random()
                raise RuntimeError("interrupted initial head")

            with patch.object(_Segmentation, "evaluate", new=fail_initial):
                with self.assertRaisesRegex(RuntimeError, "interrupted initial head"):
                    self._run(config, _Writer())
            latest = output / "patch-smoke-latest.pth.tar"
            self.assertTrue(latest.is_file())
            initial = torch.load(latest, weights_only=False)
            self.assertEqual(initial["next_epoch"], 0)
            self.assertEqual(initial["global_step"], 0)
            self.assertIsNone(initial["segmentation_probe_state"]["latest"])
            self.assertEqual(initial["segmentation_probe_state"]["protocol_hash"], "seg-v1")

            resumed = copy.deepcopy(config)
            resumed["meta"].update(load_checkpoint=True, read_checkpoint=None)
            _Segmentation.calls = []
            recovered = torch.load(self._run(resumed, _Writer())["checkpoint"], weights_only=False)
            self.assertEqual(_Segmentation.calls, [0, 2, 3])

            reference = copy.deepcopy(config)
            reference["logging"]["folder"] = str(root / "uninterrupted")
            control = torch.load(self._run(reference, _Writer())["checkpoint"], weights_only=False)
            control_initial = torch.load(root / "uninterrupted/initial-checkpoint.pth.tar", weights_only=False)
            self.assertEqual(recovered["global_step"], control["global_step"])
            self.assertEqual(recovered["segmentation_probe_state"], control["segmentation_probe_state"])
            for part in ("encoder", "predictor", "target_encoder"):
                for name, value in recovered[part].items():
                    torch.testing.assert_close(value, control[part][name], rtol=0, atol=0)
                    torch.testing.assert_close(initial[part][name], control_initial[part][name], rtol=0, atol=0)

    def test_schedule_resume_best_and_pretraining_rng_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "data", root / "with-eval"
            write_npy_dataset(dataset, [np.random.default_rng(i).normal(size=(6, 6)).astype(np.float32)
                                        for i in range(2)])
            config = PatchTrainingSmokeTest()._config(dataset, output)
            config["optimization"]["epochs"] = 3
            config["logging"].update(checkpoint_freq=2, tensorboard=True)
            config["linear_probe"] = {"enabled": False}
            config["attentive_probe"] = {"enabled": False}
            config["segmentation_probe"] = {"enabled": True, "frequency": 2}
            config["online_metrics"] = {"enabled": True, "kind": "motion", "frequency": 1}
            _Segmentation.calls, _Motion.calls = [], 0
            writer = _Writer()
            result = self._run(config, writer)
            self.assertEqual(_Segmentation.calls, [0, 2, 3])
            self.assertEqual(_Motion.calls, 4)
            trained = torch.load(result["checkpoint"], weights_only=False)
            state = trained["segmentation_probe_state"]
            self.assertEqual(state["best_epoch"], 2)
            self.assertEqual(state["latest"]["pretrain_epoch"], 3)
            best = torch.load(output / "patch-smoke-best-segmentation-map.pth.tar", weights_only=False)
            self.assertEqual(best["next_epoch"], 2)
            self.assertTrue((output / "initial-checkpoint.pth.tar").is_file())
            self.assertTrue((output / "patch-smoke-ep3.pth.tar").is_file())
            tags = {x[0] for x in writer.scalars}
            self.assertIn("segmentation_probe/babel-120/val_frame_map", tags)
            self.assertFalse(any(tag.startswith(("linear_probe/", "attentive_probe/")) for tag in tags))

            baseline = copy.deepcopy(config)
            baseline["logging"]["folder"] = str(root / "without-eval")
            baseline["segmentation_probe"]["enabled"] = False
            baseline["online_metrics"]["enabled"] = False
            control = torch.load(self._run(baseline, _Writer())["checkpoint"], weights_only=False)
            for part in ("encoder", "predictor", "target_encoder"):
                for key, tensor in trained[part].items():
                    torch.testing.assert_close(tensor, control[part][key], rtol=0, atol=0)

            resumed = copy.deepcopy(config)
            resumed["meta"].update(load_checkpoint=True, read_checkpoint="patch-smoke-ep2.pth.tar")
            _Segmentation.calls, _Motion.calls = [], 0
            resumed_checkpoint = torch.load(self._run(resumed, _Writer())["checkpoint"], weights_only=False)
            self.assertEqual(_Segmentation.calls, [3])
            self.assertEqual(_Motion.calls, 1)
            self.assertEqual(resumed_checkpoint["segmentation_probe_state"]["best_epoch"], 2)
            for part in ("encoder", "predictor", "target_encoder"):
                for key, tensor in trained[part].items():
                    torch.testing.assert_close(tensor, resumed_checkpoint[part][key], rtol=0, atol=0)
            rows = [json.loads(line) for line in (output / "segmentation-probe.jsonl").read_text().splitlines()]
            self.assertEqual([row["pretrain_epoch"] for row in rows], [0, 2, 3])

            # A compatible completed resume does not repeat any evaluation.
            resumed["meta"]["read_checkpoint"] = None
            _Segmentation.calls, _Motion.calls = [], 0
            self._run(resumed, _Writer())
            self.assertEqual((_Segmentation.calls, _Motion.calls), ([], 0))
            # Changing only segmentation protocol resets only that evaluator.
            resumed["segmentation_probe"]["test_protocol"] = "seg-v2"
            self._run(resumed, _Writer())
            self.assertEqual((_Segmentation.calls, _Motion.calls), ([3], 0))

    def test_evaluation_failure_restores_modes_flags_rng_and_propagates(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout()).train()
        model[1].eval()
        before = torch.get_rng_state().clone()

        class Failure:
            def evaluate(self, encoder):
                self.assertion = not encoder.training
                torch.rand(8)
                raise ValueError("fixture failure")

        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            _run_rank_zero(lambda: _evaluate_frozen_encoder_preserving_rng(Failure(), model), is_main=True)
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_logging_filters_metadata_but_keeps_flat_performance(self):
        writer = _Writer()
        for step in (0, 1):
            _write_tensorboard_babel_probes(writer, global_step=step,
                summaries={"babel-60": {"standardization": "train_channel_zscore", "best_epoch": 1,
                    "best_val": {"mean_average_precision": .2, "loss": .3,
                                 "top1_label_row_accuracy": .15,
                                 "classes_without_positives": 0, "classes_with_positives": 60}}},
                state={"babel-60": {"best_val_map": .2, "best_epoch": 0}})
        _write_tensorboard_online_metrics(writer, global_step=1,
            summary={"num_samples": 640, "seed": 42, "global_step": 1,
                     "representation": {"body": {"rankme": 50.}}})
        tags = [x[0] for x in writer.scalars]
        self.assertEqual(tags.count("linear_probe/babel-60/best_val_mean_average_precision"), 2)
        self.assertEqual(tags.count("linear_probe/babel-60/val_top1_label_row_accuracy"), 2)
        self.assertFalse(any("classes_" in tag or "feature_standardization" in tag for tag in tags))
        self.assertNotIn("online_metrics/num_samples", tags)
        self.assertIn("online_metrics/representation/body/rankme", tags)


if __name__ == "__main__":
    unittest.main()
