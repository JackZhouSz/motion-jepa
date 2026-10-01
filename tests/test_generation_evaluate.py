"""Whole-draw metrics, EMA-only inference, provenance and portable sampling."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from experiment.generation.evaluate import evaluate_checkpoint, load_generator, sample_checkpoint
from experiment.generation.metrics import DiversityMetrics, per_draw_mse, select_best_draw
from experiment.generation.model import FlowConfig, MotionFlow
from experiment.generation.sampling import seed_for_sample
from experiment.generation.settings import load_config
from experiment.prediction.data import PredictionDataset, prepare_caches
from experiment.prediction.model import DecoderConfig, MotionDecoder
from model import MODEL_FACTORIES
from motion_rep import MotionJEPAMotionRep
from motion_rep.geometry import y_rotation


class GenerationMetricTest(unittest.TestCase):
    def test_oracle_selects_one_entire_draw_instead_of_framewise_best(self):
        target = torch.zeros(1, 2, 1)
        generated = torch.tensor([[[[0.], [4.]], [[4.], [0.]]]])
        active = torch.ones(1, 2, dtype=torch.bool)
        torch.testing.assert_close(per_draw_mse(generated, target, active), torch.tensor([[8., 8.]]))
        best, indices = select_best_draw(generated, target, active)
        self.assertEqual(indices.tolist(), [0])
        torch.testing.assert_close(best, generated[:, 0])
        self.assertEqual(float(best.square().mean()), 8.)
        generated[:, :, 1] = float("nan")
        active[:, 1] = False
        best, indices = select_best_draw(generated, target, active)
        self.assertEqual(indices.tolist(), [0])

    def test_diversity_separates_root_translation_from_root_relative_pose(self):
        representation = MotionJEPAMotionRep(fps=30)
        rotations = torch.eye(3).expand(2, 30, 3, 3).clone()
        root = torch.zeros(2, 3)
        root[:, 1] = .1
        first = representation.encode(rotations, root, canonicalize=False)
        second = representation.encode(rotations, root + torch.tensor([2., 0., 0.]), canonicalize=False)
        generated = torch.stack((first, second))[None]
        generated[:, :, 1] = float("nan")
        metric = DiversityMetrics(torch.zeros(366), torch.ones(366), 30)
        metric.update(generated, torch.tensor([[True, False]]))
        result = metric.compute()
        self.assertAlmostEqual(result["root_pairwise_mm"], 2000., places=3)
        self.assertLess(result["root_relative_nonroot_joint_pairwise_mm"], 1.0e-3)
        self.assertEqual(result["valid_pair_frames"], 1)
        one = DiversityMetrics(torch.zeros(366), torch.ones(366), 30)
        one.update(generated[:, :1], torch.tensor([[True, False]]))
        self.assertEqual(one.compute()["clip_pairs"], 0)
        self.assertEqual(one.compute()["root_pairwise_mm"], 0.)


class GenerationEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        data_root = self.root / "data"
        representation = MotionJEPAMotionRep(fps=30)
        motions = []
        for index in range(4):
            rotations = torch.eye(3).expand(6, 30, 3, 3).clone()
            rotations[:, 3] = y_rotation(torch.arange(6) * (.04 + index * .02))
            root = torch.zeros(6, 3)
            root[:, 1] = .1
            root[:, 0] = torch.arange(6) * (.01 + index * .005)
            motions.append(representation.encode(rotations, root).numpy())
        for split in ("train", "val", "test"):
            selected = motions if split != "test" else motions[:3] + [motions[3][:3]]
            write_npy_dataset(data_root, selected, split=split, num_frames=6, fps=30)
        stats = data_root / "stats"
        stats.mkdir()
        values = np.concatenate(motions)
        np.save(stats / "mean.npy", values.mean(0).astype(np.float32))
        np.save(stats / "std.npy", values.std(0).astype(np.float32))
        torch.manual_seed(1)
        encoder = MODEL_FACTORIES["mot_patch_tiny_1d"](in_chans=366, num_frames=6, temporal_patch_size=3)
        jepa = self.root / "jepa.pth.tar"
        source_config = {
            "data": {"root_path": str(data_root), "stats_path": "stats", "num_frames": 6,
                     "motion_dim": 366, "num_joints": 30, "fps": 30, "normalize": True},
            "meta": {"model_name": "mot_patch_tiny_1d", "use_bfloat16": False},
            "patch": {"temporal_patch_size": 3},
        }
        torch.save({"format_version": 1, "config": source_config,
                    "encoder": encoder.state_dict(), "target_encoder": encoder.state_dict()}, jepa)
        self.config = load_config(overrides={
            "jepa_checkpoint": str(jepa), "dataset_root": str(data_root),
            "cache_root": str(self.root / "cache"), "output": str(self.root / "run"),
            "device": "cpu", "epochs": 2, "batch_size": 2, "cache_batch_size": 2,
            "num_workers": 0, "warmup_epochs": 0, "use_bfloat16": False, "tensorboard": False,
            "limit_train": 2, "steps": 2, "num_samples": 2, "diagnostic_count": 3, "export_count": 2,
            "flow": {"hidden_dim": 24, "depth": 1, "num_heads": 2, "ffn_dim": 48, "dropout": 0.},
        })
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prepare_caches(self.config)
        train = PredictionDataset(self.config, "train")
        val = PredictionDataset(self.config, "val")
        flow_config = FlowConfig(**self.config["flow"])
        flow = MotionFlow(192, train.token_layout, 366, flow_config)
        raw_weights = copy.deepcopy(flow.state_dict())
        ema = copy.deepcopy(raw_weights)
        ema["feature_projection.bias"] += .25
        self.saved = {"format_version": 1, "kind": "motion_flow", "model": raw_weights, "ema": ema,
                      "flow_config": asdict(flow_config), "config": self.config,
                      "model_info": train.model_info, "token_layout": train.token_layout.signature(),
                      "mean": train.mean, "std": train.std, "feature_transform": "jepa_target_layer_norm",
                      "provenance": {"train": train.provenance, "val": val.provenance}}
        self.checkpoint = self.root / "flow.pth.tar"
        torch.save(self.saved, self.checkpoint)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        self.temporary.cleanup()

    def _quiet(self, operation, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return operation(*args, **kwargs)

    def test_ema_reload_metrics_draw_zero_reuse_and_exports(self):
        model, saved = load_generator(self.checkpoint, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved["ema"][name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.feature_projection.bias, saved["model"]["feature_projection.bias"]))
        result = self._quiet(evaluate_checkpoint, self.checkpoint)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["diagnostic_samples"], 3)
        self.assertLessEqual(result["best_of_k"]["mse"], result["monte_carlo"]["mse"])
        self.assertEqual(sum(result["best_draw_counts"]), 3)
        self.assertEqual(result["protocol"]["diagnostic_indices"], [0, 1, 3])
        self.assertEqual(result["protocol"]["weights"], "ema")
        self.assertEqual(result["protocol"]["precision"]["velocity_network"], "float32")
        self.assertEqual(result["protocol"]["total_generated_trajectories"], 7)
        self.assertEqual(result["protocol"]["network_evaluations_per_trajectory"], 2)
        self.assertEqual(result["protocol"]["total_per_example_network_evaluations"], 14)
        self.assertEqual(result["protocol"]["network_forward_batch_calls"], 8)
        with np.load(result["exports"], allow_pickle=False) as export:
            generated = export["generated_motion"].copy()
            self.assertEqual(generated.shape, (2, 2, 6, 366))
            self.assertEqual(export["generated_joints"].shape, (2, 2, 6, 30, 3))
            self.assertEqual(export["sample_ids"].tolist(), ["sample-0", "sample-3"])
            self.assertEqual(export["fps"].shape, ())
            self.assertEqual(export["noise_seeds"][1, 1], seed_for_sample(42, "sample-3", 1))
            self.assertEqual(json.loads(str(export["sampling_json"]))["steps"], 2)
            for key in ("target_motion", "generated_motion", "target_joints", "generated_joints"):
                self.assertTrue(np.isfinite(export[key]).all())
                self.assertTrue((export[key][1, ..., 3:, :, :] == 0).all() if key == "generated_joints"
                                else (export[key][1, :, 3:] == 0).all() if key == "generated_motion"
                                else (export[key][1, 3:] == 0).all())
        sampled = self._quiet(sample_checkpoint, self.checkpoint, indices=[0, 3],
                              overrides={"output": str(self.root / "samples"), "batch_size": 1})
        with np.load(sampled["exports"], allow_pickle=False) as export:
            np.testing.assert_allclose(export["generated_motion"], generated, rtol=1.0e-5, atol=1.0e-6)
        self.assertTrue((self.root / "run/test-metrics.json").is_file())
        self.assertTrue((self.root / "run/test-protocol.json").is_file())
        self.assertTrue((self.root / "samples/test-samples.json").is_file())

    def test_runtime_overrides_fresh_cache_and_sampling_controls(self):
        requested = self.root / "sampling.yaml"
        requested.write_text("steps: 3\nguidance_scale: 0.5\nnum_samples: 1\nexport_count: 1\n", encoding="utf-8")
        result = self._quiet(evaluate_checkpoint, self.checkpoint, config_path=requested,
                             overrides={"guidance_scale": 1.5, "cache_root": str(self.root / "fresh-cache"),
                                        "output": str(self.root / "overrides")})
        protocol = result["protocol"]
        self.assertEqual(protocol["steps"], 3)
        self.assertEqual(protocol["guidance_scale"], 1.5)
        self.assertEqual(protocol["network_evaluations_per_trajectory"], 6)
        self.assertEqual(protocol["total_generated_trajectories"], 4)
        self.assertEqual(result["monte_carlo"], result["best_of_k"])
        self.assertEqual(result["diversity"]["clip_pairs"], 0)
        self.assertFalse((self.root / "fresh-cache/train/completed.json").exists())
        self.assertTrue((self.root / "fresh-cache/test/completed.json").is_file())
        with np.load(result["exports"], allow_pickle=False) as exported:
            self.assertEqual(exported["generated_motion"].shape[:2], (1, 1))
        with self.assertRaisesRegex(ValueError, "indices"):
            self._quiet(sample_checkpoint, self.checkpoint, indices=[0, 0])

    def test_runtime_float32_keeps_bfloat16_encoder_cache_provenance(self):
        # CPU extraction stores BF16 tokens under the same on-CUDA AMP policy.
        config = {**self.config, "cache_root": str(self.root / "bf16-cache"), "use_bfloat16": True}
        self._quiet(prepare_caches, config)
        train, val = PredictionDataset(config, "train"), PredictionDataset(config, "val")
        saved = {**self.saved, "config": config,
                 "provenance": {"train": train.provenance, "val": val.provenance}}
        checkpoint = self.root / "bf16-flow.pth.tar"
        torch.save(saved, checkpoint)
        result = self._quiet(evaluate_checkpoint, checkpoint,
                             overrides={"use_bfloat16": False, "output": str(self.root / "float32")})
        self.assertFalse(result["protocol"]["use_bfloat16"])
        self.assertTrue(result["protocol"]["encoder_extraction_precision"]["use_bfloat16"])
        self.assertTrue(result["provenance"]["use_bfloat16"])

    def test_explicit_baseline_allows_a_larger_training_subset(self):
        baseline_config = {**self.config, "limit_train": 0, "cache_root": str(self.root / "baseline-cache")}
        self._quiet(prepare_caches, baseline_config, splits=("train",))
        dataset = PredictionDataset(baseline_config, "train")
        architecture = DecoderConfig(hidden_dim=24, depth=1, num_heads=2, ffn_dim=48, dropout=0.)
        decoder = MotionDecoder(192, dataset.token_layout, 366, architecture)
        baseline = {"format_version": 1, "kind": "motion_decoder", "decoder": decoder.state_dict(),
                    "decoder_config": asdict(architecture), "model_info": dataset.model_info,
                    "token_layout": dataset.token_layout.signature(), "mean": dataset.mean, "std": dataset.std,
                    "feature_transform": "jepa_target_layer_norm", "provenance": {"train": dataset.provenance}}
        checkpoint = self.root / "decoder.pth.tar"
        torch.save(baseline, checkpoint)
        result = self._quiet(evaluate_checkpoint, self.checkpoint,
                             overrides={"deterministic_checkpoint": str(checkpoint)})
        self.assertIn("deterministic_baseline", result)
        baseline["mean"] = baseline["mean"] + 1.
        torch.save(baseline, checkpoint)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self._quiet(evaluate_checkpoint, self.checkpoint,
                        overrides={"deterministic_checkpoint": str(checkpoint)})

    def test_wrong_kind_stats_and_architecture_rejected(self):
        wrong = self.root / "wrong.pth.tar"
        torch.save({**self.saved, "kind": "motion_decoder"}, wrong)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_generator(wrong, torch.device("cpu"))
        torch.save({**self.saved, "mean": self.saved["mean"] + 1.}, wrong)
        with self.assertRaisesRegex(ValueError, "statistics"):
            self._quiet(evaluate_checkpoint, wrong)
        with self.assertRaisesRegex(ValueError, "architecture"):
            self._quiet(evaluate_checkpoint, self.checkpoint, overrides={"flow": {"hidden_dim": 48}})


if __name__ == "__main__":
    unittest.main()
