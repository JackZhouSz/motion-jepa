"""Real-cache flow training, EMA selection, and exact stochastic resume."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from _npy_fixture import write_npy_dataset
from experiment.generation.__main__ import build_parser
from experiment.generation.settings import load_config, load_checkpoint_config
from experiment.generation.train import (
    interpolate_motion, make_generator, train, validation_flow_mse,
)
from experiment.prediction.data import PredictionDataset, prepare_caches
from model import MODEL_FACTORIES
from motion_rep import MotionJEPAMotionRep
from motion_rep.geometry import y_rotation


class GenerationTrainingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        data_root = self.root / "data"
        motions = []
        rep = MotionJEPAMotionRep(fps=30)
        for sample in range(4):
            rotations = torch.eye(3).expand(6, 30, 3, 3).clone()
            rotations[:, 3] = y_rotation(torch.arange(6) * (.04 + sample * .02))
            positions = torch.zeros(6, 3)
            positions[:, 1] = .1
            positions[:, 0] = torch.arange(6) * (.01 + sample * .005)
            motions.append(rep.encode(rotations, positions).numpy())
        for split in ("train", "val", "test"):
            write_npy_dataset(data_root, motions, split=split, num_frames=6, fps=30)
        stats = data_root / "stats"
        stats.mkdir()
        values = np.concatenate(motions)
        np.save(stats / "mean.npy", values.mean(0).astype(np.float32))
        np.save(stats / "std.npy", values.std(0).astype(np.float32))
        torch.manual_seed(1)
        encoder = MODEL_FACTORIES["mot_patch_tiny_1d"](in_chans=366, num_frames=6, temporal_patch_size=3)
        self.source = self.root / "jepa.pth.tar"
        source_config = {
            "data": {"root_path": str(data_root), "stats_path": "stats", "num_frames": 6,
                     "motion_dim": 366, "num_joints": 30, "fps": 30, "normalize": True},
            "meta": {"model_name": "mot_patch_tiny_1d", "use_bfloat16": False},
            "patch": {"temporal_patch_size": 3},
        }
        torch.save({"format_version": 1, "config": source_config,
                    "encoder": encoder.state_dict(), "target_encoder": encoder.state_dict()}, self.source)
        self.config = load_config(overrides={
            "jepa_checkpoint": str(self.source), "dataset_root": str(data_root),
            "cache_root": str(self.root / "cache"), "output": str(self.root / "run"),
            "device": "cpu", "epochs": 3, "batch_size": 2, "cache_batch_size": 2,
            "num_workers": 0, "warmup_epochs": 0, "use_bfloat16": False,
            "tensorboard": False, "export_count": 2, "diagnostic_count": 3,
            "num_samples": 2, "steps": 2,
            "flow": {"hidden_dim": 24, "depth": 1, "num_heads": 2, "ffn_dim": 48, "dropout": .15},
        })
        prepare_caches(self.config)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        self.directory.cleanup()

    def test_raw_ema_checkpoint_reload_and_exact_resume(self):
        from experiment.generation.evaluate import load_generator
        from experiment.generation.visualize import load_results
        before_source = self.source.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            full = train(self.config)
            other = dict(self.config, output=str(self.root / "resumed"))
            partial = train(other, stop_after_epoch=1)
            resumed = train(other, resume=self.root / "resumed/latest.pth.tar")
        self.assertFalse(partial["complete"])
        self.assertTrue(full["complete"] and resumed["complete"])
        final = torch.load(self.root / "run/latest.pth.tar", weights_only=False)
        restored = torch.load(self.root / "resumed/latest.pth.tar", weights_only=False)
        for weights in ("model", "ema"):
            for name, value in final[weights].items():
                torch.testing.assert_close(value, restored[weights][name], rtol=0, atol=0)
        self.assertEqual(final["ema_updates"], final["global_step"])
        for left, right in zip(full["history"], resumed["history"]):
            for key in ("epoch", "global_step", "train_flow_mse", "val_flow_mse", "lr", "ema_decay"):
                self.assertEqual(left[key], right[key])
        self.assertEqual(before_source, self.source.read_bytes())
        self.assertNotIn("target_encoder", final)
        self.assertNotIn("encoder", final)
        model, saved = load_generator(self.root / "run/best.pth.tar", torch.device("cpu"))
        model_again, _ = load_generator(self.root / "run/best.pth.tar", torch.device("cpu"))
        dataset = PredictionDataset(self.config, "test")
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        noise = torch.randn(2, 1, 6, 366)
        sampled = model.sample(batch["tokens"], batch["fps"], batch["valid_frames"],
                               steps=2, initial_noise=noise, use_bfloat16=False)
        sampled_again = model_again.sample(batch["tokens"], batch["fps"], batch["valid_frames"],
                                           steps=2, initial_noise=noise, use_bfloat16=False)
        torch.testing.assert_close(sampled, sampled_again, rtol=0, atol=0)
        results = load_results(self.root / "run/test-generations.npz")
        self.assertEqual(results.generated_motion.shape, (2, 2, 6, 366))
        self.assertTrue((self.root / "run/test-metrics.json").is_file())
        self.assertTrue((self.root / "run/metrics.csv").is_file())
        self.assertTrue(np.isfinite(list(full["test"]["single_sample"].values())).all())
        with self.assertRaisesRegex(FileExistsError, "destination"):
            train(self.config, resume=self.root / "resumed/latest.pth.tar")
        with self.assertRaisesRegex(ValueError, "condition_dropout"):
            train(dict(other, condition_dropout=.2), resume=self.root / "resumed/latest.pth.tar")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train(dict(other, output=str(self.root / "relocated")), resume=self.root / "resumed/latest.pth.tar")
        self.assertTrue((self.root / "relocated/test-metrics.json").is_file())

    def test_validation_rng_independence_and_padding_interpolation(self):
        dataset = PredictionDataset(self.config, "val")
        model = make_generator(dataset, self.config).eval()
        generator = torch.Generator().manual_seed(99)
        loader = DataLoader(dataset, batch_size=2, generator=generator)
        before = torch.get_rng_state().clone()
        first = validation_flow_mse(model, loader, torch.device("cpu"), False, 42)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        second = validation_flow_mse(model, loader, torch.device("cpu"), False, 42)
        self.assertEqual(first, second)
        motion = torch.ones(2, 5, 3)
        noise = torch.full_like(motion, 2.)
        frames = torch.tensor([[True, True, True, True, False], [True] * 5])
        motion[0, -1] = float("nan")
        noise[0, -1] = float("nan")
        state, velocity = interpolate_motion(motion, noise, torch.tensor([0., 1.]), frames)
        torch.testing.assert_close(state[0, :4], torch.full((4, 3), 2.))
        torch.testing.assert_close(state[1], torch.ones(5, 3))
        self.assertTrue(torch.equal(state[0, -1], torch.zeros(3)))
        self.assertTrue(torch.equal(velocity[0, -1], torch.zeros(3)))
        torch.testing.assert_close(velocity[0, 3], -torch.ones(3))


class GenerationSettingsTest(unittest.TestCase):
    def test_saved_config_yaml_cli_precedence_and_architecture_merge(self):
        saved = load_config(overrides={"steps": 8, "flow": {"hidden_dim": 24, "num_heads": 2}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.yaml"
            path.write_text("steps: 12\nflow:\n  dropout: 0.2\n", encoding="utf-8")
            config = load_checkpoint_config(saved, path, {"steps": 16})
        self.assertEqual(config["steps"], 16)
        self.assertEqual(config["flow"]["hidden_dim"], 24)
        self.assertEqual(config["flow"]["dropout"], .2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.yaml"
            path.write_text("lr: 3e-4\nfinal_lr: 1e-6\n", encoding="utf-8")
            scientific = load_config(path)
        self.assertEqual(scientific["lr"], .0003)
        self.assertEqual(scientific["final_lr"], .000001)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            load_config(overrides={"num_samples": 0})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_config(overrides={"decoder": {}})
        parsed = build_parser().parse_args(["sample", "--checkpoint", "flow.pth.tar", "--indices", "2", "4",
                                           "--steps", "8", "--guidance-scale", "2", "--no-bfloat16"])
        self.assertEqual(parsed.indices, [2, 4])
        self.assertFalse(parsed.use_bfloat16)


if __name__ == "__main__":
    unittest.main()
