"""Cache-to-decoder training, exact epoch resume, and portable reconstruction exports."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from experiment.prediction.data import prepare_caches
from experiment.prediction.evaluate import evaluate_checkpoint, load_decoder
from experiment.prediction.settings import load_config
from experiment.prediction.train import train
from experiment.prediction.visualize import load_results
from model import MODEL_FACTORIES
from motion_rep import MotionJEPAMotionRep
from motion_rep.geometry import y_rotation


class PredictionTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        data_root = self.root / "data"
        motions = []
        rep = MotionJEPAMotionRep(fps=30)
        for sample in range(4):
            rotations = torch.eye(3).expand(6, 30, 3, 3).clone()
            rotations[:, 3] = y_rotation(torch.arange(6) * (.04 + sample * .02))
            root_positions = torch.zeros(6, 3)
            root_positions[:, 1] = .1
            root_positions[:, 0] = torch.arange(6) * (.01 + sample * .005)
            motions.append(rep.encode(rotations, root_positions).numpy())
        for split in ("train", "val", "test"):
            write_npy_dataset(data_root, motions, split=split, num_frames=6, fps=30)
        stats = data_root / "stats"
        stats.mkdir()
        values = np.concatenate(motions)
        np.save(stats / "mean.npy", values.mean(0).astype(np.float32))
        np.save(stats / "std.npy", values.std(0).astype(np.float32))
        torch.manual_seed(1)
        encoder = MODEL_FACTORIES["mot_patch_tiny_1d"](in_chans=366, num_frames=6, temporal_patch_size=3)
        checkpoint = self.root / "jepa.pth.tar"
        source_config = {
            "data": {"root_path": str(data_root), "stats_path": "stats", "num_frames": 6,
                     "motion_dim": 366, "num_joints": 30, "fps": 30, "normalize": True},
            "meta": {"model_name": "mot_patch_tiny_1d", "use_bfloat16": False},
            "patch": {"temporal_patch_size": 3},
        }
        torch.save({"format_version": 1, "config": source_config,
                    "encoder": encoder.state_dict(), "target_encoder": encoder.state_dict()}, checkpoint)
        self.config = load_config(overrides={
            "jepa_checkpoint": str(checkpoint), "dataset_root": str(data_root),
            "cache_root": str(self.root / "cache"), "output": str(self.root / "run"),
            "device": "cpu", "epochs": 2, "batch_size": 2, "cache_batch_size": 2,
            "num_workers": 0, "warmup_epochs": 0, "use_bfloat16": False,
            "tensorboard": False, "export_count": 2,
            "decoder": {"hidden_dim": 24, "depth": 1, "num_heads": 2, "ffn_dim": 48, "dropout": .1},
        })
        prepare_caches(self.config)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        self.temporary.cleanup()

    def test_checkpoint_reload_exports_and_exact_resume(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            uninterrupted = train(self.config)
            resumed_config = dict(self.config, output=str(self.root / "resumed"))
            interrupted = train(resumed_config, stop_after_epoch=1)
            self.assertFalse(interrupted["complete"])
            resumed = train(resumed_config, resume=self.root / "resumed/latest.pth.tar")
        self.assertTrue(uninterrupted["complete"] and resumed["complete"])
        final = torch.load(self.root / "run/latest.pth.tar", weights_only=False)
        restored = torch.load(self.root / "resumed/latest.pth.tar", weights_only=False)
        for name, value in final["decoder"].items():
            torch.testing.assert_close(value, restored["decoder"][name], rtol=0, atol=0)
        for left, right in zip(uninterrupted["history"], resumed["history"]):
            for name in ("epoch", "global_step", "train_mse", "val_mse", "lr"):
                self.assertEqual(left[name], right[name])
        with self.assertRaisesRegex(FileExistsError, "destination"):
            train(self.config, resume=self.root / "resumed/latest.pth.tar")
        self.assertNotIn("encoder", final)
        self.assertNotIn("target_encoder", final)
        self.assertEqual(final["feature_transform"], "jepa_target_layer_norm")
        model, saved = load_decoder(self.root / "run/best.pth.tar", torch.device("cpu"))
        model_again, _ = load_decoder(self.root / "run/best.pth.tar", torch.device("cpu"))
        tokens = torch.randn(1, 2, 192)
        frames = torch.ones(1, 6, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(model(tokens, torch.tensor([30.]), frames),
                                       model_again(tokens, torch.tensor([30.]), frames), rtol=0, atol=0)
        results = load_results(self.root / "run/test-reconstructions.npz")
        self.assertEqual(results.target_motion.shape, (2, 6, 366))
        self.assertTrue((self.root / "run/test-metrics.json").is_file())
        self.assertTrue((self.root / "run/metrics.csv").is_file())
        evaluation_config = self.root / "evaluation.yaml"
        evaluation_config.write_text(f"output: {self.root / 'evaluation'}\nexport_count: 1\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            evaluate_checkpoint(self.root / "run/best.pth.tar", config_path=evaluation_config)
        self.assertTrue((self.root / "evaluation/test-metrics.json").is_file())
        self.assertEqual(load_results(self.root / "evaluation/test-reconstructions.npz").target_motion.shape[0], 1)
        # Moving a completed run must evaluate into its current output directory.
        relocated = dict(resumed_config, output=str(self.root / "relocated"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train(relocated, resume=self.root / "resumed/latest.pth.tar")
        self.assertTrue((self.root / "relocated/test-metrics.json").is_file())
        changed = dict(resumed_config, batch_size=1)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            train(changed, resume=self.root / "resumed/latest.pth.tar")

    def test_small_training_set_overfits(self):
        config = dict(self.config, epochs=80, lr=.01, final_lr=.001,
                      decoder={**self.config["decoder"], "dropout": 0.})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = train(config)
        self.assertLess(result["history"][-1]["train_mse"], result["history"][0]["train_mse"] * .5)
        self.assertLess(result["test"]["reconstruction"]["mse"], result["test"]["zero_output_baseline"]["mse"])
        self.assertTrue(np.isfinite(list(result["test"]["reconstruction"].values())).all())


if __name__ == "__main__":
    unittest.main()
