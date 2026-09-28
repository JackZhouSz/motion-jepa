"""End-to-end CPU smoke test for patchified MotionJEPA."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _npy_fixture import write_npy_dataset
from train import main as train_main


class PatchTrainingSmokeTest(unittest.TestCase):
    def _config(self, dataset: Path, output: Path) -> dict:
        return {
            "data": {
                "batch_size": 2,
                "root_path": str(dataset),
                "meta_files": ["train.txt"],
                "num_workers": 0,
                "pin_mem": False,
                "persistent_workers": False,
                "drop_last": True,
                "num_frames": 6,
                "fps": 60,
                "motion_dim": 6,
                "num_joints": 30,
                "normalize": False,
                "stats_path": None,
            },
            "patch": {"temporal_patch_size": 3},
            "logging": {
                "folder": str(output),
                "write_tag": "patch-smoke",
                "log_freq": 1,
                "checkpoint_freq": 1,
            },
            "mask": {
                "allow_overlap": False,
                "num_enc_masks": 1,
                "num_pred_masks": 1,
                "enc_frame_mask_ratio": [0.5, 0.5],
                "pred_frame_mask_ratio": [0.5, 0.5],
            },
            "meta": {
                "seed": 0,
                "load_checkpoint": False,
                "read_checkpoint": None,
                "model_name": "mot_patch_tiny_1d",
                "predictor_name": "mot_predictor_patch_tiny_1d",
                "use_bfloat16": False,
                "use_float16": False,
            },
            "optimization": {
                "ema": [0.9, 1.0],
                "epochs": 1,
                "final_lr": 1.0e-5,
                "final_weight_decay": 0.4,
                "ipe_scale": 1.0,
                "lr": 1.0e-3,
                "start_lr": 1.0e-4,
                "warmup": 0,
                "weight_decay": 0.04,
            },
        }

    def test_train_resume_and_reject_geometry_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            write_npy_dataset(
                dataset,
                [
                    np.random.default_rng(index).normal(size=(6, 6)).astype(np.float32)
                    for index in range(2)
                ],
            )
            config = self._config(dataset, output)
            result = train_main(config, device="cpu")
            self.assertEqual(result["global_step"], 1)
            self.assertTrue(Path(result["checkpoint"]).is_file())

            config["meta"]["load_checkpoint"] = True
            config["optimization"]["epochs"] = 2
            resumed = train_main(config, device="cpu")
            self.assertEqual(resumed["global_step"], 2)

            mismatched = copy.deepcopy(config)
            mismatched["patch"]["temporal_patch_size"] = 2
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                train_main(mismatched, device="cpu")


if __name__ == "__main__":
    unittest.main()
