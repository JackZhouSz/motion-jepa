"""End-to-end CPU smoke tests for 2D patchified Motion-JEPA."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _npy_fixture import write_npy_dataset
from train import main as train_main


class Patch2DTrainingSmokeTest(unittest.TestCase):
    @staticmethod
    def _config(dataset: Path, output: Path, grouping: str) -> dict:
        token_joints = 11 if grouping == "fine11" else 7
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
                "motion_dim": 366,
                "num_joints": 30,
                "normalize": False,
                "stats_path": None,
            },
            "patch": {
                "temporal_patch_size": 3,
                "spatial_grouping": grouping,
                "spatial_pooling": "graph_mean",
            },
            "logging": {
                "folder": str(output),
                "write_tag": f"patch-2d-{grouping}",
                "log_freq": 1,
                "checkpoint_freq": 1,
            },
            "mask": {
                "allow_overlap": True,
                "num_enc_masks": 1,
                "num_pred_masks": 1,
                "enc_frame_mask_ratio": [1.0, 1.0],
                "enc_joint_mask_ratio": [1.0, 1.0],
                "pred_frame_mask_ratio": [0.5, 0.5],
                "pred_joint_mask_ratio": [1.0 / token_joints, 1.0 / token_joints],
            },
            "meta": {
                "seed": 0,
                "load_checkpoint": False,
                "read_checkpoint": None,
                "model_name": "mot_patch_tiny_2d",
                "predictor_name": "mot_predictor_patch_tiny_2d",
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

    def test_both_groupings_train_resume_and_reject_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            write_npy_dataset(
                dataset,
                [
                    np.random.default_rng(index).normal(size=(6, 366)).astype(np.float32)
                    for index in range(2)
                ],
            )
            configs = {}
            for grouping in ("fine11", "coarse7"):
                config = self._config(dataset, root / f"output-{grouping}", grouping)
                configs[grouping] = config
                result = train_main(config, device="cpu")
                self.assertEqual(result["global_step"], 1)
                self.assertTrue(Path(result["checkpoint"]).is_file())

            fine = configs["fine11"]
            fine["meta"]["load_checkpoint"] = True
            fine["optimization"]["epochs"] = 2
            resumed = train_main(fine, device="cpu")
            self.assertEqual(resumed["global_step"], 2)

            mismatched = copy.deepcopy(fine)
            mismatched["patch"]["spatial_grouping"] = "coarse7"
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                train_main(mismatched, device="cpu")

    def test_body_region_segment_trains_one_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            write_npy_dataset(
                dataset,
                [
                    np.random.default_rng(index).normal(size=(6, 366)).astype(np.float32)
                    for index in range(2)
                ],
            )
            config = self._config(dataset, root / "output-body-region", "coarse7")
            config["mask"] = {
                "strategy": "body_region_segment",
                "allow_overlap": False,
                "num_enc_masks": 1,
                "num_pred_masks": 1,
                "num_regions": 2,
                "pred_frame_mask_ratio": [0.5, 0.5],
                "graph_mask_ratio": [2.0 / 7.0, 3.0 / 7.0],
            }
            result = train_main(config, device="cpu")
            self.assertEqual(result["global_step"], 1)
            self.assertTrue(Path(result["checkpoint"]).is_file())

    def test_random_body_segment_trains_one_step_with_four_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            write_npy_dataset(
                dataset,
                [
                    np.random.default_rng(index).normal(size=(6, 366)).astype(np.float32)
                    for index in range(2)
                ],
            )
            config = self._config(dataset, root / "output-random-body", "coarse7")
            config["mask"] = {
                "strategy": "random_body_segment",
                "allow_overlap": False,
                "num_enc_masks": 1,
                "num_pred_masks": 4,
                "pred_frame_mask_ratio": [0.15, 0.25],
                "body_mask_ratio": [1.0 / 7.0, 3.0 / 7.0],
            }
            result = train_main(config, device="cpu")
            self.assertEqual(result["global_step"], 1)
            self.assertTrue(Path(result["checkpoint"]).is_file())


if __name__ == "__main__":
    unittest.main()
