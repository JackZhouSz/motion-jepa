"""V2 configuration, masked loss, and mixed-length checkpoint integration."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from _npy_fixture import write_npy_dataset
from mask import MaskCollator1D, MaskCollator1DV2, PatchMaskCollator1DV2
from mask.utils import index_mask_validity, repeat_mask_blocks
from model import TokenLayout
from train import _build_mask_collator, _prediction_loss, main as train_main


class V2TrainingTest(unittest.TestCase):
    def test_mask_visualization_excludes_padding_indices(self):
        from visualize_mask import _mask_to_numpy

        for patchified in (False, True):
            layout = TokenLayout("1d", patchified, 150, 50 if patchified else 150,
                                 temporal_patch_size=3 if patchified else 1)
            rendered = _mask_to_numpy(
                torch.tensor([[0, 2, -1]]), layout=layout, sample_index=0,
            )
            self.assertEqual(np.flatnonzero(rendered).tolist(), [0, 2])

    def test_sample_balanced_loss_ignores_padding_in_every_mask_combination(self):
        masks = [
            torch.tensor([[0, -1, -1], [0, 1, 2]]),
            torch.tensor([[1, -1, -1], [1, 2, 3]]),
        ]
        active = repeat_mask_blocks(index_mask_validity(masks), 2, 2)
        prediction = torch.arange(8 * 3 * 2, dtype=torch.float32).reshape(8, 3, 2) / 10
        prediction[~active] = 999.0
        prediction.requires_grad_(True)
        target = torch.zeros_like(prediction)
        expected = torch.stack([
            F.smooth_l1_loss(row[valid], truth[valid])
            for row, truth, valid in zip(prediction, target, active)
        ]).mean()
        actual = _prediction_loss(
            prediction, target, masks, num_enc_masks=2, kind="1d",
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(bool((prediction.grad[~active] == 0).all()))
        self.assertTrue(bool((prediction.grad[active] != 0).any()))

    def test_legacy_loss_is_unchanged(self):
        prediction = torch.tensor([[[1.0], [2.0]]])
        target = torch.zeros_like(prediction)
        masks = [torch.tensor([[0, 1]])]
        expected = F.smooth_l1_loss(prediction, target)
        for kind in ("1d", "2d"):
            torch.testing.assert_close(
                _prediction_loss(prediction, target, masks, num_enc_masks=1, kind=kind),
                expected,
            )

    def test_configs_select_v2_and_legacy_default_stays_v1(self):
        project = Path(__file__).resolve().parents[1]
        for patchified in (False, True):
            path = project / "configs" / (
                "mjepa_patch_1d_base_v2.yaml" if patchified else "mjepa_1d_base_v2.yaml"
            )
            config = yaml.safe_load(path.read_text())
            layout = TokenLayout("1d", patchified, 150, 50 if patchified else 150,
                                 temporal_patch_size=3 if patchified else 1)
            collator = _build_mask_collator(config, layout)
            self.assertIsInstance(collator, PatchMaskCollator1DV2 if patchified else MaskCollator1DV2)
            self.assertEqual(config["mask"]["min_context_ratio"], 0.2)
            self.assertEqual(collator.context_selection, "random")
        config["mask"].pop("version")
        config["mask"].pop("min_context_ratio")
        config["mask"].pop("min_context_tokens")
        self.assertIsInstance(
            _build_mask_collator(config, TokenLayout("1d", False, 150, 150)),
            MaskCollator1D,
        )

    def test_v2_rejects_2d_and_unknown_context_selection(self):
        config = {"mask": {"version": "v2"}}
        with self.assertRaisesRegex(ValueError, "only 1D"):
            _build_mask_collator(config, TokenLayout("2d", False, 150, 150,
                                                   raw_num_joints=30, token_num_joints=30))
        config["mask"]["context_selection"] = "unknown"
        with self.assertRaisesRegex(ValueError, "context_selection"):
            _build_mask_collator(config, TokenLayout("1d", False, 150, 150))

    @staticmethod
    def _config(dataset, output, patchified):
        config = {
            "data": {
                "batch_size": 3, "root_path": str(dataset), "meta_files": ["train.txt"],
                "num_workers": 0, "pin_mem": False, "persistent_workers": False,
                "drop_last": True, "num_frames": 24, "fps": 30, "motion_dim": 6,
                "num_joints": 30, "normalize": False,
            },
            "logging": {
                "folder": str(output), "write_tag": "v2-smoke", "log_freq": 1,
                "checkpoint_freq": 1,
            },
            "mask": {
                "version": "v2", "allow_overlap": False, "num_enc_masks": 2,
                "context_selection": "random",
                "num_pred_masks": 2, "enc_frame_mask_ratio": [0.85, 1.0],
                "pred_frame_mask_ratio": [0.25, 0.25], "min_context_ratio": 0.2,
            },
            "meta": {
                "seed": 0, "load_checkpoint": False, "read_checkpoint": None,
                "model_name": "mot_patch_tiny_1d" if patchified else "mot_tiny_1d",
                "predictor_name": "mot_predictor_patch_tiny_1d" if patchified else "mot_predictor_tiny_1d",
                "use_bfloat16": False, "use_float16": False,
            },
            "optimization": {
                "ema": [0.9, 1.0], "epochs": 2, "final_lr": 1e-5,
                "final_weight_decay": 0.4, "ipe_scale": 1.0, "lr": 1e-3,
                "start_lr": 1e-4, "warmup": 0, "weight_decay": 0.04,
            },
        }
        if patchified:
            config["patch"] = {"temporal_patch_size": 3}
        return config

    def test_mixed_lengths_train_and_resume_match_uninterrupted_weights(self):
        for patchified in (False, True):
            with self.subTest(patchified=patchified), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root / "dataset"
                write_npy_dataset(dataset, [
                    np.random.default_rng(length).normal(size=(length, 6)).astype(np.float32)
                    for length in (12, 17, 24)
                ], num_frames=24, fps=30)
                config = self._config(dataset, root / "full", patchified)
                full_result = train_main(config, device="cpu")
                self.assertEqual(full_result["global_step"], 2)
                full = torch.load(full_result["checkpoint"], map_location="cpu", weights_only=False)
                resumed_config = copy.deepcopy(config)
                resumed_config["logging"]["folder"] = str(root / "resumed")
                resumed_config["meta"].update(
                    load_checkpoint=True,
                    read_checkpoint=str(root / "full" / "v2-smoke-ep1.pth.tar"),
                )
                resumed_result = train_main(resumed_config, device="cpu")
                resumed = torch.load(resumed_result["checkpoint"], map_location="cpu", weights_only=False)
                self.assertEqual(resumed_result["global_step"], 2)
                self.assertEqual(resumed["mask_states"], full["mask_states"])
                for component in ("encoder", "predictor", "target_encoder"):
                    for name in full[component]:
                        torch.testing.assert_close(
                            resumed[component][name], full[component][name], rtol=0, atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
