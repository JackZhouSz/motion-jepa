"""Position configuration, legacy checkpoints, and frozen feature provenance."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from _npy_fixture import write_npy_dataset
from experiment.linear_probe.features import load_frozen_encoder
from experiment.segmentation_probe.online import _matches_provenance
from helper import (
    architecture_signature,
    architecture_signature_from_config,
    init_mjepa_encoder_from_config,
    init_mjepa_model_from_config,
    normalize_architecture_signature,
    position_encoding_from_config,
    position_encoding_from_model,
)
from train import _load_checkpoint, main as train_main


def _config(patchified=True):
    result = {
        "data": {"num_frames": 24, "motion_dim": 6, "num_joints": 30, "fps": 30},
        "meta": {
            "model_name": "mot_patch_tiny_1d" if patchified else "mot_tiny_1d",
            "predictor_name": (
                "mot_predictor_patch_tiny_1d" if patchified else "mot_predictor_tiny_1d"
            ),
        },
    }
    if patchified:
        result["patch"] = {"temporal_patch_size": 3}
    return result


class PositionEncodingConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_legacy_defaults_and_invalid_configuration(self):
        self.assertEqual(position_encoding_from_config({}), {"temporal": "absolute"})
        self.assertEqual(
            position_encoding_from_config({"position_encoding": {"temporal": "rope"}}),
            {"temporal": "rope", "rope_theta": 100.0, "rope_time_scale": 1.0},
        )
        for settings in (
            "rope", None, {"temporal": "unknown"}, {"rope_thetta": 100.0},
            {"temporal": "rope", "rope_theta": 0},
            {"temporal": "rope", "rope_theta": float("inf")},
            {"temporal": "rope", "rope_time_scale": -1},
            {"temporal": "rope", "rope_time_scale": float("nan")},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                position_encoding_from_config({"position_encoding": settings})

    def test_signatures_and_frozen_encoder_preserve_position_settings(self):
        motion = torch.randn(2, 24, 6)
        fps = torch.tensor([30, 60])
        for patchified in (False, True):
            for mode in ("absolute", "rope"):
                with self.subTest(patchified=patchified, mode=mode):
                    config = _config(patchified)
                    if mode == "rope":
                        config["position_encoding"] = {
                            "temporal": "rope", "rope_theta": 256.0, "rope_time_scale": 7.0,
                        }
                    encoder, predictor = init_mjepa_model_from_config(config, torch.device("cpu"))
                    signature = architecture_signature(
                        encoder, predictor, **config["meta"], motion_dim=6,
                    )
                    self.assertEqual(signature, architecture_signature_from_config(config))
                    self.assertEqual(
                        position_encoding_from_model(predictor), position_encoding_from_config(config),
                    )
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "encoder.pth.tar"
                        torch.save({
                            "format_version": 1, "config": config,
                            "target_encoder": encoder.state_dict(),
                        }, path)
                        loaded, _, _ = load_frozen_encoder(path, "target_encoder", torch.device("cpu"))
                    self.assertEqual(
                        position_encoding_from_model(loaded), position_encoding_from_config(config),
                    )
                    with torch.inference_mode():
                        torch.testing.assert_close(
                            loaded(motion, fps), encoder.eval()(motion, fps), rtol=0, atol=0,
                        )

    def test_2d_rope_is_rejected_at_all_config_entry_points(self):
        for patchified in (False, True):
            config = _config(patchified)
            config["data"]["motion_dim"] = 366
            config["meta"] = {key: value.replace("_1d", "_2d")
                              for key, value in config["meta"].items()}
            config["position_encoding"] = {"temporal": "rope"}
            for factory in (init_mjepa_model_from_config, init_mjepa_encoder_from_config):
                with self.subTest(patchified=patchified, factory=factory.__name__):
                    with self.assertRaisesRegex(ValueError, "only 1D"):
                        factory(config, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "only 1D"):
                architecture_signature_from_config(config)

    def test_legacy_signature_and_cache_provenance_mean_absolute(self):
        signature = architecture_signature_from_config(_config())
        legacy = {key: value for key, value in signature.items() if key != "position_encoding"}
        self.assertEqual(normalize_architecture_signature(legacy), signature)
        self.assertNotIn("position_encoding", legacy)
        old_cache = {"target_encoder_sha256": "same-weights", "feature_dim": 192}
        absolute = {**old_cache, "position_encoding": {"temporal": "absolute"}}
        rope = {**old_cache, "position_encoding": {
            "temporal": "rope", "rope_theta": 100.0, "rope_time_scale": 1.0,
        }}
        self.assertTrue(_matches_provenance(old_cache, absolute))
        self.assertFalse(_matches_provenance(old_cache, rope))
        self.assertFalse(_matches_provenance(absolute, rope))
        changed_scale = copy.deepcopy(rope)
        changed_scale["position_encoding"]["rope_time_scale"] = 30.0
        self.assertFalse(_matches_provenance(rope, changed_scale))

    def test_rope_example_retains_baseline_training_and_mask_settings(self):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        baseline = yaml.safe_load((config_root / "mjepa_patch_1d_base_v2.yaml").read_text())
        rope = yaml.safe_load((config_root / "mjepa_patch_1d_base_v2_rope.yaml").read_text())
        self.assertEqual(position_encoding_from_config(rope)["temporal"], "rope")
        self.assertNotEqual(baseline["logging"]["folder"], rope["logging"]["folder"])
        self.assertNotEqual(baseline["logging"]["write_tag"], rope["logging"]["write_tag"])
        rope.pop("position_encoding")
        for key in ("folder", "write_tag"):
            rope["logging"][key] = baseline["logging"][key]
        self.assertEqual(rope, baseline)

    def _assert_resume_mismatch(self, path, architecture):
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            _load_checkpoint(
                path, device=torch.device("cpu"), encoder=None, predictor=None,
                target_encoder=None, optimizer=None, scaler=None, lr_scheduler=None,
                wd_scheduler=None, momentum_scheduler=None, mask_collator=None,
                rank=0, world_size=1, architecture=architecture,
            )

    def test_v2_exact_resume_legacy_absolute_and_rope_and_reject_position_change(self):
        for mode, patchified in (("absolute", True), ("rope", False), ("rope", True)):
            with self.subTest(mode=mode, patchified=patchified), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset = root / "dataset"
                write_npy_dataset(dataset, [
                    np.random.default_rng(length).normal(size=(length, 6)).astype(np.float32)
                    for length in (12, 17, 24)
                ], num_frames=24, fps=30)
                config = _config(patchified)
                config["data"].update(
                    root_path=str(dataset), meta_files=["train.txt"], batch_size=3,
                    num_workers=0, pin_mem=False, persistent_workers=False,
                    drop_last=True, normalize=False,
                )
                config["logging"] = {
                    "folder": str(root / "full"), "write_tag": "positions",
                    "log_freq": 1, "checkpoint_freq": 1,
                }
                config["mask"] = {
                    "version": "v2", "allow_overlap": False, "num_enc_masks": 2,
                    "context_selection": "random", "num_pred_masks": 2,
                    "enc_frame_mask_ratio": [0.85, 1.0], "pred_frame_mask_ratio": [0.25, 0.25],
                    "min_context_ratio": 0.2,
                }
                config["meta"].update(seed=0, load_checkpoint=False, use_bfloat16=False)
                config["optimization"] = {
                    "ema": [0.9, 1.0], "epochs": 2, "final_lr": 1e-5,
                    "final_weight_decay": 0.4, "ipe_scale": 1.0, "lr": 1e-3,
                    "start_lr": 1e-4, "warmup": 0, "weight_decay": 0.04,
                }
                if mode == "rope":
                    config["position_encoding"] = {"temporal": "rope"}
                full_result = train_main(config, device="cpu")
                full = torch.load(full_result["checkpoint"], map_location="cpu", weights_only=False)
                first_path = root / "full" / "positions-ep1.pth.tar"
                if mode == "absolute":
                    legacy = torch.load(first_path, map_location="cpu", weights_only=False)
                    legacy["architecture"].pop("position_encoding")
                    torch.save(legacy, first_path)
                resumed_config = copy.deepcopy(config)
                resumed_config["logging"]["folder"] = str(root / "resumed")
                resumed_config["meta"].update(load_checkpoint=True, read_checkpoint=str(first_path))
                result = train_main(resumed_config, device="cpu")
                resumed = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
                self.assertEqual(resumed["global_step"], full["global_step"])
                self.assertEqual(resumed["mask_states"], full["mask_states"])
                for component in ("encoder", "predictor", "target_encoder"):
                    for name in full[component]:
                        torch.testing.assert_close(
                            resumed[component][name], full[component][name], rtol=0, atol=0,
                        )
                other_mode = copy.deepcopy(full["architecture"])
                other_mode["position_encoding"] = {"temporal": "rope" if mode == "absolute" else "absolute"}
                self._assert_resume_mismatch(first_path, other_mode)
                if mode == "rope":
                    for setting, value in (("rope_theta", 10000.0), ("rope_time_scale", 30.0)):
                        changed = copy.deepcopy(full["architecture"])
                        changed["position_encoding"][setting] = value
                        self._assert_resume_mismatch(first_path, changed)


if __name__ == "__main__":
    unittest.main()
