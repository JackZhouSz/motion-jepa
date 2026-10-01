"""Real-backbone coverage for frozen full-token prediction caches."""

from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from _npy_fixture import write_npy_dataset
from experiment.prediction import data
from model import MODEL_FACTORIES


class PredictionDataTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.dataset_root = self.root / "motions"
        generator = np.random.default_rng(42)
        self.raw = {
            "train": [generator.normal(size=(6, 366)).astype(np.float32),
                      generator.normal(size=(3, 366)).astype(np.float32)],
            "val": [generator.normal(size=(6, 366)).astype(np.float32)],
            "test": [generator.normal(size=(6, 366)).astype(np.float32)],
        }
        for split, motions in self.raw.items():
            write_npy_dataset(self.dataset_root, motions, split=split, num_frames=6, fps=30)
        stats = self.dataset_root / "stats"
        stats.mkdir()
        self.mean = np.linspace(-1., 1., 366, dtype=np.float32)
        self.std = np.linspace(.5, 1.5, 366, dtype=np.float32)
        self.std[0] = 0.
        np.save(stats / "mean.npy", self.mean)
        np.save(stats / "std.npy", self.std)
        self.source_config = {
            "data": {"root_path": str(self.dataset_root), "stats_path": "stats",
                     "num_frames": 6, "motion_dim": 366, "num_joints": 30,
                     "fps": 30, "normalize": True},
            "meta": {"model_name": "mot_patch_tiny_1d", "use_bfloat16": False},
            "patch": {"temporal_patch_size": 3},
        }
        torch.manual_seed(42)
        encoder = MODEL_FACTORIES["mot_patch_tiny_1d"](
            in_chans=366, num_frames=6, temporal_patch_size=3
        )
        self.checkpoint = self.root / "jepa.pth.tar"
        torch.save({"format_version": 1, "config": self.source_config,
                    "encoder": encoder.state_dict(), "target_encoder": encoder.state_dict()},
                   self.checkpoint)
        self.config = {"jepa_checkpoint": str(self.checkpoint),
                       "dataset_root": str(self.dataset_root),
                       "cache_root": str(self.root / "cache"),
                       "cache_batch_size": 1, "num_workers": 0,
                       "device": "cpu", "use_bfloat16": False,
                       "limit_train": 0, "limit_val": 0, "limit_test": 0}
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)
        self.directory.cleanup()

    def test_bf16_tokens_exact_teacher_transform_and_normalized_raw_targets(self):
        loaded = []
        original_loader = data.load_frozen_encoder

        def capture_source(*args, **kwargs):
            result = original_loader(*args, **kwargs)
            encoder = result[0]
            before = copy.deepcopy(encoder.state_dict())
            outputs = []
            encoder.register_forward_hook(lambda _, __, output: outputs.append(output.requires_grad))
            loaded.append((encoder, before, outputs))
            return result

        with patch.object(data, "load_frozen_encoder", side_effect=capture_source):
            data.prepare_caches(self.config)
        encoder, before, outputs = loaded[0]
        self.assertTrue(outputs)
        self.assertFalse(any(outputs))
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                            for parameter in encoder.parameters()))
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

        dataset = data.PredictionDataset(self.config, "train")
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.model_info["token_layout"], dataset.token_layout.signature())
        self.assertEqual(dataset.provenance["feature_transform"], "jepa_target_layer_norm")
        self.assertEqual(dataset.provenance["checkpoint_key"], "target_encoder")
        stored = np.load(self.root / "cache/train/tokens.npy", mmap_mode="r")
        self.assertEqual(stored.dtype, np.uint16)
        self.assertEqual(stored.shape, (2, 2, 192))
        effective_std = np.where(self.std < 1.0e-6, 1., self.std)
        torch.testing.assert_close(dataset.mean, torch.from_numpy(self.mean))
        torch.testing.assert_close(dataset.std, torch.from_numpy(effective_std))
        for index in range(2):
            item = dataset[index]
            self.assertEqual(item["tokens"].dtype, torch.bfloat16)
            self.assertEqual(item["tokens"].shape, (2, 192))
            self.assertEqual(item["sample_id"], f"sample-{index}")
            self.assertEqual(item["fps"], 30.)
            self.assertEqual(item["length"], len(self.raw["train"][index]))
            expected = np.zeros((6, 366), dtype=np.float32)
            expected[:item["length"]] = (self.raw["train"][index] - self.mean) / effective_std
            torch.testing.assert_close(item["motion"], torch.from_numpy(expected), rtol=0, atol=0)
            with torch.no_grad():
                encoded = encoder(item["motion"][None], torch.tensor([30.]),
                                  valid_frames=item["valid_frames"][None])
                active = encoder.token_layout.valid_token_mask(item["valid_frames"][None])
                transformed = torch.zeros_like(encoded)
                transformed[active] = F.layer_norm(encoded[active], (192,))
            torch.testing.assert_close(item["tokens"], transformed[0].to(torch.bfloat16), rtol=0, atol=0)
        self.assertTrue(torch.equal(dataset[1]["tokens"][1], torch.zeros(192, dtype=torch.bfloat16)))
        with self.assertRaises(IndexError):
            dataset[2]
        self.assertEqual(dataset[-1]["sample_id"], "sample-1")

    def test_completed_cache_reused_and_source_change_rejected(self):
        data.prepare_caches(self.config, splits=("train",))
        token_file = self.root / "cache/train/tokens.npy"
        before = token_file.read_bytes()
        data.prepare_caches(self.config, splits=("train",))
        self.assertEqual(token_file.read_bytes(), before)
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        checkpoint["target_encoder"]["norm.bias"] = checkpoint["target_encoder"]["norm.bias"] + .1
        torch.save(checkpoint, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "provenance"):
            data.PredictionDataset(self.config, "train")
        with self.assertRaisesRegex(ValueError, "provenance"):
            data.prepare_caches(self.config, splits=("train",))
        self.assertEqual(token_file.read_bytes(), before)

    def test_stats_layout_dataset_and_subset_mismatch_rejected(self):
        data.prepare_caches(self.config, splits=("train",))
        mean_path = self.dataset_root / "stats/mean.npy"
        np.save(mean_path, self.mean + .2)
        with self.assertRaisesRegex(ValueError, "statistics"):
            data.PredictionDataset(self.config, "train")
        np.save(mean_path, self.mean)
        marker = self.root / "cache/train/completed.json"
        original_metadata = json.loads(marker.read_text())
        changed = copy.deepcopy(original_metadata)
        changed["model_info"]["token_layout"]["temporal_patch_size"] = 1
        marker.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "layout"):
            data.PredictionDataset(self.config, "train")
        marker.write_text(json.dumps(original_metadata))
        limited = dict(self.config, limit_train=1)
        with self.assertRaisesRegex(ValueError, "provenance"):
            data.PredictionDataset(limited, "train")
        meta_path = self.dataset_root / "meta.json"
        changed_meta = json.loads(meta_path.read_text())
        changed_meta["source_dataset"] = "changed"
        meta_path.write_text(json.dumps(changed_meta))
        with self.assertRaisesRegex(ValueError, "provenance"):
            data.PredictionDataset(self.config, "train")

    def test_partial_cache_rejected_and_first_n_limit_is_deterministic(self):
        limited = dict(self.config, limit_train=1)
        data.prepare_caches(limited, splits=("train",))
        dataset = data.PredictionDataset(limited, "train")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["sample_id"], "sample-0")
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(restored._tokens)
        torch.testing.assert_close(restored[0]["tokens"], dataset[0]["tokens"], rtol=0, atol=0)
        (self.root / "cache/train/completed.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
            data.PredictionDataset(limited, "train")
        data.prepare_caches(limited, splits=("train",))
        self.assertEqual(len(data.PredictionDataset(limited, "train")), 1)

    def test_two_dimensional_cache_keeps_spatial_tokens(self):
        source = copy.deepcopy(self.source_config)
        source["meta"]["model_name"] = "mot_patch_tiny_2d"
        source["patch"].update(spatial_grouping="coarse7", spatial_pooling="graph_mean")
        encoder = MODEL_FACTORIES["mot_patch_tiny_2d"](
            in_chans=366, num_frames=6, num_joints=30, temporal_patch_size=3,
            spatial_grouping="coarse7", spatial_pooling="graph_mean",
        )
        torch.save({"format_version": 1, "config": source,
                    "target_encoder": encoder.state_dict()}, self.checkpoint)
        data.prepare_caches(self.config, splits=("val",))
        dataset = data.PredictionDataset(self.config, "val")
        item = dataset[0]
        self.assertEqual(item["tokens"].shape, (2, 8, 192))
        encoder.eval().requires_grad_(False)
        with torch.no_grad():
            expected = encoder(item["motion"][None], torch.tensor([30.]),
                               valid_frames=item["valid_frames"][None])
            expected = F.layer_norm(expected.float(), (192,)).to(torch.bfloat16)
        torch.testing.assert_close(item["tokens"], expected[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
