"""Train-token weighting, statistics integrity, and existing-cache upgrades."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiment.linear_probe.features import _sha256_file
from experiment.tmr.dataset import PreparedPairDataset
from experiment.tmr.features import load_prepared_datasets, prepare_caches, prepare_jepa_statistics, write_ragged_bank
from experiment.tmr.normalization import FEATURE_STD_EPSILON, ensure_feature_statistics
from test_tmr_data import _SpatialEncoder, _fake_text_backbone, _write_dataset


class TMRNormalizationTests(unittest.TestCase):
    def test_streamed_population_statistics_and_constant_channel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = [torch.tensor([[1., 4., 7.], [3., 8., 7.]]), torch.tensor([[5., 12., 7.]])]
            bank = write_ragged_bank(root / "jepa/train", ["a", "b"], [2, 1], 3, {}, [sequences])
            mean, std, metadata = ensure_feature_statistics(bank, chunk_size=2)
            np.testing.assert_allclose(mean, [3, 8, 7])
            np.testing.assert_allclose(np.load(root / "jepa/stats/std.npy"), [np.sqrt(8 / 3), np.sqrt(32 / 3), 0])
            self.assertEqual(metadata["num_tokens"], 3)
            self.assertEqual(metadata["ddof"], 0)
            self.assertAlmostEqual(float(std[2]), FEATURE_STD_EPSILON)
            text = write_ragged_bank(root / "text", ["caption"], [1], 3, {}, [[torch.ones(1, 3)]])
            records = [{
                "sample_id": key, "caption_id": "caption", "caption_ids": ["caption"],
                "source_id": key, "start_frame": 0, "end_frame": len(sequence),
            } for key, sequence in zip(bank.keys, sequences)]
            dataset = PreparedPairDataset(records, text, motion_bank=bank, motion_mean=mean, motion_std=std)
            tokens = torch.cat([dataset[row]["motion_tokens"] for row in range(len(dataset))])
            torch.testing.assert_close(tokens.mean(0), torch.zeros(3))
            torch.testing.assert_close(tokens.std(0, correction=0), torch.tensor([1., 1., 0.]))
            again = ensure_feature_statistics(bank, recompute=True, chunk_size=1)
            torch.testing.assert_close(again[0], mean, rtol=0, atol=0)
            torch.testing.assert_close(again[1], std, rtol=0, atol=0)
            self.assertEqual(again[2], metadata)
            np.save(root / "jepa/stats/mean.npy", np.zeros(3, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "stale"):
                ensure_feature_statistics(bank)
            ensure_feature_statistics(bank, recompute=True)
            values = np.load(root / "jepa/train/values.npy", mmap_mode="r+")
            values[0, 0] = 0
            values.flush()
            with self.assertRaisesRegex(ValueError, "stale"):
                ensure_feature_statistics(bank)

    def test_existing_cache_upgrade_and_statistics_only_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            cache = Path(temporary) / "cache"
            checkpoint = Path(temporary) / "jepa.pth"
            checkpoint.write_bytes(b"fake weights")
            info = {"num_frames": 4, "fps": 2, "motion_dim": 3, "feature_dim": 3, "token_num_frames": 2}
            with patch("experiment.tmr.features.load_text_backbone", side_effect=_fake_text_backbone), patch("experiment.tmr.features.load_frozen_encoder", return_value=(_SpatialEncoder(), {}, info)):
                prepare_caches(
                    dataset_root=root, annotations_path=annotations, cache_root=cache,
                    input_source="jepa", jepa_checkpoint=checkpoint, stats_path=root / "stats",
                    text_model="fake", device="cpu",
                )
            marker = cache / "jepa/prepared.json"
            metadata = json.loads(marker.read_text())
            original_stats = metadata.pop("jepa_feature_stats")
            marker.write_text(json.dumps(metadata))
            shutil.rmtree(cache / "jepa/stats")
            banks = [cache / "text", *[cache / "jepa" / split for split in ("train", "val", "test")]]
            hashes = {path: _sha256_file(path) for bank in banks for path in bank.iterdir()}
            with patch("experiment.tmr.features.load_text_backbone", side_effect=AssertionError), patch("experiment.tmr.features.load_frozen_encoder", side_effect=AssertionError):
                datasets, upgraded = load_prepared_datasets(root, annotations, cache, "jepa", checkpoint, text_model="fake")
                self.assertEqual(upgraded["jepa_feature_stats"], original_stats)
                self.assertEqual(json.loads(marker.read_text()), upgraded)
                for split in ("train", "val", "test"):
                    torch.testing.assert_close(datasets[split].motion_mean, datasets["train"].motion_mean)
                    torch.testing.assert_close(datasets[split].motion_std, datasets["train"].motion_std)
                np.save(cache / "jepa/stats/std.npy", np.ones(3, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "stale"):
                    load_prepared_datasets(root, annotations, cache, "jepa", checkpoint, text_model="fake")
                repaired = prepare_jepa_statistics(cache, recompute=True)
                self.assertEqual(repaired, original_stats)
                load_prepared_datasets(root, annotations, cache, "jepa", checkpoint, text_model="fake")
            self.assertEqual(hashes, {path: _sha256_file(path) for path in hashes})

    def test_empty_train_bank_cannot_compute_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            bank = write_ragged_bank(Path(temporary) / "jepa/train", [], [], 3, {}, [])
            with self.assertRaisesRegex(ValueError, "nonempty"):
                ensure_feature_statistics(bank)


if __name__ == "__main__":
    unittest.main()
