"""Fixed train fractions, nested sampling, and subset-only JEPA statistics."""

import tempfile
import unittest
from pathlib import Path

import torch

from experiment.tmr.dataset import json_digest, select_training_subset
from experiment.tmr.features import write_ragged_bank
from experiment.tmr.normalization import ensure_feature_statistics


class TMRSubsetTests(unittest.TestCase):
    def test_seeded_subsets_are_nested_and_leave_global_rng_unchanged(self):
        records = [{"sample_id": f"train-{index}"} for index in range(101)]
        rng_before = torch.get_rng_state().clone()
        small, metadata = select_training_subset(records, .1, 42)
        large, _ = select_training_subset(records, .5, 42)
        repeat, repeated_metadata = select_training_subset(records, .1, 42)
        different, _ = select_training_subset(records, .1, 43)
        self.assertEqual(len(small), 10)
        self.assertEqual(len(large), 50)
        self.assertEqual(small, repeat)
        self.assertEqual(metadata, repeated_metadata)
        self.assertNotEqual(small, different)
        self.assertNotEqual(small, records[:10])
        small_ids = [record["sample_id"] for record in small]
        self.assertTrue(set(small_ids) <= {record["sample_id"] for record in large})
        self.assertEqual(small, sorted(small, key=records.index))
        self.assertEqual(metadata["sample_ids_sha256"], json_digest(small_ids))
        self.assertEqual(metadata["available_samples"], len(records))
        full, full_metadata = select_training_subset(records, 1, 42)
        self.assertIs(full, records)
        self.assertIsNone(full_metadata)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_invalid_and_too_small_fractions_fail_before_training(self):
        from experiment.tmr.train import build_parser, run

        records = [{"sample_id": str(index)} for index in range(10)]
        for fraction in (0, -1, 1.01, float("nan"), float("inf")):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "fraction"):
                    select_training_subset(records, fraction)
                args = build_parser().parse_args([])
                args.train_fraction = fraction
                with self.assertRaisesRegex(ValueError, "fraction"):
                    run(args)
        for fraction in (.01, .19):
            with self.assertRaisesRegex(ValueError, "at least two"):
                select_training_subset(records, fraction)

    def test_subset_statistics_skip_unselected_tokens_and_preserve_full_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = [
                torch.tensor([[1., 4., 7.], [3., 8., 7.]]),
                torch.full((4, 3), 1000.),
                torch.tensor([[5., 12., 7.]]),
            ]
            bank = write_ragged_bank(root / "jepa/train", ["a", "b", "c"], [2, 4, 1], 3, {}, [sequences])
            _, _, full_metadata = ensure_feature_statistics(bank)
            full_files = {path: path.read_bytes() for path in (root / "jepa/stats").iterdir()}
            mean, std, metadata = ensure_feature_statistics(bank, sample_ids=["a", "c"], chunk_size=1)
            selected = torch.cat([bank["a"], bank["c"]]).double()
            torch.testing.assert_close(mean, selected.mean(0).float())
            torch.testing.assert_close(std, selected.std(0, correction=0).float().clamp_min(1e-6))
            self.assertEqual(metadata["num_tokens"], 3)
            self.assertEqual(metadata["num_samples"], 2)
            self.assertEqual(metadata["sample_ids_sha256"], json_digest(["a", "c"]))
            self.assertEqual(len(list((root / "jepa/stats-subsets").iterdir())), 1)
            repeated = ensure_feature_statistics(bank, sample_ids=["a", "c"])
            self.assertEqual(repeated[2], metadata)
            alternative = ensure_feature_statistics(bank, sample_ids=["b", "c"])
            self.assertNotEqual(alternative[2]["mean_sha256"], metadata["mean_sha256"])
            self.assertEqual(len(list((root / "jepa/stats-subsets").iterdir())), 2)
            self.assertEqual(full_files, {path: path.read_bytes() for path in full_files})
            self.assertEqual(ensure_feature_statistics(bank)[2], full_metadata)
            with self.assertRaisesRegex(ValueError, "unique training"):
                ensure_feature_statistics(bank, sample_ids=["a", "a"])


if __name__ == "__main__":
    unittest.main()
