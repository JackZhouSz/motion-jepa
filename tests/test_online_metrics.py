"""Unit tests for online representation-quality metric primitives."""

from __future__ import annotations

import unittest

import torch

from experiment.online_metrics import (
    _PredictionAccumulator,
    _TargetMoments,
    _target_metadata,
    covariance_metrics,
    effective_rank,
    feature_matrix_metrics,
    mean_off_diagonal_cosine,
    prediction_gain,
)


class OnlineMetricsTest(unittest.TestCase):
    def test_rankme_distinguishes_rank_one_and_isotropic_features(self):
        rank_one = torch.ones(16, 4)
        isotropic = torch.eye(4)
        self.assertAlmostEqual(effective_rank(rank_one), 1.0, places=6)
        self.assertAlmostEqual(effective_rank(isotropic), 4.0, places=6)
        self.assertLess(
            feature_matrix_metrics(rank_one)["rankme"],
            feature_matrix_metrics(isotropic)["rankme"],
        )

    def test_std_cosine_and_covariance_detect_collapsed_samples(self):
        collapsed = torch.ones(8, 3)
        diverse = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertEqual(feature_matrix_metrics(collapsed)["mean_std"], 0.0)
        self.assertAlmostEqual(mean_off_diagonal_cosine(collapsed), 1.0)
        covariance = covariance_metrics(diverse)
        self.assertGreater(covariance["effective_rank"], 1.0)
        self.assertGreaterEqual(covariance["largest_eigenvalue_ratio"], 0.0)
        self.assertGreaterEqual(covariance["off_diagonal_abs_mean"], 0.0)

    def test_prediction_gain_exact_and_trivial_predictors(self):
        target = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
        moments = _TargetMoments()
        moments.update(target)
        baseline = moments.baseline_mse()
        exact = _PredictionAccumulator()
        exact.update(target, target)
        trivial = _PredictionAccumulator()
        trivial.update(target.mean(dim=0, keepdim=True).expand_as(target), target)
        self.assertAlmostEqual(prediction_gain(exact.summary()["mse"], baseline), 1.0)
        self.assertAlmostEqual(
            prediction_gain(trivial.summary()["mse"], baseline), 0.0
        )
        self.assertEqual(prediction_gain(0.0, 0.0), 0.0)

    def test_target_metadata_preserves_mask_order_and_temporal_distance(self):
        context = torch.zeros(1, 5, 2, dtype=torch.bool)
        context[0, 0, 0] = True
        context[0, 4, 1] = True
        first = torch.zeros_like(context)
        first[0, 1:3, 0] = True
        second = torch.zeros_like(context)
        second[0, 1, 1] = True
        mask_ids, spatial_ids, distances = _target_metadata(
            [context], [first, second]
        )
        torch.testing.assert_close(mask_ids, torch.tensor([0, 0, 1]))
        torch.testing.assert_close(spatial_ids, torch.tensor([0, 0, 1]))
        self.assertEqual(distances, ["1", "2-3", "2-3"])


if __name__ == "__main__":
    unittest.main()
