"""Deterministic motion decoding and physically meaningful reconstruction metrics."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from experiment.prediction.metrics import ReconstructionMetrics, masked_mse
from experiment.prediction.model import DecoderConfig, MotionDecoder
from model.token_layout import TokenLayout
from motion_rep.geometry import cont6d_to_matrix, matrix_to_cont6d, y_rotation


class MotionDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = DecoderConfig(hidden_dim=24, depth=2, num_heads=3, ffn_dim=48, dropout=0.15)

    @staticmethod
    def _layout(kind: str, patchified: bool) -> TokenLayout:
        patch = 3 if patchified else 1
        joint_kwargs = {"raw_num_joints": 30, "token_num_joints": 12 if patchified else 30} if kind == "2d" else {}
        return TokenLayout(
            kind=kind,
            patchified=patchified,
            raw_num_frames=8,
            token_num_frames=8 // patch,
            temporal_patch_size=patch,
            **joint_kwargs,
        )

    def _tokens(self, layout: TokenLayout) -> torch.Tensor:
        shape = (2, layout.token_num_frames)
        if layout.kind == "2d":
            shape += (int(layout.token_num_joints),)
        return torch.randn(*shape, 12)

    def test_four_layouts_decode_raw_frames_with_deterministic_eval(self) -> None:
        for kind in ("1d", "2d"):
            for patchified in (False, True):
                with self.subTest(kind=kind, patchified=patchified):
                    layout = self._layout(kind, patchified)
                    model = MotionDecoder(12, layout, config=self.config).eval()
                    tokens = self._tokens(layout)
                    valid = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
                    with torch.no_grad():
                        output = model(tokens, torch.tensor([30, 60]), valid)
                        repeated = model(tokens, torch.tensor([30, 60]), valid)
                    self.assertEqual(output.shape, (2, 8, 366))
                    self.assertTrue(torch.isfinite(output).all())
                    torch.testing.assert_close(output, repeated, atol=0, rtol=0)
                    self.assertEqual(torch.count_nonzero(output[~valid]).item(), 0)
                    if patchified:
                        # Both the full clip's uncovered tail and the shorter
                        # clip's incomplete patch still have frame queries.
                        self.assertGreater(float(output[0, 6:].abs().sum()), 0)
                        self.assertGreater(float(output[1, 3:5].abs().sum()), 0)

    def test_padding_cannot_affect_valid_predictions(self) -> None:
        for kind in ("1d", "2d"):
            with self.subTest(kind=kind):
                layout = self._layout(kind, True)
                model = MotionDecoder(12, layout, config=self.config).eval()
                tokens = self._tokens(layout)
                valid = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
                active = layout.valid_token_mask(valid)
                if kind == "2d":
                    active = active.unsqueeze(-1).expand(-1, -1, int(layout.token_num_joints))
                altered = tokens.masked_fill(~active[..., None], float("nan"))
                with torch.no_grad():
                    original = model(tokens, torch.tensor([30, 30]), valid)
                    changed = model(altered, torch.tensor([30, 30]), valid)
                torch.testing.assert_close(original, changed)

    def test_spatial_token_mask_is_respected_and_temporal_mask_broadcasts(self) -> None:
        layout = self._layout("2d", True)
        model = MotionDecoder(12, layout, config=self.config).eval()
        tokens = self._tokens(layout)
        valid = torch.ones(2, 8, dtype=torch.bool)
        grid_mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
        grid_mask[:, :, 0] = False
        altered = tokens.masked_fill(~grid_mask[..., None], float("inf"))
        with torch.no_grad():
            reference = model(tokens, torch.tensor([30, 30]), valid, token_mask=grid_mask)
            actual = model(altered, torch.tensor([30, 30]), valid, token_mask=grid_mask)
            temporal_mask = torch.tensor([[True, False], [True, False]])
            temporal = model(tokens, torch.tensor([30, 30]), valid, token_mask=temporal_mask)
            expanded = model(tokens, torch.tensor([30, 30]), valid, token_mask=temporal_mask[..., None].expand_as(grid_mask))
        torch.testing.assert_close(reference, actual)
        torch.testing.assert_close(temporal, expanded)

    def test_valid_tail_frames_contribute_to_mse_and_gradient(self) -> None:
        model = MotionDecoder(12, self._layout("1d", True), config=self.config)
        valid = torch.arange(8)[None] < torch.tensor([5, 5])[:, None]
        output = model(torch.randn(2, 2, 12), torch.tensor([30, 30]), valid)
        output.retain_grad()
        target = output.detach().clone()
        target[:, 3:5] += 1
        target[:, 5:] = float("nan")
        loss = masked_mse(output, target, valid)
        torch.testing.assert_close(loss, torch.tensor(0.4))
        loss.backward()
        self.assertGreater(float(output.grad[:, 3:5].abs().sum()), 0)
        self.assertEqual(torch.count_nonzero(output.grad[:, 5:]).item(), 0)
        self.assertGreater(float(model.input_projection.weight.grad.abs().sum()), 0)

    def test_empty_memory_and_invalid_inputs_are_descriptive(self) -> None:
        layout = self._layout("1d", True)
        model = MotionDecoder(12, layout, config=self.config)
        tokens = self._tokens(layout)
        valid = torch.ones(2, 8, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "valid JEPA memory"):
            model(tokens, torch.tensor([30, 30]), torch.arange(8)[None].expand(2, -1) < 2)
        with self.assertRaisesRegex(ValueError, "valid JEPA memory"):
            model(tokens, torch.tensor([30, 30]), valid, torch.zeros(2, 2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "positive finite"):
            model(tokens, torch.tensor([30, 0]), valid)
        with self.assertRaisesRegex(ValueError, "Expected tokens"):
            model(torch.randn(2, 3, 12), torch.tensor([30, 30]), valid)
        with self.assertRaisesRegex(ValueError, "finite"):
            model(torch.full_like(tokens, float("nan")), torch.tensor([30, 30]), valid)

    def test_state_dict_roundtrip(self) -> None:
        layout = self._layout("2d", True)
        original = MotionDecoder(12, layout, config=self.config).eval()
        restored = MotionDecoder(12, layout, config=self.config).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        tokens = self._tokens(layout)
        valid = torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(
                original(tokens, torch.tensor([30, 30]), valid),
                restored(tokens, torch.tensor([30, 30]), valid),
            )


class ReconstructionMetricsTest(unittest.TestCase):
    @staticmethod
    def _motion() -> torch.Tensor:
        with np.load(Path(__file__).with_name("assets") / "motion_jepa_golden.npz", allow_pickle=False) as fixture:
            return torch.from_numpy(fixture["features"]).clone()

    def test_identity_is_zero_in_feature_and_physical_spaces(self) -> None:
        raw = self._motion().unsqueeze(0)
        mean, std = torch.linspace(-0.2, 0.2, 366), torch.linspace(0.5, 1.5, 366)
        normalized = (raw - mean) / std
        valid = torch.arange(8)[None] < 5
        normalized[:, 5:] = float("nan")
        metrics = ReconstructionMetrics(mean, std, fps=30)
        metrics.update(normalized, normalized, valid)
        result = metrics.compute()
        for name, value in result.items():
            self.assertAlmostEqual(value, 1.0 if name == "contact_f1" else 0.0, places=7, msg=name)

    def test_frame_weighting_matches_one_combined_update(self) -> None:
        target = self._motion()[:4].unsqueeze(0)
        prediction = target.clone()
        prediction[:, 0, 0] += 2
        mean, std = torch.zeros(366), torch.ones(366)
        metrics = ReconstructionMetrics(mean, std, fps=30)
        metrics.update(prediction[:, :1], target[:, :1], torch.ones(1, 1, dtype=torch.bool))
        metrics.update(prediction[:, 1:], target[:, 1:], torch.ones(1, 3, dtype=torch.bool))
        combined = ReconstructionMetrics(mean, std, fps=30)
        combined.update(prediction, target, torch.ones(1, 4, dtype=torch.bool))
        for name, value in metrics.compute().items():
            self.assertAlmostEqual(value, combined.compute()[name], places=7, msg=name)
        self.assertAlmostEqual(metrics.compute()["mse"], 1 / 366, places=7)
        self.assertAlmostEqual(metrics.compute()["root_error_mm"], 500.0, places=5)

    def test_contact_f1_thresholds_denormalized_features(self) -> None:
        target = self._motion()[:2].unsqueeze(0)
        target[..., 362:] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        prediction = target.clone()
        prediction[..., 362:] = 0.6
        mean, std = torch.zeros(366), torch.ones(366)
        mean[362:], std[362:] = 0.8, 0.1
        normalized_prediction, normalized_target = (prediction - mean) / std, (target - mean) / std
        self.assertTrue((normalized_prediction[..., 362:] < 0.5).all())
        metrics = ReconstructionMetrics(mean, std, fps=30)
        metrics.update(normalized_prediction, normalized_target, torch.ones(1, 2, dtype=torch.bool))
        self.assertAlmostEqual(metrics.compute()["contact_f1"], 2 / 3)

    def test_fk_metrics_follow_rotations_and_root_not_local_position_fields(self) -> None:
        target = self._motion()[:1].unsqueeze(0)
        prediction = target.clone()
        prediction[..., 5:92] += 10
        metrics = ReconstructionMetrics(torch.zeros(366), torch.ones(366), fps=30)
        metrics.update(prediction, target, torch.ones(1, 1, dtype=torch.bool))
        self.assertGreater(metrics.compute()["local_positions_mse"], 0)
        self.assertEqual(metrics.compute()["mpjpe_mm"], 0)
        root_rotation = cont6d_to_matrix(target[..., 92:98])
        prediction[..., 92:98] = matrix_to_cont6d(
            y_rotation(torch.tensor(torch.pi / 2, dtype=target.dtype)) @ root_rotation
        )
        rotated = ReconstructionMetrics(torch.zeros(366), torch.ones(366), fps=30)
        rotated.update(prediction, target, torch.ones(1, 1, dtype=torch.bool))
        self.assertGreater(rotated.compute()["mpjpe_mm"], 0)
        self.assertAlmostEqual(rotated.compute()["rotation_error_deg"], 3.0, places=4)

    def test_empty_and_non_motion_jepa_metrics_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "366-dimensional"):
            ReconstructionMetrics(torch.zeros(6), torch.ones(6), fps=30)
        metrics = ReconstructionMetrics(torch.zeros(366), torch.ones(366), fps=30)
        with self.assertRaisesRegex(ValueError, "No valid"):
            metrics.compute()
        with self.assertRaisesRegex(ValueError, "only motion_jepa_366"):
            metrics.update(torch.zeros(1, 2, 6), torch.zeros(1, 2, 6), torch.ones(1, 2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            masked_mse(torch.zeros(1, 2, 6), torch.zeros(1, 2, 6), torch.zeros(1, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
