"""Regression tests for structured deterministic MotionJEPA masks."""

from __future__ import annotations

import unittest

import torch

from mask import MaskCollator1D, MaskCollator2D


class MaskCollatorTest(unittest.TestCase):
    def setUp(self):
        self.batch_1d = [(torch.zeros(12, 6), 60) for _ in range(2)]
        self.batch_2d = [(torch.zeros(12, 366), 60) for _ in range(2)]

    def test_1d_state_restores_next_mask(self):
        first = MaskCollator1D(12, (0.5, 0.5), (0.25, 0.25), nenc=2, npred=2)
        first(self.batch_1d)
        state = first.state_dict()
        expected = first(self.batch_1d)[1:]
        restored = MaskCollator1D(12, (0.5, 0.5), (0.25, 0.25), nenc=2, npred=2)
        restored.load_state_dict(state)
        actual = restored(self.batch_1d)[1:]
        for expected_group, actual_group in zip(expected, actual):
            for expected_mask, actual_mask in zip(expected_group, actual_group):
                torch.testing.assert_close(actual_mask, expected_mask)

    def test_2d_context_excludes_targets_when_overlap_is_disabled(self):
        collator = MaskCollator2D(
            12,
            30,
            enc_frame_mask_ratio=(0.8, 0.8),
            enc_joint_mask_ratio=(0.8, 0.8),
            pred_frame_mask_ratio=(0.25, 0.25),
            pred_joint_mask_ratio=(0.25, 0.25),
            nenc=2,
            npred=3,
            allow_overlap=False,
        )
        _, contexts, targets = collator(self.batch_2d)
        target_union = torch.stack(targets).any(dim=0)
        for context in contexts:
            self.assertFalse(bool((context & target_union).any()))
            self.assertTrue(bool(context.any()))

    def test_2d_overlap_can_represent_the_same_cells(self):
        collator = MaskCollator2D(
            4,
            30,
            enc_frame_mask_ratio=(1.0, 1.0),
            enc_joint_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(1.0, 1.0),
            pred_joint_mask_ratio=(1.0, 1.0),
            allow_overlap=True,
        )
        _, contexts, targets = collator([(torch.zeros(4, 366), 60)])
        self.assertTrue(bool((contexts[0] & targets[0]).all()))


if __name__ == "__main__":
    unittest.main()
