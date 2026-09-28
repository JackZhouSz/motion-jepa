"""Tests for temporal-patch mask collation."""

from __future__ import annotations

import unittest

import torch

from mask import (
    PatchBodyRegionSegmentMaskCollator2D,
    PatchMaskCollator1D,
    PatchMaskCollator2D,
    PatchRandomBodySegmentMaskCollator2D,
    PatchRandomSpatialSegmentMaskCollator2D,
)
from mask.body_region_collator import COARSE7_GRAPH_EDGES
from model import TokenLayout
from train import _build_mask_collator


class PatchMaskCollatorTest(unittest.TestCase):
    def test_mixed_lengths_map_to_complete_patches(self):
        collator = PatchMaskCollator1D(
            8,
            3,
            enc_frame_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(0.5, 0.5),
            allow_overlap=True,
        )
        batch = [(torch.zeros(8, 6), 60, 8), (torch.zeros(8, 6), 60, 5)]
        collated, contexts, targets = collator(batch)
        torch.testing.assert_close(collated[2], torch.tensor([8, 5]))
        for mask in contexts + targets:
            self.assertTrue((mask[0] < 2).all())
            self.assertTrue((mask[1] < 1).all())

    def test_state_restores_next_patch_mask(self):
        kwargs = dict(
            raw_num_frames=12,
            temporal_patch_size=3,
            enc_frame_mask_ratio=(0.5, 0.5),
            pred_frame_mask_ratio=(0.25, 0.25),
            nenc=2,
            npred=2,
        )
        batch = [(torch.zeros(12, 6), 60, 12) for _ in range(2)]
        first = PatchMaskCollator1D(**kwargs)
        first(batch)
        state = first.state_dict()
        expected = first(batch)[1:]
        restored = PatchMaskCollator1D(**kwargs)
        restored.load_state_dict(state)
        actual = restored(batch)[1:]
        for expected_group, actual_group in zip(expected, actual):
            for expected_mask, actual_mask in zip(expected_group, actual_group):
                torch.testing.assert_close(actual_mask, expected_mask)

    def test_rejects_samples_without_a_complete_patch(self):
        collator = PatchMaskCollator1D(8, 3)
        with self.assertRaisesRegex(ValueError, "complete temporal patch"):
            collator([(torch.zeros(8, 6), 60, 2)])

    def test_2d_masks_use_patch_and_group_coordinates(self):
        collator = PatchMaskCollator2D(
            raw_num_frames=8,
            raw_num_joints=30,
            token_num_joints=7,
            temporal_patch_size=3,
            spatial_grouping="coarse7",
            enc_frame_mask_ratio=(1.0, 1.0),
            enc_joint_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(0.5, 0.5),
            pred_joint_mask_ratio=(0.15, 0.15),
            allow_overlap=True,
        )
        batch = [(torch.zeros(8, 366), 60, 8), (torch.zeros(8, 366), 60, 5)]
        collated, contexts, targets = collator(batch)
        torch.testing.assert_close(collated[2], torch.tensor([8, 5]))
        for mask in contexts + targets:
            self.assertEqual(mask.shape, (2, 2, 7))
            self.assertFalse(mask[1, 1:].any())

    def test_2d_state_rejects_different_grouping(self):
        common = dict(
            raw_num_frames=9,
            raw_num_joints=30,
            temporal_patch_size=3,
            enc_frame_mask_ratio=(0.5, 0.5),
            enc_joint_mask_ratio=(0.5, 0.5),
            pred_frame_mask_ratio=(0.25, 0.25),
            pred_joint_mask_ratio=(0.25, 0.25),
        )
        fine = PatchMaskCollator2D(
            token_num_joints=11, spatial_grouping="fine11", **common
        )
        coarse = PatchMaskCollator2D(
            token_num_joints=7, spatial_grouping="coarse7", **common
        )
        joint30 = PatchMaskCollator2D(
            token_num_joints=30, spatial_grouping="joint30", **common
        )
        self.assertEqual(joint30.layout.token_num_joints, 30)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            coarse.load_state_dict(fine.state_dict())

    @staticmethod
    def _body_region_collator(**kwargs):
        options = {
            "raw_num_frames": 12,
            "raw_num_joints": 30,
            "token_num_joints": 7,
            "temporal_patch_size": 3,
            "spatial_grouping": "coarse7",
            "spatial_pooling": "graph_mean",
            "pred_frame_mask_ratio": (0.5, 0.5),
            "graph_mask_ratio": (2.0 / 7.0, 2.0 / 7.0),
        }
        options.update(kwargs)
        return PatchBodyRegionSegmentMaskCollator2D(**options)

    def test_body_region_candidates_cover_every_connected_subset(self):
        collator = self._body_region_collator(
            graph_mask_ratio=(2.0 / 7.0, 3.0 / 7.0)
        )
        expected_pairs = {tuple(sorted(edge)) for edge in COARSE7_GRAPH_EDGES}
        self.assertEqual(set(collator.connected_regions(2)), expected_pairs)
        self.assertEqual(len(collator.connected_regions(3)), 9)
        for region in collator.connected_regions(3):
            selected = set(region)
            selected_edges = [
                edge for edge in COARSE7_GRAPH_EDGES if set(edge) <= selected
            ]
            self.assertEqual(len(selected_edges), 2)

    def test_body_region_is_contiguous_and_context_is_exact_complement(self):
        collator = self._body_region_collator()
        batch = [(torch.zeros(12, 366), 60, 12), (torch.zeros(12, 366), 60, 9)]
        collated, contexts, targets = collator(batch)
        torch.testing.assert_close(collated[2], torch.tensor([12, 9]))
        self.assertEqual(len(contexts), 1)
        self.assertEqual(len(targets), 1)
        context, target = contexts[0], targets[0]
        self.assertEqual(target.flatten(1).sum(1).tolist(), [4, 4])
        for index, valid_frames in enumerate((4, 3)):
            active_frames = torch.nonzero(target[index].any(dim=1)).flatten()
            self.assertEqual(len(active_frames), 2)
            self.assertTrue(bool((active_frames[1:] - active_frames[:-1] == 1).all()))
            groups = target[index, active_frames[0]]
            self.assertEqual(int(groups.sum()), 2)
            self.assertTrue(bool((target[index, active_frames] == groups).all()))
            valid = torch.zeros_like(target[index])
            valid[:valid_frames] = True
            self.assertFalse(bool((context[index] & target[index]).any()))
            torch.testing.assert_close(context[index] | target[index], valid)

    def test_body_region_state_restores_next_mask(self):
        batch = [(torch.zeros(12, 366), 60, 12) for _ in range(2)]
        first = self._body_region_collator()
        first(batch)
        state = first.state_dict()
        expected = first(batch)[1:]
        restored = self._body_region_collator()
        restored.load_state_dict(state)
        actual = restored(batch)[1:]
        for expected_group, actual_group in zip(expected, actual):
            for expected_mask, actual_mask in zip(expected_group, actual_group):
                torch.testing.assert_close(actual_mask, expected_mask)

    def test_body_region_union_has_fixed_non_overlapping_coverage(self):
        collator = self._body_region_collator(num_regions=2)
        batch = [(torch.zeros(12, 366), 60, 12) for _ in range(4)]
        _, contexts, targets = collator(batch)
        context, target = contexts[0], targets[0]
        self.assertEqual(target.flatten(1).sum(1).tolist(), [8, 8, 8, 8])
        valid = torch.ones_like(target)
        self.assertFalse(bool((context & target).any()))
        torch.testing.assert_close(context | target, valid)

    def test_builder_selects_body_region_and_preserves_legacy_default(self):
        layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=12,
            token_num_frames=4,
            temporal_patch_size=3,
            raw_num_joints=30,
            token_num_joints=7,
        )
        body_config = {
            "patch": {
                "spatial_grouping": "coarse7",
                "spatial_pooling": "graph_mean",
            },
            "mask": {
                "strategy": "body_region_segment",
                "allow_overlap": False,
                "num_enc_masks": 1,
                "num_pred_masks": 1,
                "pred_frame_mask_ratio": [0.3, 0.6],
                "graph_mask_ratio": [2.0 / 7.0, 3.0 / 7.0],
            },
        }
        self.assertIsInstance(
            _build_mask_collator(body_config, layout),
            PatchBodyRegionSegmentMaskCollator2D,
        )

        legacy_config = {
            "patch": body_config["patch"],
            "mask": {
                "allow_overlap": False,
                "num_enc_masks": 1,
                "num_pred_masks": 1,
                "enc_frame_mask_ratio": [1.0, 1.0],
                "enc_joint_mask_ratio": [1.0, 1.0],
                "pred_frame_mask_ratio": [0.5, 0.5],
                "pred_joint_mask_ratio": [1.0 / 7.0, 1.0 / 7.0],
            },
        }
        self.assertIsInstance(
            _build_mask_collator(legacy_config, layout), PatchMaskCollator2D
        )

    def test_builder_rejects_invalid_body_region_mask_counts_and_overlap(self):
        layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=12,
            token_num_frames=4,
            temporal_patch_size=3,
            raw_num_joints=30,
            token_num_joints=7,
        )
        config = {
            "patch": {
                "spatial_grouping": "coarse7",
                "spatial_pooling": "graph_mean",
            },
            "mask": {
                "strategy": "body_region_segment",
                "allow_overlap": False,
                "num_enc_masks": 1,
                "num_pred_masks": 2,
                "pred_frame_mask_ratio": [0.3, 0.6],
                "graph_mask_ratio": [2.0 / 7.0, 3.0 / 7.0],
            },
        }
        with self.assertRaisesRegex(ValueError, "one encoder and one target"):
            _build_mask_collator(config, layout)
        config["mask"]["num_pred_masks"] = 1
        config["mask"]["allow_overlap"] = True
        with self.assertRaisesRegex(ValueError, "allow_overlap=false"):
            _build_mask_collator(config, layout)

    @staticmethod
    def _random_body_collator(**kwargs):
        options = {
            "raw_num_frames": 12,
            "raw_num_joints": 30,
            "token_num_joints": 7,
            "temporal_patch_size": 3,
            "spatial_grouping": "coarse7",
            "spatial_pooling": "graph_mean",
            "pred_frame_mask_ratio": (0.25, 0.25),
            "body_mask_ratio": (1.0 / 7.0, 3.0 / 7.0),
            "npred": 4,
        }
        options.update(kwargs)
        return PatchRandomBodySegmentMaskCollator2D(**options)

    def test_random_body_masks_are_separate_uniform_segments(self):
        collator = self._random_body_collator()
        batch = [(torch.zeros(12, 366), 60, 12) for _ in range(8)]
        _, contexts, targets = collator(batch)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(len(targets), 4)

        target_union = torch.stack(targets).any(dim=0)
        self.assertEqual(
            target_union.flatten(1).sum(1).tolist(),
            [sum(int(target[index].sum()) for target in targets) for index in range(8)],
        )
        target_counts = [target.flatten(1).sum(1) for target in targets]
        for target_count in target_counts[1:]:
            torch.testing.assert_close(target_count, target_counts[0])
        for target in targets:
            for sample in target:
                active_frames = torch.nonzero(sample.any(dim=1)).flatten()
                self.assertEqual(len(active_frames), 1)
                groups = sample[active_frames[0]]
                self.assertGreaterEqual(int(groups.sum()), 1)
                self.assertLessEqual(int(groups.sum()), 3)
        valid = torch.ones_like(contexts[0])
        self.assertFalse(bool((contexts[0] & target_union).any()))
        torch.testing.assert_close(contexts[0] | target_union, valid)

    def test_random_body_state_restores_next_masks(self):
        batch = [(torch.zeros(12, 366), 60, 12) for _ in range(2)]
        first = self._random_body_collator()
        first(batch)
        state = first.state_dict()
        expected = first(batch)[1:]
        restored = self._random_body_collator()
        restored.load_state_dict(state)
        actual = restored(batch)[1:]
        for expected_group, actual_group in zip(expected, actual):
            for expected_mask, actual_mask in zip(expected_group, actual_group):
                torch.testing.assert_close(actual_mask, expected_mask)

    def test_builder_selects_random_body_segment(self):
        layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=12,
            token_num_frames=4,
            temporal_patch_size=3,
            raw_num_joints=30,
            token_num_joints=7,
        )
        config = {
            "patch": {
                "spatial_grouping": "coarse7",
                "spatial_pooling": "graph_mean",
            },
            "mask": {
                "strategy": "random_body_segment",
                "allow_overlap": False,
                "num_enc_masks": 1,
                "num_pred_masks": 4,
                "pred_frame_mask_ratio": [0.15, 0.25],
                "body_mask_ratio": [1.0 / 7.0, 3.0 / 7.0],
            },
        }
        self.assertIsInstance(
            _build_mask_collator(config, layout),
            PatchRandomBodySegmentMaskCollator2D,
        )

    @staticmethod
    def _random_spatial_collator(**kwargs):
        options = {
            "raw_num_frames": 90,
            "raw_num_joints": 30,
            "token_num_joints": 8,
            "temporal_patch_size": 3,
            "spatial_grouping": "coarse7",
            "spatial_pooling": "graph_mean",
            "pred_frame_mask_ratio": (0.4, 0.6),
            "pred_spatial_mask_count": 4,
            "target_union_ratio": (0.55, 0.65),
            "npred": 4,
        }
        options.update(kwargs)
        return PatchRandomSpatialSegmentMaskCollator2D(**options)

    def test_random_spatial_masks_use_integer_count_overlap_and_union_bounds(self):
        collator = self._random_spatial_collator()
        batch = [(torch.zeros(90, 366), 30, 90) for _ in range(16)]
        _, contexts, targets = collator(batch)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(len(targets), 4)
        target_count = int(targets[0][0].sum())
        self.assertGreaterEqual(target_count, 12 * 4)
        self.assertLessEqual(target_count, 18 * 4)
        for target in targets:
            self.assertEqual(target.flatten(1).sum(1).tolist(), [target_count] * 16)
            for sample in target:
                active_frames = torch.nonzero(sample.any(dim=1)).flatten()
                self.assertTrue(
                    bool((active_frames[1:] - active_frames[:-1] == 1).all())
                )
                self.assertEqual(int(sample[active_frames[0]].sum()), 4)
        union = torch.stack(targets).any(dim=0)
        union_ratios = union.flatten(1).float().mean(dim=1)
        self.assertTrue(bool(((union_ratios >= 0.55) & (union_ratios <= 0.65)).all()))
        self.assertFalse(bool((contexts[0] & union).any()))
        torch.testing.assert_close(contexts[0] | union, torch.ones_like(union))
        self.assertTrue(bool((torch.stack(targets).sum(dim=0) > 1).any()))

    def test_random_spatial_masks_can_select_trajectory_and_restore_state(self):
        batch = [(torch.zeros(90, 366), 30, 90) for _ in range(8)]
        first = self._random_spatial_collator()
        saw_trajectory = False
        for _ in range(8):
            _, _, targets = first(batch)
            saw_trajectory |= any(bool(target[..., 0].any()) for target in targets)
        self.assertTrue(saw_trajectory)
        state = first.state_dict()
        expected = first(batch)[1:]
        restored = self._random_spatial_collator()
        restored.load_state_dict(state)
        actual = restored(batch)[1:]
        for expected_group, actual_group in zip(expected, actual):
            for expected_mask, actual_mask in zip(expected_group, actual_group):
                torch.testing.assert_close(actual_mask, expected_mask)

    def test_builder_selects_random_spatial_segment(self):
        layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=90,
            token_num_frames=30,
            temporal_patch_size=3,
            raw_num_joints=30,
            token_num_joints=8,
        )
        config = {
            "patch": {
                "spatial_grouping": "coarse7",
                "spatial_pooling": "graph_mean",
            },
            "mask": {
                "strategy": "random_spatial_segment",
                "num_enc_masks": 1,
                "num_pred_masks": 4,
                "pred_frame_mask_ratio": [0.4, 0.6],
                "pred_spatial_mask_count": 4,
                "target_union_ratio": [0.55, 0.65],
                "allow_target_overlap": True,
                "allow_context_target_overlap": False,
            },
        }
        self.assertIsInstance(
            _build_mask_collator(config, layout),
            PatchRandomSpatialSegmentMaskCollator2D,
        )


if __name__ == "__main__":
    unittest.main()
