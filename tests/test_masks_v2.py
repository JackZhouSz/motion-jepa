"""Variable-length 1D masking preserves geometry and uses explicit padding."""

from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import torch

from mask import MaskCollator1D, MaskCollator1DV2, PatchMaskCollator1DV2


class MaskCollatorV2Test(unittest.TestCase):
    @staticmethod
    def batch(lengths, frames=150):
        return [(torch.zeros(frames, 6), 30, length) for length in lengths]

    @staticmethod
    def indices(block, row):
        return block[row][block[row] >= 0]

    def test_own_target_lengths_and_padding_without_batch_truncation(self):
        collator = MaskCollator1DV2(
            150,
            enc_frame_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(0.2, 0.2),
            nenc=2,
            npred=4,
        )
        collated, contexts, targets = collator(self.batch([60, 150]))
        torch.testing.assert_close(collated[2], torch.tensor([60, 150]))
        self.assertEqual({tuple(block.shape) for block in targets}, {(2, 30)})
        self.assertEqual(len({tuple(block.shape) for block in contexts}), 1)
        for target in targets:
            self.assertEqual(len(self.indices(target, 0)), 12)
            self.assertEqual(len(self.indices(target, 1)), 30)
            self.assertTrue(bool((target[0, 12:] == -1).all()))
        for row, length in enumerate((60, 150)):
            union = torch.cat([self.indices(target, row) for target in targets]).unique()
            expected = torch.arange(length)[~torch.isin(torch.arange(length), union)]
            for context in contexts:
                indices = self.indices(context, row)
                # Full context complement is kept, including its later frames.
                torch.testing.assert_close(indices, expected)
                self.assertGreaterEqual(len(indices), math.ceil(length * 0.2))
                self.assertTrue(bool((indices[1:] > indices[:-1]).all()))
                self.assertFalse(bool(torch.isin(indices, union).any()))
                self.assertTrue(bool((context[row, len(indices):] == -1).all()))

    def test_changing_other_lengths_does_not_change_sample_geometry(self):
        kwargs = dict(num_frames=150, nenc=2, npred=4)
        for earlier_length in (30, 60, 150):
            first = MaskCollator1DV2(**kwargs)
            second = MaskCollator1DV2(**kwargs)
            expected = first(self.batch([60, 150, 60]))[1:]
            actual = second(self.batch([earlier_length, 150, 30]))[1:]
            for expected_group, actual_group in zip(expected, actual):
                for expected_block, actual_block in zip(expected_group, actual_group):
                    torch.testing.assert_close(
                        self.indices(expected_block, 1), self.indices(actual_block, 1)
                    )

    def test_short_patch_length_excludes_incomplete_patch_and_keeps_raw_lengths(self):
        collator = PatchMaskCollator1DV2(
            150,
            3,
            enc_frame_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(0.2, 0.2),
        )
        collated, contexts, targets = collator(self.batch([60, 62, 150]))
        torch.testing.assert_close(collated[2], torch.tensor([60, 62, 150]))
        for row, valid_tokens in enumerate((20, 20, 50)):
            for block in contexts + targets:
                selected = self.indices(block, row)
                self.assertTrue(bool((selected < valid_tokens).all()))
                self.assertTrue(bool((selected[1:] > selected[:-1]).all()))
            self.assertEqual(len(self.indices(targets[0], row)), round(valid_tokens * 0.2))

    def test_context_guard_is_respected_over_many_variable_batches(self):
        collator = PatchMaskCollator1DV2(150, 3, nenc=2)
        lengths = [60, 61, 90, 120, 149, 150]
        for _ in range(24):
            _, contexts, targets = collator(self.batch(lengths))
            for row, length in enumerate(lengths):
                valid_tokens = length // 3
                union = torch.cat([self.indices(target, row) for target in targets])
                for context in contexts:
                    indices = self.indices(context, row)
                    self.assertGreaterEqual(len(indices), math.ceil(valid_tokens * 0.2))
                    self.assertFalse(bool(torch.isin(indices, union).any()))

    def test_context_selection_preserves_targets_and_uses_guarded_minimum(self):
        for patchified in (False, True):
            constructor = PatchMaskCollator1DV2 if patchified else MaskCollator1DV2
            kwargs = {"raw_num_frames": 150, "temporal_patch_size": 3} if patchified else {
                "num_frames": 150
            }
            kwargs.update(nenc=2, npred=4)
            lengths = [60, 62, 90, 150]
            collators = {
                mode: constructor(**kwargs, context_selection=mode)
                for mode in ("all", "prefix", "random")
            }
            found_later_random_token = False
            for _ in range(16):
                results = {
                    mode: collator(self.batch(lengths))
                    for mode, collator in collators.items()
                }
                all_contexts, all_targets = results["all"][1:]
                minimum = min(
                    len(self.indices(block, row))
                    for block in all_contexts
                    for row in range(len(lengths))
                )
                for mode in ("prefix", "random"):
                    selected_contexts, selected_targets = results[mode][1:]
                    for target, expected in zip(selected_targets, all_targets):
                        torch.testing.assert_close(target, expected)
                    self.assertEqual(len({tuple(block.shape) for block in selected_contexts}), 1)
                    for row, raw_length in enumerate(lengths):
                        valid_length = raw_length // 3 if patchified else raw_length
                        required = max(minimum, math.ceil(valid_length * 0.2))
                        target_union = torch.cat([
                            self.indices(target, row) for target in selected_targets
                        ])
                        for selected_block, candidate_block in zip(selected_contexts, all_contexts):
                            candidates = self.indices(candidate_block, row)
                            selected = self.indices(selected_block, row)
                            self.assertEqual(len(selected), required)
                            self.assertTrue(bool(torch.isin(selected, candidates).all()))
                            self.assertTrue(bool((selected[1:] > selected[:-1]).all()))
                            self.assertTrue(bool((selected < valid_length).all()))
                            self.assertFalse(bool(torch.isin(selected, target_union).any()))
                            self.assertTrue(bool((selected_block[row, len(selected):] == -1).all()))
                            if mode == "prefix":
                                torch.testing.assert_close(selected, candidates[:required])
                            else:
                                found_later_random_token |= bool(
                                    torch.isin(selected, candidates[required:]).any()
                                )
            self.assertTrue(found_later_random_token)

    def test_random_selection_reaches_all_candidates_without_prefix_bias(self):
        collator = MaskCollator1DV2(10, npred=1, context_selection="random")
        candidates = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9])
        short_candidates = torch.tensor([2, 3, 4])
        counts = torch.zeros(10, dtype=torch.long)
        for _ in range(512):
            geometry = [
                ([candidates], [torch.tensor([2, 3])]),
                ([short_candidates], [torch.tensor([0, 1])]),
            ]
            with patch.object(collator, "_sample", side_effect=geometry):
                _, contexts, targets = collator(self.batch([10, 10], frames=10))
            selected = self.indices(contexts[0], 0)
            self.assertEqual(len(selected), 3)
            self.assertTrue(bool((selected[1:] > selected[:-1]).all()))
            torch.testing.assert_close(self.indices(contexts[0], 1), short_candidates)
            torch.testing.assert_close(targets[0][0], torch.tensor([2, 3]))
            counts += torch.bincount(selected, minlength=10)
        expected = 512 * 3 / len(candidates)
        self.assertLess(float((counts[candidates] - expected).abs().max()), 50)
        self.assertEqual(int(counts[[2, 3]].sum()), 0)

    def test_selected_contexts_restore_next_masks_and_preserve_global_rng(self):
        batch = self.batch([60, 62, 90, 150])
        for constructor, kwargs in (
            (MaskCollator1DV2, {"num_frames": 150}),
            (PatchMaskCollator1DV2, {"raw_num_frames": 150, "temporal_patch_size": 3}),
        ):
            for mode in ("prefix", "random"):
                first = constructor(**kwargs, nenc=2, context_selection=mode)
                rng_before = torch.get_rng_state().clone()
                first(batch)
                state = copy.deepcopy(first.state_dict())
                expected = first(batch)[1:]
                restored = constructor(**kwargs, nenc=2, context_selection=mode)
                restored.load_state_dict(state)
                actual = restored(batch)[1:]
                torch.testing.assert_close(torch.get_rng_state(), rng_before)
                for expected_group, actual_group in zip(expected, actual):
                    for expected_block, actual_block in zip(expected_group, actual_group):
                        torch.testing.assert_close(actual_block, expected_block)
                other_mode = "prefix" if mode == "random" else "random"
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    constructor(**kwargs, nenc=2, context_selection=other_mode).load_state_dict(state)

    def test_old_v2_state_migrates_to_all_selection_only(self):
        batch = self.batch([60, 90, 150])
        for constructor, kwargs in (
            (MaskCollator1DV2, {"num_frames": 150}),
            (PatchMaskCollator1DV2, {"raw_num_frames": 150, "temporal_patch_size": 3}),
        ):
            first = constructor(**kwargs)
            first(batch)
            legacy = copy.deepcopy(first.state_dict())
            legacy["configuration"].pop("context_selection")
            expected = first(batch)[1:]
            restored = constructor(**kwargs, context_selection="all")
            restored.load_state_dict(legacy)
            for expected_group, actual_group in zip(expected, restored(batch)[1:]):
                for expected_block, actual_block in zip(expected_group, actual_group):
                    torch.testing.assert_close(actual_block, expected_block)
            self.assertNotIn("context_selection", legacy["configuration"])
            for mode in ("prefix", "random"):
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    constructor(**kwargs, context_selection=mode).load_state_dict(legacy)

    def test_overlap_allowed_does_not_subtract_targets(self):
        collator = MaskCollator1DV2(
            60,
            enc_frame_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(1.0, 1.0),
            allow_overlap=True,
        )
        _, contexts, targets = collator(self.batch([30, 60], frames=60))
        for row, length in enumerate((30, 60)):
            expected = torch.arange(length)
            for block in contexts + targets:
                torch.testing.assert_close(self.indices(block, row), expected)

    def test_exhausted_sampling_fails_without_context_leakage_fallback(self):
        collator = MaskCollator1DV2(
            10,
            enc_frame_mask_ratio=(0.8, 0.8),
            pred_frame_mask_ratio=(0.5, 0.5),
            npred=2,
        )
        # Feasible in principle, but these target draws cover every token.
        draws = [torch.arange(5), torch.arange(5, 10), torch.arange(8)] * 128
        with patch.object(collator, "_interval", side_effect=draws) as interval:
            with self.assertRaisesRegex(ValueError, "after 128 attempts"):
                collator(self.batch([10], frames=10))
        self.assertEqual(interval.call_count, 384)

    def test_frame_samples_without_explicit_lengths_use_full_length(self):
        collator = MaskCollator1DV2(60)
        batch = [(torch.zeros(60, 6), 30) for _ in range(2)]
        _, contexts, targets = collator(batch)
        self.assertTrue(all(block.shape[0] == 2 for block in contexts + targets))

    def test_state_restores_next_masks_and_sampling_preserves_global_rng(self):
        for constructor, kwargs in (
            (MaskCollator1DV2, {"num_frames": 150}),
            (PatchMaskCollator1DV2, {"raw_num_frames": 150, "temporal_patch_size": 3}),
        ):
            first = constructor(**kwargs)
            batch = self.batch([60, 90, 150])
            rng_before = torch.get_rng_state().clone()
            first(batch)
            torch.testing.assert_close(torch.get_rng_state(), rng_before)
            state = copy.deepcopy(first.state_dict())
            expected = first(batch)[1:]
            restored = constructor(**kwargs)
            restored.load_state_dict(state)
            actual = restored(batch)[1:]
            for expected_group, actual_group in zip(expected, actual):
                for expected_block, actual_block in zip(expected_group, actual_group):
                    torch.testing.assert_close(actual_block, expected_block)

    def test_state_rejects_v1_or_missing_version_and_config_changes(self):
        v2 = MaskCollator1DV2(150)
        for old_state in (MaskCollator1D(150).state_dict(), {"counter": 2}):
            with self.assertRaisesRegex(ValueError, "matching V2"):
                v2.load_state_dict(old_state)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            MaskCollator1DV2(150, min_context_ratio=0.3).load_state_dict(v2.state_dict())
        with self.assertRaisesRegex(ValueError, "matching V2"):
            PatchMaskCollator1DV2(150, 3).load_state_dict(v2.state_dict())

    def test_configuration_and_invalid_lengths_fail_clearly(self):
        for kwargs in (
            {"enc_frame_mask_ratio": (0.4, 1.1)},
            {"pred_frame_mask_ratio": (0.4,)},
            {"enc_frame_mask_ratio": (float("nan"), 1.0)},
            {"npred": 0},
            {"min_context_tokens": 0},
            {"min_context_ratio": 1.1},
            {"pred_frame_mask_ratio": (1.0, 1.0)},
            {"min_context_ratio": 0.9, "pred_frame_mask_ratio": (0.2, 0.2)},
            {"context_selection": "typo"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MaskCollator1DV2(150, **kwargs)
        collator = MaskCollator1DV2(150, min_context_tokens=40)
        with self.assertRaisesRegex(ValueError, "minimum context"):
            collator(self.batch([30]))
        for length in (0, 151):
            with self.assertRaisesRegex(ValueError, "Valid lengths"):
                MaskCollator1DV2(150)(self.batch([length]))
        with self.assertRaisesRegex(ValueError, "complete temporal patch"):
            PatchMaskCollator1DV2(150, 3)(self.batch([2]))
        with self.assertRaisesRegex(ValueError, "positive"):
            PatchMaskCollator1DV2(150, 0)
        with self.assertRaisesRegex(ValueError, "nonempty batch"):
            MaskCollator1DV2(150)([])

    def test_worker_batches_share_counter_and_keep_all_valid_indices(self):
        collator = PatchMaskCollator1DV2(150, 3)
        loader = torch.utils.data.DataLoader(
            self.batch([60, 150, 90, 150]),
            batch_size=2,
            num_workers=2,
            collate_fn=collator,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(collator.state_dict()["counter"], 1)
        for collated, contexts, targets in batches:
            for row, raw_length in enumerate(collated[2].tolist()):
                for block in contexts + targets:
                    self.assertTrue(bool((self.indices(block, row) < raw_length // 3).all()))


if __name__ == "__main__":
    unittest.main()
