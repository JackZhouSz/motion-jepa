"""Context trimming modes preserve mask geometry and exact resume."""
import copy
import unittest
from unittest.mock import patch

import torch

from mask import MaskCollator1D, PatchMaskCollator1D
from model import TokenLayout
from train import _build_mask_collator


class ContextSelectionTest(unittest.TestCase):
    def test_random_is_uniform_over_candidates_and_leaves_targets_intact(self):
        def active(indices):
            result = torch.zeros(10, dtype=torch.bool)
            result[indices] = True
            return result

        # First sample: 8 available context tokens; second: 3. Common K=3.
        intervals = [active([2, 3]), active(list(range(10))),
                     active([0, 1]), active(list(range(5)))]
        candidates = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9])
        batch = [(torch.zeros(10, 1), 30, 10)] * 2
        prefix = MaskCollator1D(10, npred=1, context_selection='prefix')
        random = MaskCollator1D(10, npred=1, context_selection='random')
        counts = torch.zeros(10, dtype=torch.long)
        for _ in range(512):
            with patch.object(prefix, '_interval', side_effect=[x.clone() for x in intervals]):
                _, enc_prefix, pred_prefix = prefix(batch)
            with patch.object(random, '_interval', side_effect=[x.clone() for x in intervals]):
                _, enc_random, pred_random = random(batch)
            torch.testing.assert_close(pred_random[0], pred_prefix[0])
            torch.testing.assert_close(enc_prefix[0][0], candidates[:3])
            torch.testing.assert_close(enc_random[0][1], torch.tensor([2, 3, 4]))
            selected = enc_random[0][0]
            self.assertTrue(torch.isin(selected, candidates).all())
            self.assertTrue((selected[1:] > selected[:-1]).all())
            counts += torch.bincount(selected, minlength=10)
        # Fixed seeds: each of the 8 candidates has inclusion probability 3/8.
        self.assertLess(float((counts[candidates] - 512*3/8).abs().max()), 50)
        self.assertEqual(int(counts[[2, 3]].sum()), 0)

    def test_patch_random_keeps_counts_targets_sorted_valid_and_disjoint(self):
        lengths = [150, 141, 105, 78] * 8
        batch = [(torch.zeros(150, 1), 30, n) for n in lengths]
        kwargs = dict(raw_num_frames=150, temporal_patch_size=3, nenc=2, npred=4)
        prefix = PatchMaskCollator1D(**kwargs)
        random = PatchMaskCollator1D(**kwargs, context_selection='random')
        changed = False
        for _ in range(8):
            _, contexts_p, targets_p = prefix(batch)
            _, contexts_r, targets_r = random(batch)
            for a, b in zip(targets_p, targets_r):
                torch.testing.assert_close(a, b)
            for a, b in zip(contexts_p, contexts_r):
                self.assertEqual(a.shape, b.shape)
                changed |= not torch.equal(a, b)
                self.assertTrue((b[:, 1:] > b[:, :-1]).all())
                for sample, length in enumerate(lengths):
                    self.assertTrue((b[sample] < length//3).all())
                    self.assertTrue((b[sample] >= 0).all())
                    target = torch.cat([t[sample] for t in targets_r])
                    self.assertFalse(torch.isin(b[sample], target).any())
        self.assertTrue(changed)

    def test_random_restores_next_masks_and_legacy_is_prefix(self):
        for patchified in (False, True):
            def make(mode):
                return (PatchMaskCollator1D(150, 3, context_selection=mode) if patchified
                        else MaskCollator1D(50, context_selection=mode))
            frames = 150 if patchified else 50
            batch = [(torch.zeros(frames, 1), 30, frames)] * 16
            for mode in ('prefix', 'random'):
                first = make(mode);first(batch)
                state = first.state_dict()
                expected = first(batch)[1:]
                restored = make(mode);restored.load_state_dict(state)
                actual = restored(batch)[1:]
                for a_group, b_group in zip(expected, actual):
                    for a, b in zip(a_group, b_group):torch.testing.assert_close(a, b)
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    make('random' if mode=='prefix' else 'prefix').load_state_dict(state)
            legacy = make('prefix');legacy(batch)
            saved = copy.deepcopy(legacy.state_dict())
            saved['configuration'].pop('context_selection')
            restored = make('prefix');restored.load_state_dict(saved)
            for a_group, b_group in zip(legacy(batch)[1:], restored(batch)[1:]):
                for a, b in zip(a_group, b_group):torch.testing.assert_close(a, b)
            self.assertNotIn('context_selection', saved['configuration'])
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                make('random').load_state_dict(saved)

    def test_config_wiring_and_invalid_modes(self):
        args = {'mask': dict(enc_frame_mask_ratio=[.85,1.], pred_frame_mask_ratio=[.15,.2],
                            num_enc_masks=1, num_pred_masks=4, allow_overlap=False)}
        for patchified in (False, True):
            layout = TokenLayout(kind='1d', patchified=patchified, raw_num_frames=150,
                                 token_num_frames=50 if patchified else 150,
                                 temporal_patch_size=3 if patchified else 1)
            self.assertEqual(_build_mask_collator(args, layout).context_selection, 'prefix')
            args['mask']['context_selection'] = 'random'
            self.assertEqual(_build_mask_collator(args, layout).context_selection, 'random')
            args['mask']['context_selection'] = 'typo'
            with self.assertRaisesRegex(ValueError, 'context_selection'):_build_mask_collator(args, layout)
            args['mask'].pop('context_selection')
        with self.assertRaisesRegex(ValueError, 'context_selection'):
            MaskCollator1D(50, context_selection='typo')
        args['mask']['context_selection'] = 'random'
        layout = TokenLayout(kind='2d', patchified=False, raw_num_frames=50, token_num_frames=50,
                             raw_num_joints=30, token_num_joints=30)
        with self.assertRaisesRegex(ValueError, 'requires a 1D'):_build_mask_collator(args, layout)


if __name__ == '__main__':unittest.main()
