"""Variable-cardinality 1D context/target padding must not change JEPA outputs."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from mask.utils import apply_index_masks, index_mask_validity, repeat_mask_blocks
from model import (
    MotionPatchTransformer1D,
    MotionPatchTransformerPredictor1D,
    MotionTransformer1D,
    MotionTransformerPredictor1D,
)


class IndexMaskPaddingTest(unittest.TestCase):
    def test_gather_zeros_padding_and_preserves_block_order_and_gradients(self):
        x = torch.arange(12.0).reshape(2, 3, 2).requires_grad_()
        masks = [torch.tensor([[2, -1], [0, 1]]), torch.tensor([[1, 0], [2, -1]])]
        output = apply_index_masks(x, masks)
        self.assertEqual(index_mask_validity(masks).tolist(), [
            [True, False], [True, True], [True, True], [True, False]
        ])
        torch.testing.assert_close(output, torch.tensor([
            [[4., 5.], [0., 0.]], [[6., 7.], [8., 9.]],
            [[2., 3.], [0., 1.]], [[10., 11.], [0., 0.]],
        ]))
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_rejects_invalid_sentinel_shapes_and_bounds(self):
        x = torch.zeros(2, 3, 4)
        malformed = [
            [torch.tensor([[0, -2], [0, 1]])],
            [torch.tensor([[0, 3], [0, 1]])],
            [torch.tensor([[0, 1]])],
            [torch.tensor([0, 1])],
            [torch.tensor([[0., 1.], [0., 1.]])],
            [torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long)],
            [torch.empty(2, 0, dtype=torch.long)],
            [],
        ]
        for masks in malformed:
            with self.subTest(masks=masks), self.assertRaises(ValueError):
                apply_index_masks(x, masks)


class Padded1DModelTest(unittest.TestCase):
    position_encoding = "absolute"

    def setUp(self):
        torch.manual_seed(31)
        self.motion = torch.randn(2, 18, 6)
        self.fps = torch.tensor([30, 60])
        self.valid = torch.arange(18).unsqueeze(0) < torch.tensor([[18], [10]])

    def _setup(self, patchified):
        if patchified:
            encoder = MotionPatchTransformer1D(
                6, 18, temporal_patch_size=3, embed_dim=12, depth=2, num_heads=3,
                position_encoding=self.position_encoding,
            )
            predictor = MotionPatchTransformerPredictor1D(
                18, 3, 12, 12, depth=2, num_heads=3,
                position_encoding=self.position_encoding,
            )
            contexts = [
                torch.tensor([[0, 2, 4], [0, -1, -1]]),
                torch.tensor([[1, 3, 5], [1, -1, -1]]),
            ]
            targets = [
                torch.tensor([[1, 3], [2, -1]]),
                torch.tensor([[0, 5], [0, 2]]),
            ]
        else:
            encoder = MotionTransformer1D(
                6, 18, embed_dim=12, depth=2, num_heads=3,
                position_encoding=self.position_encoding,
            )
            predictor = MotionTransformerPredictor1D(
                18, 12, 12, depth=2, num_heads=3,
                position_encoding=self.position_encoding,
            )
            contexts = [
                torch.tensor([[0, 2, 4, 6], [0, 2, -1, -1]]),
                torch.tensor([[1, 3, 5, 7], [1, 3, 5, -1]]),
            ]
            targets = [
                torch.tensor([[8, 9, 10], [6, 7, -1]]),
                torch.tensor([[11, 12, 13], [8, -1, -1]]),
            ]
        return encoder.eval(), predictor.eval(), contexts, targets

    def test_padded_batch_matches_individual_unpadded_calls_with_multiple_masks(self):
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                encoder, predictor, contexts, targets = self._setup(patchified)
                with torch.no_grad():
                    encoded = encoder(self.motion, self.fps, contexts, self.valid)
                    predicted = predictor(encoded, self.fps, contexts, targets)
                    for enc_id, context_mask in enumerate(contexts):
                        for sample in range(2):
                            row = enc_id * 2 + sample
                            context = context_mask[sample]
                            context = context[context >= 0].unsqueeze(0)
                            individual = encoder(
                                self.motion[sample:sample + 1], self.fps[sample:sample + 1],
                                [context], self.valid[sample:sample + 1],
                            )
                            torch.testing.assert_close(
                                encoded[row, index_mask_validity(contexts)[row]], individual[0],
                                atol=2e-6, rtol=1e-5,
                            )
                            for pred_id, target_mask in enumerate(targets):
                                target = target_mask[sample]
                                target = target[target >= 0].unsqueeze(0)
                                single_prediction = predictor(
                                    individual, self.fps[sample:sample + 1], [context], [target]
                                )
                                prediction_row = pred_id * len(contexts) * 2 + row
                                torch.testing.assert_close(
                                    predicted[prediction_row, :target.shape[1]], single_prediction[0],
                                    atol=2e-6, rtol=1e-5,
                                )
                    context_active = index_mask_validity(contexts)
                    target_active = repeat_mask_blocks(index_mask_validity(targets), 2, 2)
                    self.assertEqual(torch.count_nonzero(encoded[~context_active]).item(), 0)
                    self.assertEqual(torch.count_nonzero(predicted[~target_active]).item(), 0)

    def test_padding_values_do_not_change_valid_outputs_and_valid_frames_optional(self):
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                encoder, predictor, contexts, targets = self._setup(patchified)
                changed_motion = self.motion.clone()
                changed_motion[1, 10:] = torch.randn_like(changed_motion[1, 10:]) * 10000
                with torch.no_grad():
                    original = encoder(self.motion, self.fps, contexts, self.valid)
                    changed = encoder(changed_motion, self.fps, contexts, self.valid)
                    torch.testing.assert_close(original, changed)
                    torch.testing.assert_close(original, encoder(self.motion, self.fps, contexts))
                    changed_context = original.clone()
                    changed_context[~index_mask_validity(contexts)] = float('nan')
                    torch.testing.assert_close(
                        predictor(original, self.fps, contexts, targets),
                        predictor(changed_context, self.fps, contexts, targets),
                    )

    def test_rejects_empty_context_target_and_selection_of_padded_tokens(self):
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                encoder, predictor, contexts, targets = self._setup(patchified)
                bad_context = [contexts[0].clone()]
                bad_context[0][1] = -1
                with self.assertRaisesRegex(ValueError, 'valid token'):
                    encoder(self.motion, self.fps, bad_context, self.valid)
                bad_context[0][1, 0] = 3 if patchified else 10
                with self.assertRaisesRegex(ValueError, 'padded'):
                    encoder(self.motion, self.fps, bad_context, self.valid)
                encoded = encoder(self.motion, self.fps, contexts, self.valid)
                bad_target = [targets[0].clone()]
                bad_target[0][1] = -1
                with self.assertRaisesRegex(ValueError, 'valid token'):
                    predictor(encoded, self.fps, contexts, bad_target)
                with self.assertRaisesRegex(ValueError, 'token shape'):
                    predictor(encoded[:, :-1], self.fps, contexts, targets)

    def test_valid_target_loss_has_no_padding_or_teacher_gradients(self):
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                encoder, predictor, contexts, targets = self._setup(patchified)
                teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
                motion = self.motion.clone().requires_grad_()
                with torch.no_grad():
                    target = apply_index_masks(teacher(motion, self.fps, valid_frames=self.valid), targets)
                    target = repeat_mask_blocks(target, 2, len(contexts))
                encoded = encoder(motion, self.fps, contexts, self.valid)
                encoded.retain_grad()
                predicted = predictor(encoded, self.fps, contexts, targets)
                active = repeat_mask_blocks(index_mask_validity(targets), 2, len(contexts))
                F.smooth_l1_loss(predicted[active], target[active]).backward()
                self.assertTrue(any(p.grad is not None for p in encoder.parameters()))
                self.assertTrue(any(p.grad is not None for p in predictor.parameters()))
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertEqual(torch.count_nonzero(motion.grad[1, 10:]).item(), 0)
                self.assertEqual(torch.count_nonzero(encoded.grad[~index_mask_validity(contexts)]).item(), 0)
                self.assertTrue(torch.isfinite(motion.grad).all())


class RoPEPadded1DModelTest(Padded1DModelTest):
    """Run the same padding, multi-mask ordering and gradient contract with RoPE."""

    position_encoding = "rope"


if __name__ == '__main__':
    unittest.main()
