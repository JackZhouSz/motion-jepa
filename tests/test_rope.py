"""RoPE must preserve motion time coordinates through sparse token selection."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from mask.utils import apply_index_masks
from model import (
    MotionPatchTransformer1D,
    MotionPatchTransformerPredictor1D,
    MotionTransformer1D,
    MotionTransformerPredictor1D,
)
from model.modules import SelfAttention
from model.pos_embs import (
    RotaryPosEmbed1D,
    apply_rotary_pos_emb,
    temporal_token_positions,
)


def _gather_tokens(tokens, indices):
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class RotaryPositionsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_frame_and_patch_centers_use_physical_time_with_mixed_fps(self):
        fps = torch.tensor([30.0, 60.0])
        frames = temporal_token_positions(11, fps)
        patches = temporal_token_positions(11, fps, temporal_patch_size=3)
        torch.testing.assert_close(frames, torch.arange(11)[None, :] / fps[:, None])
        torch.testing.assert_close(patches, torch.tensor([[1.0, 4.0, 7.0]]) / fps[:, None])
        self.assertEqual(patches.shape, (2, 3))
        self.assertEqual(patches.dtype, torch.float32)
        torch.testing.assert_close(frames[0, 1], frames[1, 2])

    def test_rotation_preserves_norm_and_uses_interleaved_coordinate_pairs(self):
        query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        key = -query
        rotary = RotaryPosEmbed1D(4, theta=100.0)(torch.tensor([[1.0]]))
        rotated_query, rotated_key = apply_rotary_pos_emb(query, key, rotary)
        angles = torch.tensor([1.0, 0.1])
        expected = torch.stack(
            [query[..., ::2] * angles.cos() - query[..., 1::2] * angles.sin(),
             query[..., ::2] * angles.sin() + query[..., 1::2] * angles.cos()],
            dim=-1,
        ).flatten(-2)
        torch.testing.assert_close(rotated_query, expected)
        torch.testing.assert_close(rotated_key, -expected)
        torch.testing.assert_close(rotated_query.norm(dim=-1), query.norm(dim=-1))

    def test_common_time_shift_preserves_attention_but_changing_gaps_does_not(self):
        query = torch.randn(2, 3, 4, 8)
        key = torch.randn_like(query)
        positions = torch.tensor([[0.0, 0.1, 0.6, 1.0], [0.0, 0.05, 0.3, 0.5]])
        embedding = RotaryPosEmbed1D(8, theta=100.0, time_scale=30.0)

        def scores(times):
            q, k = apply_rotary_pos_emb(query, key, embedding(times))
            return q @ k.transpose(-1, -2)

        original = scores(positions)
        translated = scores(positions + torch.tensor([[2.0], [3.0]]))
        torch.testing.assert_close(original, translated, atol=3e-5, rtol=1e-5)
        changed = positions.clone()
        changed[:, 2] += 0.2
        self.assertGreater((original - scores(changed)).abs().max().item(), 0.1)

    def test_time_scale_and_half_module_keep_float32_phases(self):
        positions = torch.tensor([[0.0, 0.125, 8.5]])
        scaled = RotaryPosEmbed1D(8, theta=100.0, time_scale=30.0).half()(positions)
        explicit = RotaryPosEmbed1D(8, theta=100.0)(positions * 30.0)
        for actual, expected in zip(scaled, explicit):
            self.assertEqual(actual.shape, (1, 1, 3, 4))
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected)
        query = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        rotated, _ = apply_rotary_pos_emb(query, query, scaled)
        self.assertEqual(rotated.dtype, query.dtype)
        self.assertTrue(torch.isfinite(rotated).all())

    def test_sparse_attention_matches_dense_masked_outputs_and_gradients(self):
        dense_attention = SelfAttention(16, 2).eval()
        sparse_attention = copy.deepcopy(dense_attention)
        dense_input = torch.randn(2, 7, 16, requires_grad=True)
        sparse_input = dense_input.detach().clone().requires_grad_()
        indices = torch.tensor([[0, 2, 6], [1, 3, 5]])
        active = torch.zeros(2, 7, dtype=torch.bool).scatter_(1, indices, True)
        positions = temporal_token_positions(7, torch.tensor([30.0, 60.0]))
        embedding = RotaryPosEmbed1D(8, time_scale=30.0)
        dense_output = _gather_tokens(
            dense_attention(dense_input, active, rotary=embedding(positions)), indices
        )
        sparse_output = sparse_attention(
            _gather_tokens(sparse_input, indices),
            rotary=embedding(positions.gather(1, indices)),
        )
        torch.testing.assert_close(dense_output, sparse_output, atol=2e-6, rtol=1e-5)
        weights = torch.randn_like(dense_output)
        (dense_output * weights).sum().backward()
        (sparse_output * weights).sum().backward()
        torch.testing.assert_close(dense_input.grad, sparse_input.grad, atol=2e-6, rtol=1e-5)
        for dense_parameter, sparse_parameter in zip(
            dense_attention.parameters(), sparse_attention.parameters()
        ):
            torch.testing.assert_close(
                dense_parameter.grad, sparse_parameter.grad, atol=3e-6, rtol=1e-5
            )
        self.assertEqual(torch.count_nonzero(sparse_input.grad[~active]).item(), 0)

        # Reindexing retained tokens to consecutive positions destroys original gaps.
        compressed_positions = temporal_token_positions(3, torch.tensor([30.0, 60.0]))
        compressed_output = sparse_attention(
            _gather_tokens(sparse_input.detach(), indices),
            rotary=embedding(compressed_positions),
        )
        self.assertGreater((sparse_output - compressed_output).abs().max().item(), 1e-4)


class RotaryModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.motion = torch.randn(2, 18, 6)
        self.fps = torch.tensor([30.0, 60.0])

    @staticmethod
    def _models(patchified, **position_kwargs):
        if patchified:
            encoder = MotionPatchTransformer1D(
                6, 18, temporal_patch_size=3, embed_dim=16, depth=2, num_heads=2,
                **position_kwargs,
            )
            predictor = MotionPatchTransformerPredictor1D(
                18, 3, 16, 12, depth=2, num_heads=3, **position_kwargs,
            )
        else:
            encoder = MotionTransformer1D(
                6, 18, embed_dim=16, depth=2, num_heads=2, **position_kwargs,
            )
            predictor = MotionTransformerPredictor1D(
                18, 16, 12, depth=2, num_heads=3, **position_kwargs,
            )
        return encoder.eval(), predictor.eval()

    def test_default_matches_explicit_absolute_and_existing_state_dict(self):
        contexts = [torch.tensor([[0, 2, 5], [1, 3, 4]])]
        targets = [torch.tensor([[1, 3], [0, 5]])]
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                default_encoder, default_predictor = self._models(patchified)
                absolute_encoder, absolute_predictor = self._models(
                    patchified, position_encoding="absolute"
                )
                absolute_encoder.load_state_dict(default_encoder.state_dict(), strict=True)
                absolute_predictor.load_state_dict(default_predictor.state_dict(), strict=True)
                with torch.no_grad():
                    default_context = default_encoder(self.motion, self.fps, contexts)
                    absolute_context = absolute_encoder(self.motion, self.fps, contexts)
                    torch.testing.assert_close(default_context, absolute_context, atol=0, rtol=0)
                    torch.testing.assert_close(
                        default_predictor(default_context, self.fps, contexts, targets),
                        absolute_predictor(absolute_context, self.fps, contexts, targets),
                        atol=0, rtol=0,
                    )

    def test_token_permutations_preserve_original_coordinates(self):
        contexts = [torch.tensor([[0, 2, 5], [1, 3, 4]])]
        targets = [torch.tensor([[1, 3], [0, 5]])]
        permutation = torch.tensor([2, 0, 1])
        permuted_contexts = [contexts[0][:, permutation]]
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                encoder, predictor = self._models(
                    patchified, position_encoding="rope", rope_time_scale=30.0
                )
                with torch.no_grad():
                    context = encoder(self.motion, self.fps, contexts)
                    permuted = encoder(self.motion, self.fps, permuted_contexts)
                    torch.testing.assert_close(permuted, context[:, permutation], atol=2e-6, rtol=1e-5)
                    prediction = predictor(context, self.fps, contexts, targets)
                    reordered = predictor(permuted, self.fps, permuted_contexts, targets)
                    torch.testing.assert_close(prediction, reordered, atol=2e-6, rtol=1e-5)
                    reversed_targets = predictor(context, self.fps, contexts, [targets[0].flip(1)])
                    torch.testing.assert_close(
                        prediction.flip(1), reversed_targets, atol=2e-6, rtol=1e-5
                    )

    def test_predictor_uses_target_time_and_common_origin_for_both_streams(self):
        contexts = [torch.tensor([[0, 2], [0, 2]])]
        targets = [torch.tensor([[1, 4], [1, 4]])]
        context_features = torch.randn(2, 2, 16)
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                _, predictor = self._models(
                    patchified, position_encoding="rope", rope_time_scale=30.0
                )
                with torch.no_grad():
                    original = predictor(context_features, self.fps, contexts, targets)
                    shifted = predictor(
                        context_features, self.fps, [contexts[0] + 1], [targets[0] + 1]
                    )
                    changed_targets = predictor(
                        context_features, self.fps, contexts, [targets[0] + 1]
                    )
                torch.testing.assert_close(original, shifted, atol=2e-6, rtol=1e-5)
                self.assertGreater((original - changed_targets).abs().max().item(), 1e-7)
                self.assertGreater((original[:, 0] - original[:, 1]).abs().max().item(), 1e-7)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 attention required",
    )
    def test_cuda_bfloat16_teacher_context_predictor_backward(self):
        contexts = [torch.tensor([[0, 2, 5], [0, 2, -1]], device="cuda")]
        targets = [torch.tensor([[1, 3], [1, -1]], device="cuda")]
        active_targets = targets[0] >= 0
        valid_frames = torch.arange(18, device="cuda")[None, :] < torch.tensor(
            [[18], [12]], device="cuda"
        )
        fps = self.fps.cuda()
        for patchified in (False, True):
            with self.subTest(patchified=patchified):
                encoder, predictor = self._models(
                    patchified, position_encoding="rope", rope_time_scale=30.0
                )
                encoder = encoder.cuda().train()
                predictor = predictor.cuda().train()
                teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
                motion = self.motion.cuda().requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        target = apply_index_masks(
                            teacher(motion, fps, valid_frames=valid_frames), targets
                        )
                    context = encoder(motion, fps, contexts, valid_frames)
                    predicted = predictor(context, fps, contexts, targets)
                    loss = F.smooth_l1_loss(predicted[active_targets], target[active_targets])
                self.assertEqual(predicted.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(torch.count_nonzero(predicted[~active_targets]).item(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(motion.grad).all())
                self.assertEqual(torch.count_nonzero(motion.grad[1, 12:]).item(), 0)
                for model in (encoder, predictor):
                    gradients = [parameter.grad for parameter in model.parameters()]
                    self.assertTrue(any(gradient is not None for gradient in gradients))
                    self.assertTrue(all(
                        gradient is None or torch.isfinite(gradient).all()
                        for gradient in gradients
                    ))
                self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
