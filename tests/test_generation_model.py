"""Flow conditioning, null CFG, masked Euler sampling, and stable sample seeds."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from experiment.generation.model import FlowConfig, MotionFlow
from experiment.generation.sampling import sample_motion, seed_for_sample, seeded_noise, seeded_times
from model.token_layout import TokenLayout


class _ConstantFlow(nn.Module):
    def __init__(self, velocity: torch.Tensor, layout: TokenLayout):
        super().__init__()
        self.register_buffer("velocity", velocity)
        self.token_layout = layout
        self.motion_dim = velocity.shape[-1]
        self.calls = []

    def forward(self, noisy_motion, time, tokens, fps, valid_frames,
                token_mask=None, condition_drop=None):
        self.calls.append((time.detach().clone(), condition_drop.detach().clone(), self.training))
        return self.velocity.expand_as(noisy_motion)


class _GuidedConstantFlow(_ConstantFlow):
    def forward(self, noisy_motion, time, tokens, fps, valid_frames,
                token_mask=None, condition_drop=None):
        self.calls.append((time.detach().clone(), condition_drop.detach().clone(), self.training))
        value = torch.where(condition_drop, -1.0, 2.0)
        return value[:, None, None].expand_as(noisy_motion)


class MotionFlowTest(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.config = FlowConfig(hidden_dim=24, depth=2, num_heads=3, ffn_dim=48, dropout=0.0)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    @staticmethod
    def _layout(kind="1d", patchified=True):
        patch = 3 if patchified else 1
        joint_args = {"raw_num_joints": 30, "token_num_joints": 12 if patchified else 30} if kind == "2d" else {}
        return TokenLayout(kind=kind, patchified=patchified, raw_num_frames=8,
                           token_num_frames=8 // patch, temporal_patch_size=patch, **joint_args)

    @staticmethod
    def _tokens(layout, batch=2):
        shape = (batch, layout.token_num_frames)
        if layout.kind == "2d":
            shape += (int(layout.token_num_joints),)
        return torch.randn(*shape, 12)

    def test_four_layouts_have_finite_velocity_and_partial_tail_gradients(self):
        for kind in ("1d", "2d"):
            for patchified in (False, True):
                with self.subTest(kind=kind, patchified=patchified):
                    layout = self._layout(kind, patchified)
                    model = MotionFlow(12, layout, config=self.config)
                    valid = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
                    noisy = torch.randn(2, 8, 366, requires_grad=True)
                    output = model(noisy, torch.tensor([0.0, 1.0]), self._tokens(layout),
                                   torch.tensor([30, 60]), valid)
                    self.assertEqual(output.shape, (2, 8, 366))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(torch.count_nonzero(output[~valid]).item(), 0)
                    output[0, 6:].square().sum().backward()
                    self.assertGreater(float(noisy.grad[0, 6:].abs().sum()), 0)
                    self.assertEqual(torch.count_nonzero(noisy.grad[1, 5:]).item(), 0)

    def test_padding_invariance_for_motion_tokens_and_spatial_masks(self):
        for kind in ("1d", "2d"):
            with self.subTest(kind=kind):
                layout = self._layout(kind)
                model = MotionFlow(12, layout, config=self.config).eval()
                tokens, noisy = self._tokens(layout), torch.randn(2, 8, 366)
                valid = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
                token_active = layout.valid_token_mask(valid)
                if kind == "2d":
                    token_active = token_active[..., None].expand(tokens.shape[:-1]).clone()
                    token_active[:, :, 0] = False
                changed_tokens = tokens.masked_fill(~token_active[..., None], float("inf"))
                changed_motion = noisy.masked_fill(~valid[..., None], float("nan"))
                with torch.no_grad():
                    expected = model(noisy, torch.tensor([.2, .8]), tokens, torch.tensor([30, 30]),
                                     valid, token_mask=token_active)
                    actual = model(changed_motion, torch.tensor([.2, .8]), changed_tokens,
                                   torch.tensor([30, 30]), valid, token_mask=token_active)
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_whole_condition_dropout_prevents_content_leak(self):
        for kind in ("1d", "2d"):
            with self.subTest(kind=kind):
                layout = self._layout(kind)
                model = MotionFlow(12, layout, config=self.config).eval()
                tokens, noisy = self._tokens(layout), torch.randn(2, 8, 366)
                valid = torch.ones(2, 8, dtype=torch.bool)
                drop = torch.tensor([True, False])
                changed = tokens.clone()
                changed[0] = float("nan")
                with torch.no_grad():
                    original = model(noisy, torch.tensor([.2, .8]), tokens, torch.tensor([30, 30]), valid,
                                     condition_drop=drop)
                    altered = model(noisy, torch.tensor([.2, .8]), changed, torch.tensor([30, 30]), valid,
                                    condition_drop=drop)
                torch.testing.assert_close(original, altered, atol=0, rtol=0)
                model(noisy, torch.tensor([.2, .8]), changed, torch.tensor([30, 30]), valid,
                      condition_drop=drop).square().mean().backward()
                self.assertGreater(float(model.null_memory.grad.abs().sum()), 0)

    def test_null_memory_handles_missing_patches_without_all_padding_attention(self):
        layout = self._layout()
        model = MotionFlow(12, layout, config=self.config).eval()
        valid = torch.arange(8)[None].expand(2, -1) < 2
        args = (torch.randn(2, 8, 366), torch.tensor([.5, .5]), self._tokens(layout),
                torch.tensor([30, 30]), valid)
        with self.assertRaisesRegex(ValueError, "conditioned sample"):
            model(*args)
        output = model(*args, condition_drop=torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(torch.count_nonzero(output[~valid]).item(), 0)
        with self.assertRaisesRegex(ValueError, "valid raw frame"):
            model(*args[:-1], torch.zeros_like(valid), condition_drop=torch.ones(2, dtype=torch.bool))

    def test_flow_time_conditioning_is_separate_and_checked(self):
        layout = self._layout()
        model = MotionFlow(12, layout, config=self.config).eval()
        noisy, tokens, valid = torch.randn(2, 8, 366), self._tokens(layout), torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            early = model(noisy, torch.zeros(2), tokens, torch.full((2,), 30), valid)
            late = model(noisy, torch.ones(2), tokens, torch.full((2,), 30), valid)
        self.assertGreater(float((early - late).abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "flow time"):
            model(noisy, torch.tensor([0.0, 1.1]), tokens, torch.full((2,), 30), valid)

    def test_seeded_sampling_is_deterministic_restores_mode_and_decodes_tail(self):
        layout = self._layout("2d")
        config = FlowConfig(**(vars(self.config) | {"dropout": .2}))
        model = MotionFlow(12, layout, config=config).train()
        tokens = self._tokens(layout)
        valid = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
        noise = torch.stack([seeded_noise(["a", "b"], 17, draw, (8, 366), "cpu") for draw in range(2)], dim=1)
        altered_noise = noise.masked_fill(~valid[:, None, :, None], float("nan"))
        output = model.sample(tokens, torch.tensor([30, 30]), valid, num_samples=2,
                              steps=3, initial_noise=altered_noise, use_bfloat16=False)
        repeated = model.sample(tokens, torch.tensor([30, 30]), valid, num_samples=2,
                                steps=3, initial_noise=noise, use_bfloat16=False)
        self.assertTrue(model.training)
        self.assertEqual(output.shape, (2, 2, 8, 366))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, repeated, atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(output.masked_select(~valid[:, None, :, None])).item(), 0)
        self.assertGreater(float(output[0, :, 6:].abs().sum()), 0)
        self.assertGreater(float((output[:, 0] - output[:, 1]).abs().sum()), 0)
        bad_noise = noise.clone()
        bad_noise[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Initial noise"):
            model.sample(tokens, torch.tensor([30, 30]), valid, num_samples=2, initial_noise=bad_noise)

    def test_model_state_roundtrip(self):
        layout = self._layout("1d")
        model = MotionFlow(12, layout, config=self.config).eval()
        restored = MotionFlow(12, layout, config=FlowConfig(**vars(self.config))).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        tokens, noisy = self._tokens(layout), torch.randn(2, 8, 366)
        valid = torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(
                model(noisy, torch.full((2,), .5), tokens, torch.full((2,), 30), valid),
                restored(noisy, torch.full((2,), .5), tokens, torch.full((2,), 30), valid),
                atol=0, rtol=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bfloat16_network_with_fp32_euler_state(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        layout = self._layout("2d")
        model = MotionFlow(12, layout, config=self.config).cuda()
        valid = (torch.arange(8)[None] < torch.tensor([8, 5])[:, None]).cuda()
        tokens = self._tokens(layout).cuda().bfloat16()
        noisy = torch.randn(2, 8, 366, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(noisy, torch.tensor([.2, .8], device="cuda"), tokens,
                           torch.tensor([30, 30], device="cuda"), valid,
                           condition_drop=torch.tensor([True, False], device="cuda"))
            loss = output.float().square().mean()
        loss.backward()
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(model.motion_projection.weight.grad).all())
        sampled = model.sample(tokens, torch.tensor([30, 30], device="cuda"), valid,
                               steps=3, initial_noise=noisy, guidance_scale=1.5)
        self.assertEqual(sampled.dtype, torch.float32)
        self.assertTrue(torch.isfinite(sampled).all())


class EulerAndSeedTest(unittest.TestCase):
    def _layout(self):
        return TokenLayout(kind="1d", patchified=False, raw_num_frames=5, token_num_frames=5)

    def test_oracle_constant_velocity_euler_reaches_endpoint(self):
        layout = self._layout()
        noise, target = torch.randn(1, 5, 6), torch.randn(1, 5, 6)
        valid = torch.tensor([[True, True, True, False, False]])
        model = _ConstantFlow(target - noise, layout).train()
        result = sample_motion(model, torch.zeros(1, 5, 3), torch.tensor([30]), valid,
                               steps=7, initial_noise=noise, use_bfloat16=False)
        torch.testing.assert_close(result[:, 0, :3], target[:, :3])
        self.assertEqual(torch.count_nonzero(result[:, :, 3:]).item(), 0)
        self.assertTrue(model.training)
        self.assertTrue(all(not training for _, _, training in model.calls))
        self.assertEqual(float(model.calls[0][0]), 0.0)
        self.assertLess(float(model.calls[-1][0]), 1.0)

    def test_cfg_scale_zero_one_and_extrapolation(self):
        valid = torch.ones(1, 5, dtype=torch.bool)
        for scale, endpoint, expected_drop in ((0.0, -1.0, True), (1.0, 2.0, False), (3.0, 8.0, None)):
            with self.subTest(scale=scale):
                model = _GuidedConstantFlow(torch.zeros(1, 5, 6), self._layout()).eval()
                result = sample_motion(model, torch.zeros(1, 5, 3), torch.tensor([30]), valid,
                                       steps=4, guidance_scale=scale,
                                       initial_noise=torch.zeros(1, 5, 6), use_bfloat16=False)
                torch.testing.assert_close(result, torch.full_like(result, endpoint))
                self.assertFalse(model.training)
                self.assertEqual(len(model.calls), 8 if expected_drop is None else 4)
                if expected_drop is not None:
                    self.assertTrue(all(bool(drop[0]) == expected_drop for _, drop, _ in model.calls))

    def test_per_sample_seeds_are_batch_independent_and_do_not_touch_global_rng(self):
        before = torch.get_rng_state().clone()
        batch = seeded_noise(["left", "right"], 42, 0, (5, 6), "cpu")
        alone = seeded_noise(["right"], 42, 0, (5, 6), "cpu")
        torch.testing.assert_close(batch[1], alone[0], atol=0, rtol=0)
        times = seeded_times(["left", "right"], 42, "cpu")
        torch.testing.assert_close(times[1], seeded_times(["right"], 42, "cpu")[0], atol=0, rtol=0)
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
        self.assertTrue(((times >= 0) & (times < 1)).all())
        self.assertNotEqual(seed_for_sample(42, "left"), seed_for_sample(42, "left", 1))
        self.assertNotEqual(seed_for_sample(42, "left"), seed_for_sample(42, "left", stream="other"))
        self.assertTrue(0 <= seed_for_sample(42, "left") < (1 << 63))

    def test_sampling_validation_and_mode_restore_on_model_error(self):
        model = _ConstantFlow(torch.full((1, 5, 6), float("nan")), self._layout()).train()
        args = (model, torch.zeros(1, 5, 3), torch.tensor([30]), torch.ones(1, 5, dtype=torch.bool))
        with self.assertRaisesRegex(FloatingPointError, "velocity"):
            sample_motion(*args, steps=2)
        self.assertTrue(model.training)
        for changes in ({"steps": 0}, {"num_samples": 0}, {"guidance_scale": float("inf")}, {"guidance_scale": -1}):
            with self.assertRaises(ValueError):
                sample_motion(*args, **changes)


if __name__ == "__main__":
    unittest.main()
