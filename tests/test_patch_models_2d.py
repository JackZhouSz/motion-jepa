"""Tests for temporal and anatomical 2D patch models."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from experiment.linear_probe.features import load_frozen_encoder, pool_encoder_output
from experiment.linear_probe.train_classifier import _extract_token_features
from mask import PatchMaskCollator2D
from mask.utils import gather_grid_masks, repeat_mask_blocks
from model import (
    MotionFeatureTokenizer2D,
    MotionPatchTransformer2D,
    MotionPatchTransformerPredictor2D,
    get_spatial_grouping,
    mot_patch_tiny_2d,
)
from model.modules import (
    AxialBlock2D,
    DropPath,
    PredictorAxialBlock2D,
    prepare_packed_axial_layout,
)


class PatchModel2DTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.motion = torch.randn(2, 8, 366)
        self.fps = torch.full((2,), 60)

    @staticmethod
    def _encoder(grouping: str = "fine11") -> MotionPatchTransformer2D:
        return MotionPatchTransformer2D(
            366,
            8,
            30,
            temporal_patch_size=3,
            spatial_grouping=grouping,
            embed_dim=24,
            depth=2,
            num_heads=3,
        )

    def test_group_presets_cover_soma30_once(self):
        for name, expected in (("joint30", 30), ("fine11", 11), ("coarse7", 7)):
            groups = get_spatial_grouping(name)
            self.assertEqual(len(groups), expected)
            joints = [joint for _, indices in groups for joint in indices]
            self.assertEqual(sorted(joints), list(range(30)))
            self.assertEqual(len(joints), len(set(joints)))

    def test_prepared_attention_masks_make_empty_rows_safe(self):
        active = torch.zeros((2, 3, 4), dtype=torch.bool)
        active[0, 1, 2] = True
        temporal, spatial = AxialBlock2D.prepare_attention_masks(active)
        self.assertTrue(temporal.any(dim=1).all())
        self.assertTrue(spatial.any(dim=1).all())

        streams = torch.stack([active, torch.zeros_like(active)], dim=1)
        temporal, spatial = PredictorAxialBlock2D.prepare_attention_masks(streams)
        self.assertTrue(temporal.any(dim=1).all())
        self.assertTrue(spatial.any(dim=1).all())

    def test_prepared_attention_masks_match_per_block_preparation(self):
        block = AxialBlock2D(24, 3).eval()
        active = torch.zeros((2, 3, 4), dtype=torch.bool)
        active[0, :2, :3] = True
        active[1, 1:, 1:] = True
        values = torch.randn(2, 3, 4, 24)
        prepared = block.prepare_attention_masks(active)
        with torch.no_grad():
            direct = block(values, active)
            reused = block(values, active, prepared)
        torch.testing.assert_close(direct, reused)

    def test_packed_layout_round_trip_and_target_order(self):
        active = torch.zeros((2, 2, 3, 4), dtype=torch.bool)
        active[0, 0, :2, :3] = True
        active[0, 1, 1:, 2:] = True
        active[1, 0, 1:, 1:] = True
        active[1, 1, :2, :2] = True
        layout = prepare_packed_axial_layout(active)
        canonical = torch.arange(len(layout.dense_indices))
        temporal = canonical.index_select(0, layout.temporal_order)
        spatial = temporal.index_select(0, layout.spatial_from_temporal)
        torch.testing.assert_close(
            spatial.index_select(0, layout.temporal_from_spatial), temporal
        )
        torch.testing.assert_close(
            temporal.index_select(0, layout.canonical_from_temporal), canonical
        )
        expected_targets = canonical[
            (layout.dense_indices % (2 * 3 * 4)) // (3 * 4) == 1
        ]
        torch.testing.assert_close(
            temporal.index_select(0, layout.target_from_temporal), expected_targets
        )
        self.assertEqual(layout.spatial_padded_seqlen, 8)
        self.assertEqual(layout.spatial_padded_mask.shape, (6, 8))

    def test_disjoint_predictor_layout_uses_joint_width_spatial_padding(self):
        active = torch.zeros((2, 2, 3, 4), dtype=torch.bool)
        active[:, 0, :, :3] = True
        active[:, 1, :, 3:] = True
        layout = prepare_packed_axial_layout(active, spatial_padded_seqlen=4)
        self.assertEqual(layout.spatial_padded_seqlen, 4)
        self.assertEqual(layout.spatial_padded_mask.shape, (6, 4))
        self.assertLess(int(layout.spatial_padded_indices.max()), 6 * 4)

    def test_packed_drop_path_preserves_logical_batch_masks(self):
        active = torch.zeros((3, 2, 4), dtype=torch.bool)
        active[0, :, :2] = True
        active[1, 1:, 1:] = True
        active[2, :, 2:] = True
        layout = prepare_packed_axial_layout(active)
        dense_values = torch.ones(3, 2, 4, 5)
        packed_values = dense_values.reshape(-1, 5).index_select(
            0, layout.dense_indices
        ).index_select(0, layout.temporal_order)
        dense_drop = DropPath(0.4).train()
        packed_drop = DropPath(0.4).train()
        torch.manual_seed(19)
        dense_output = dense_drop(dense_values)
        torch.manual_seed(19)
        packed_output = packed_drop.forward_packed(
            packed_values, layout.temporal_batch_ids, layout.logical_batch_size
        )
        expected = dense_output.reshape(-1, 5).index_select(
            0, layout.dense_indices
        ).index_select(0, layout.temporal_order)
        torch.testing.assert_close(packed_output, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA varlen attention required")
    def test_persistent_packed_block_supports_drop_path(self):
        dense = PredictorAxialBlock2D(48, 3, drop_path=0.2).cuda().train()
        packed = copy.deepcopy(dense)
        active = torch.zeros((2, 2, 3, 4), device="cuda", dtype=torch.bool)
        active[0, 0, :2, :3] = True
        active[0, 1, 1:, 2:] = True
        active[1, 0, 1:, 1:] = True
        active[1, 1, :2, :2] = True
        layout = prepare_packed_axial_layout(active)
        dense_input = torch.randn(
            2, 2, 3, 4, 48, device="cuda", requires_grad=True
        )
        packed_input = dense_input.detach().reshape(-1, 48).index_select(
            0, layout.dense_indices
        ).index_select(0, layout.temporal_order).clone().requires_grad_(True)
        masks = dense.prepare_attention_masks(active)
        torch.cuda.manual_seed_all(29)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            dense_output = dense(dense_input, active, masks)
        torch.cuda.manual_seed_all(29)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            packed_output = packed.forward_packed(packed_input, layout)
        expected = dense_output.reshape(-1, 48).index_select(
            0, layout.dense_indices
        ).index_select(0, layout.temporal_order)
        torch.testing.assert_close(expected, packed_output, rtol=3.0e-2, atol=1.0e-2)
        expected.float().square().mean().backward()
        packed_output.float().square().mean().backward()
        for dense_parameter, packed_parameter in zip(
            dense.parameters(), packed.parameters()
        ):
            torch.testing.assert_close(
                dense_parameter.grad, packed_parameter.grad,
                rtol=5.0e-2, atol=5.0e-4,
            )

    def test_active_mlp_matches_dense_output_and_gradients(self):
        dense = PredictorAxialBlock2D(12, 3).eval()
        gathered = copy.deepcopy(dense)
        active = torch.zeros((2, 2, 3, 4), dtype=torch.bool)
        active[0, 0, :2, :3] = True
        active[0, 1, 1:, 2:] = True
        active[1, 0, 1:, 1:] = True
        active[1, 1, :2, :2] = True
        dense_input = torch.randn(2, 2, 3, 4, 12, requires_grad=True)
        gathered_input = dense_input.detach().clone().requires_grad_(True)
        masks = dense.prepare_attention_masks(active)
        indices = active.flatten().nonzero(as_tuple=False).flatten()

        dense_output = dense(dense_input, active, masks)
        gathered_output = gathered(gathered_input, active, masks, indices)
        torch.testing.assert_close(dense_output, gathered_output)
        dense_output.square().sum().backward()
        gathered_output.square().sum().backward()
        torch.testing.assert_close(dense_input.grad, gathered_input.grad)
        for dense_parameter, gathered_parameter in zip(
            dense.parameters(), gathered.parameters()
        ):
            torch.testing.assert_close(dense_parameter.grad, gathered_parameter.grad)

    def test_active_mlp_falls_back_to_dense_when_dropout_is_configured(self):
        block = PredictorAxialBlock2D(12, 3, drop=0.1).eval()
        active = torch.zeros((2, 2, 3, 4), dtype=torch.bool)
        active[:, :, :2, :3] = True
        values = torch.randn(2, 2, 3, 4, 12)
        masks = block.prepare_attention_masks(active)
        indices = active.flatten().nonzero(as_tuple=False).flatten()

        with torch.no_grad():
            dense = block(values, active, masks)
            with_gather_indices = block(values, active, masks, indices)
        torch.testing.assert_close(dense, with_gather_indices)

    def test_shape_tail_and_padding_invariance(self):
        valid = torch.arange(8).unsqueeze(0) < torch.tensor([[8], [5]])
        changed = self.motion.clone()
        changed[1, 5:] = torch.randn_like(changed[1, 5:]) * 1000
        for grouping, groups in (("fine11", 12), ("coarse7", 8)):
            encoder = self._encoder(grouping).eval()
            with torch.no_grad():
                output = encoder(self.motion, self.fps, valid_frames=valid)
                modified = encoder(changed, self.fps, valid_frames=valid)
            self.assertEqual(output.shape, (2, 2, groups, 24))
            torch.testing.assert_close(output[1, 0], modified[1, 0])
            self.assertEqual(torch.count_nonzero(output[1, 1]).item(), 0)

    def test_graph_has_no_cross_group_edges(self):
        for grouping in ("fine11", "coarse7"):
            encoder = self._encoder(grouping)
            group_by_joint = {
                joint: group
                for group, (_, joints) in enumerate(encoder.groups)
                for joint in joints
            }
            adjacency = encoder.patch_embed.adjacency
            rows, columns = torch.nonzero(adjacency, as_tuple=True)
            for row, column in zip(rows.tolist(), columns.tolist()):
                self.assertEqual(group_by_joint[row], group_by_joint[column])

    def test_target_time_and_group_values_cannot_leak_into_other_patches(self):
        encoder = self._encoder("fine11").eval()
        with torch.no_grad():
            original = encoder.patch_embed(self.motion)

        temporal_change = self.motion.clone()
        temporal_change[:, 3:6] = torch.randn_like(temporal_change[:, 3:6]) * 1000
        with torch.no_grad():
            temporal_output = encoder.patch_embed(temporal_change)
        torch.testing.assert_close(original[:, 0], temporal_output[:, 0])

        tokenizer = MotionFeatureTokenizer2D(embed_dim=1)
        feature_ids = torch.arange(366, dtype=torch.float32).reshape(1, 1, 366)
        routed = tokenizer.split_features(feature_ids)
        left_upper_arm = dict(get_spatial_grouping("fine11"))["left_upper_arm"]
        indices = torch.cat([routed[joint].flatten() for joint in left_upper_arm]).long()
        spatial_change = self.motion.clone()
        spatial_change[:, :3, indices] = torch.randn_like(
            spatial_change[:, :3, indices]
        ) * 1000
        with torch.no_grad():
            spatial_output = encoder.patch_embed(spatial_change)
        target_group = 1 + [name for name, _ in encoder.groups].index("left_upper_arm")
        visible = [index for index in range(12) if index != target_group]
        torch.testing.assert_close(original[:, 0, visible], spatial_output[:, 0, visible])
        context_mask = torch.ones((2, 2, 12), dtype=torch.bool)
        context_mask[:, 0, target_group] = False
        with torch.no_grad():
            context_original = encoder(self.motion, self.fps, [context_mask])
            context_modified = encoder(spatial_change, self.fps, [context_mask])
        torch.testing.assert_close(context_original, context_modified)

    def test_multi_mask_forward_backward_and_stem_gradients(self):
        batch = [(self.motion[index], 60, 8) for index in range(2)]
        _, masks_enc, masks_pred = PatchMaskCollator2D(
            raw_num_frames=8,
            raw_num_joints=30,
            token_num_joints=8,
            temporal_patch_size=3,
            spatial_grouping="coarse7",
            enc_frame_mask_ratio=(1.0, 1.0),
            enc_joint_mask_ratio=(1.0, 1.0),
            pred_frame_mask_ratio=(0.5, 0.5),
            pred_joint_mask_ratio=(0.25, 0.25),
            nenc=2,
            npred=2,
            allow_overlap=True,
        )(batch)
        encoder = self._encoder("coarse7")
        predictor = MotionPatchTransformerPredictor2D(
            8,
            30,
            3,
            "coarse7",
            embed_dim=24,
            predictor_embed_dim=12,
            depth=2,
            num_heads=3,
        )
        target = encoder(self.motion, self.fps)
        context = encoder(self.motion, self.fps, masks_enc)
        prediction = predictor(context, self.fps, masks_enc, masks_pred)
        expected = repeat_mask_blocks(
            gather_grid_masks(target, masks_pred), len(self.motion), len(masks_enc)
        )
        self.assertEqual(prediction.shape, expected.shape)
        loss = F.smooth_l1_loss(prediction, expected.detach())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(encoder.patch_embed.temporal_conv.groups, 30)
        self.assertEqual(tuple(encoder.patch_embed.joint_feature_indices.shape), (30, 13))
        self.assertIsNotNone(encoder.patch_embed.temporal_conv.weight.grad)
        self.assertIsNotNone(encoder.patch_embed.trajectory_conv.weight.grad)
        self.assertIsNotNone(encoder.patch_embed.graph_projection.weight.grad)

    def test_trajectory_and_body_routing_are_disjoint_and_semantic(self):
        encoder = self._encoder("coarse7")
        stem = encoder.patch_embed
        trajectory = set(stem.trajectory_feature_indices.tolist())
        body = {
            index
            for row, width in zip(stem.joint_feature_indices.tolist(), stem.body_input_dims)
            for index in row[:width]
        }
        self.assertFalse(trajectory & body)
        self.assertEqual(trajectory | body, set(range(366)))
        self.assertEqual(trajectory, {0, 2, 3, 4})
        self.assertIn(1, body)

        with torch.no_grad():
            original = stem(self.motion)
            trajectory_changed = self.motion.clone()
            trajectory_changed[:, :3, [0, 2, 3, 4]] += 100.0
            trajectory_output = stem(trajectory_changed)
            torch.testing.assert_close(original[:, 0, 1:], trajectory_output[:, 0, 1:])

            pelvis_changed = self.motion.clone()
            pelvis_changed[:, :3, [1, *range(92, 98), *range(272, 275)]] += 100.0
            pelvis_output = stem(pelvis_changed)
            torch.testing.assert_close(original[:, 0, 0], pelvis_output[:, 0, 0])
            torch.testing.assert_close(original[:, 0, 2:], pelvis_output[:, 0, 2:])

        parameter_names = dict(encoder.named_parameters())
        self.assertFalse(any("type_embed" in name for name in parameter_names))
        self.assertEqual(encoder.positions.group.shape[2], 8)

    def test_frozen_mean_and_temporal_token_probe_support(self):
        encoder = mot_patch_tiny_2d(
            num_frames=8, temporal_patch_size=3, spatial_grouping="coarse7"
        )
        config = {
            "data": {
                "num_frames": 8,
                "motion_dim": 366,
                "num_joints": 30,
                "fps": 60,
            },
            "patch": {
                "temporal_patch_size": 3,
                "spatial_grouping": "coarse7",
                "spatial_pooling": "graph_mean",
            },
            "meta": {
                "model_name": "mot_patch_tiny_2d",
                "predictor_name": "mot_predictor_patch_tiny_2d",
                "use_bfloat16": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch-2d.pth.tar"
            torch.save(
                {"format_version": 1, "target_encoder": encoder.state_dict(), "config": config},
                path,
            )
            loaded, _, info = load_frozen_encoder(
                path, "target_encoder", torch.device("cpu")
            )
        self.assertEqual(info["token_num_joints"], 8)
        self.assertEqual(info["spatial_grouping"], "coarse7")
        pooled = pool_encoder_output(
            torch.ones(1, 2, 8, 4), torch.tensor([5]), loaded.token_layout
        )
        torch.testing.assert_close(pooled, torch.ones(1, 4))
        sample = (self.motion[0], torch.tensor(60), torch.tensor(5), torch.tensor(0), "x")
        payload = _extract_token_features(
            loaded,
            [sample],
            device=torch.device("cpu"),
            batch_size=1,
            num_workers=0,
        )
        self.assertEqual(payload["features"].shape, (1, 2, 192))
        torch.testing.assert_close(payload["lengths"], torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
