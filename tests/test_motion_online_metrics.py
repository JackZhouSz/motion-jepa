"""Causality, candidate isolation, kinematics and immutable online evaluation."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from experiment.motion_online_metrics import (
    MotionOnlineMetrics, aligned_future, causal_past, positions_from_features,
    representation_metrics, retrieval_metrics, tensorboard_metrics, trajectory_errors,
)
from model import MotionPatchTransformer1D
from motion_rep.feet import foot_detect_from_pos_and_vel
from motion_rep.geometry import velocity, y_rotation
from motion_rep.reps.motion_jepa_motionrep import MotionJEPAMotionRep as Rep
from skeleton import SOMASkeleton30


def raw_motion(speed=.1, frames=12, fps=6):
    raw = torch.zeros(frames, 366)
    raw[:, 0] = torch.arange(frames) * speed / fps
    raw[:, 1] = 1
    raw[:, 3] = 1
    body = torch.zeros(frames, 29, 3)
    body[..., 0] = torch.linspace(-.3, .3, 29)
    body[..., 1] = .05
    raw[:, Rep.LOCAL_POSITIONS] = body.flatten(1)
    raw[:, Rep.GLOBAL_ROTATIONS] = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(30)
    return refresh_channels(raw, fps)


def refresh_channels(raw, fps):
    raw = raw.clone()
    positions = positions_from_features(raw[None])
    speeds = velocity(positions, fps)
    raw[:, Rep.VELOCITIES] = speeds[0].flatten(1)
    raw[:, Rep.FOOT_CONTACTS] = foot_detect_from_pos_and_vel(positions, speeds, SOMASkeleton30(), .15, .1)[0]
    return raw


def fixture(root):
    dataset = root / "data"
    rows = []
    for split, count in (("train", 4), ("val", 8)):
        motions = [raw_motion(.04 * (i + 1)).numpy() for i in range(count)]
        write_npy_dataset(dataset, motions, split=split, num_frames=12, fps=6)
        for i in range(count):
            rows.append(dict(id=f"sample-{i}", source_id=f"{split}-source-{i}", split=split,
                             motion_path=f"motions/{split}/sample-{i}.npy", length=12, fps=6,
                             metadata=dict(is_mirror="True" if i == count - 1 else "False", take_actor=f"actor-{i}")))
    (dataset / "index.json").write_text(json.dumps(rows))
    stats = dataset / "stats"; stats.mkdir()
    np.save(stats / "mean.npy", np.zeros(366, dtype=np.float32))
    np.save(stats / "std.npy", np.ones(366, dtype=np.float32))
    config = dict(data=dict(root_path=str(dataset), num_frames=12, fps=6, motion_dim=366,
                            normalize=True, stats_path="stats"),
                  patch=dict(temporal_patch_size=3), meta=dict(use_bfloat16=False),
                  logging=dict(folder=str(root / "output")))
    options = dict(num_queries=2, num_gallery=4, retrieval_gallery_size=4,
                   calibration_samples=3, past_frames=6, horizons_seconds=[.5, 1.], batch_size=3)
    return config, options


class MotionMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_future_and_derived_channels_cannot_change_causal_embedding(self):
        original = raw_motion(.04)
        changed = original.clone()
        changed[6:, 0] += 20
        changed = refresh_channels(changed, 6)
        self.assertFalse(torch.equal(original[5, Rep.VELOCITIES], changed[5, Rep.VELOCITIES]))
        self.assertFalse(torch.equal(original[5, Rep.FOOT_CONTACTS], changed[5, Rep.FOOT_CONTACTS]))
        a, b = causal_past(original[None], 6, 6), causal_past(changed[None], 6, 6)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        encoder = MotionPatchTransformer1D(366, 12, temporal_patch_size=3, embed_dim=12, depth=1, num_heads=3).eval()
        padded = torch.zeros(2, 12, 366); padded[0, :6] = a[0]; padded[1, :6] = b[0]
        padded[1, 6:] = torch.randn_like(padded[1, 6:]) * 1000
        active = torch.arange(12)[None].expand(2, -1) < 6
        with torch.inference_mode():
            tokens = encoder(padded, torch.full((2,), 6.), valid_frames=active)
        torch.testing.assert_close(tokens[0, :2], tokens[1, :2], rtol=0, atol=0)
        torch.testing.assert_close(a[0, 5, Rep.VELOCITIES], original[4, Rep.VELOCITIES])

    def test_contact_recomputed_at_current_height_not_copied(self):
        raw = raw_motion(.0)
        positions = raw[:, Rep.LOCAL_POSITIONS].reshape(12, 29, 3)
        # Cross the height threshold slowly enough to retain low velocity.
        positions[4, :, 1] = .09
        positions[5:, :, 1] = .11
        raw = refresh_channels(raw, 6)
        past = causal_past(raw[None], 6, 6)
        self.assertTrue((past[0, -2, Rep.FOOT_CONTACTS] == 1).all())
        self.assertTrue((past[0, -1, Rep.FOOT_CONTACTS] == 0).all())

    def test_yaw_translation_alignment_and_analytic_errors(self):
        query = raw_motion(.2)[None]
        donor = query.clone()
        angle = .7; rotation = y_rotation(torch.tensor(angle))
        translation = torch.tensor([4., .3, -2.])
        positions = positions_from_features(query) @ rotation.T + translation
        donor[..., Rep.ROOT_POSITION] = positions[:, :, 0]
        origin = positions[:, :, :1].clone(); origin[..., 1] = 0
        donor[..., Rep.LOCAL_POSITIONS] = (positions[:, :, 1:] - origin).flatten(-2)
        donor[..., Rep.ROOT_HEADING] = torch.tensor([math.cos(angle), math.sin(angle)])
        prediction = aligned_future(query, donor, 6)
        truth = positions_from_features(query[:, 6:])
        torch.testing.assert_close(prediction, truth, atol=1e-6, rtol=1e-5)
        shifted = truth.clone(); shifted[..., 0] += 1
        errors = trajectory_errors(shifted, truth, 6, [.5, 1.])
        for values in errors.values():
            self.assertAlmostEqual(values["root_ade_xz"], 1)
            self.assertAlmostEqual(values["root_fde_xz"], 1)
            self.assertAlmostEqual(values["root_relative_joint_error"], 0, places=6)

    def test_tie_safe_retrieval(self):
        scores = torch.zeros(2, 8)
        result = retrieval_metrics(scores, torch.tensor([0, 7]))
        self.assertAlmostEqual(result["recall_at_1"], 1/8)
        self.assertAlmostEqual(result["recall_at_5"], 5/8)
        self.assertAlmostEqual(result["mrr"], sum(1/i for i in range(1, 9))/8)
        result = retrieval_metrics(torch.tensor([[0., 2., 1.]]), torch.tensor([1]))
        self.assertEqual(result, dict(recall_at_1=1., recall_at_5=1., mrr=1.))

    def test_rank_and_tensorboard_whitelist(self):
        isotropic = torch.cat([torch.eye(4), -torch.eye(4)])
        diverse = representation_metrics(isotropic, .3, .11)
        collapsed = representation_metrics(torch.ones(8, 4), 0., 0.)
        self.assertAlmostEqual(diverse["raw"]["covariance"]["effective_rank"], 4.)
        self.assertEqual(collapsed["near_zero_channels"], 4)
        summary = dict(representation=diverse, retrieval=dict(recall_at_1=.5, num_queries=128),
                       future_transfer=dict(feature=dict(horizon_1s=dict(root_ade_xz=.2)),
                                            baselines=dict(constant_velocity=dict(horizon_1s=dict(root_ade_xz=.4)))),
                       metadata=dict(num_sources=640), elapsed_seconds=.2, pretrain_epoch=1)
        values = tensorboard_metrics(summary)
        self.assertIn("representation/raw/mean_std", values)
        self.assertEqual(values["representation/temporal_mean_variance"], .11)
        self.assertIn("representation/standardized/covariance/effective_rank", values)
        self.assertNotIn("representation/standardized/mean_std", values)
        self.assertNotIn("retrieval/num_queries", values)
        self.assertIn("future_transfer/feature/horizon_1s/root_ade_xz", values)
        self.assertFalse(any("baselines" in key for key in values))
        self.assertFalse(any("metadata" in key or "epoch" in key for key in values))

    def test_full_bank_rejects_short_clips_even_when_horizon_fits(self):
        evaluator = MotionOnlineMetrics.__new__(MotionOnlineMetrics)
        evaluator.num_frames, evaluator.fps = 12, 6
        evaluator.past_frames, evaluator.future_frames = 6, 3
        records = [dict(id=str(length), source_id=str(length), split="val", length=length,
                        fps=6, metadata=dict(is_mirror=False, take_actor="actor")) for length in (9, 12)]
        selected = evaluator._select_sources(records, "val", 1, np.random.default_rng(42))
        self.assertEqual([record["length"] for record in selected], [12])
        with self.assertRaisesRegex(ValueError, "found 1"):
            evaluator._select_sources(records, "val", 2, np.random.default_rng(42))

    def test_temporal_variance_is_mean_variance_not_squared_mean_std(self):
        class FixedEncoder(torch.nn.Module):
            token_layout = SimpleNamespace(valid_token_lengths=lambda lengths: lengths // 3)

            def forward(self, motion, fps, valid_frames):
                tokens = torch.tensor([[0., 0.], [2., 4.], [999., 999.], [999., 999.]])
                return tokens[None].expand(len(motion), -1, -1)

        evaluator = MotionOnlineMetrics.__new__(MotionOnlineMetrics)
        evaluator.num_frames, evaluator.fps, evaluator.batch_size = 12, 6, 2
        evaluator.device, evaluator.use_bfloat16 = torch.device("cpu"), False
        evaluator.mean, evaluator.std = torch.zeros(366), torch.ones(366)
        _, std, variance = evaluator._encode(FixedEncoder(), torch.zeros(1, 6, 366), torch.tensor([6]))
        self.assertAlmostEqual(std, 1.5)
        self.assertAlmostEqual(variance, 2.5)
        self.assertNotEqual(variance, std ** 2)

    def test_banks_artifacts_repeatability_and_stale_data(self):
        with tempfile.TemporaryDirectory() as directory:
            config, options = fixture(Path(directory))
            evaluator = MotionOnlineMetrics(config, options, device=torch.device("cpu"))
            sources = [row["source_id"] for row in evaluator.records]
            self.assertEqual(len(sources), len(set(sources)))
            self.assertTrue(set(sources[:2]).isdisjoint(sources[2:]))
            self.assertEqual(evaluator.retrieval_indices[:2], [0, 1])
            self.assertTrue(all(row["metadata"]["is_mirror"] == "False" for row in evaluator.records))
            encoder = MotionPatchTransformer1D(366, 12, temporal_patch_size=3, embed_dim=12, depth=1, num_heads=3).eval().requires_grad_(False)
            before = {key: value.clone() for key, value in encoder.state_dict().items()}
            a = evaluator.evaluate(encoder)
            replay = MotionOnlineMetrics(config, options, device=torch.device("cpu"))
            b = replay.evaluate(encoder)
            for key in ("retrieval", "future_transfer", "representation", "protocol_hash", "metadata"):
                self.assertEqual(a[key], b[key])
            for key, value in encoder.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
            self.assertTrue((Path(config["logging"]["folder"])/"online_metrics/baselines.pt").exists())
            row = evaluator.records[0]
            path = Path(config["data"]["root_path"]) / row["motion_path"]
            raw = np.load(path); raw[0, 0] += 1; np.save(path, raw)
            with self.assertRaisesRegex(ValueError, "protocol/data changed"):
                MotionOnlineMetrics(config, options, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
