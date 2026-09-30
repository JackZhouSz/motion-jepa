"""Frame readout, masking, metric ties, cache isolation and exact head resume."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from experiment.segmentation_probe import (
    FrameLinearProbe, FrameMetricAccumulator, OnlineSegmentationProbe,
    binary_average_precision, complete_patch_frame_mask,
)
from experiment.segmentation_probe.online import _cache_digest, fit_token_standardizer, train_head_epoch
from model import MotionPatchTransformer1D, TokenLayout


class _Dataset(Dataset):
    class_names = tuple(f"class-{i}" for i in range(120))
    class_indices_60 = tuple(range(60))

    def __init__(self, root, split, **kwargs):
        self.split = split
        self.count = 4 if split == "train" else 3

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        length = (8, 7, 5, 8)[index]
        generator = np.random.default_rng(index + (10 if self.split == "val" else 0))
        motion = generator.normal(size=(8, 6)).astype(np.float32)
        motion[length:] = 0
        labels = np.zeros((8, 120), dtype=np.float32)
        labels[:length, (0 if index % 2 == 0 else 70)] = 1
        labels[1] = 0  # Context retained, supervision ignored.
        if index == 0:
            labels[2, 70] = 1  # A multilabel frame.
        return motion, 30, length, labels, labels.any(-1), f"{self.split}-{index}"


def _config(root):
    dataset = root / "data"
    dataset.mkdir(exist_ok=True)
    (dataset / "meta.json").write_text(json.dumps({"fixture": True}))
    stats = root / "stats"
    stats.mkdir(exist_ok=True)
    np.save(stats / "mean.npy", np.zeros(6, dtype=np.float32))
    np.save(stats / "std.npy", np.ones(6, dtype=np.float32))
    return {
        "data": {"root_path": str(root), "stats_path": "stats", "num_frames": 8,
                 "fps": 30, "motion_dim": 6},
        "meta": {"use_bfloat16": False, "model_name": "fixture"},
        "patch": {"temporal_patch_size": 2},
        "logging": {"folder": str(root / "output")},
    }, {"dataset_root": str(dataset), "epochs": 3, "batch_size": 2,
        "feature_batch_size": 2, "num_workers": 0, "seed": 42}


class SegmentationProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_phase_rows_produce_distinct_ordered_frame_logits(self):
        head = FrameLinearProbe(1, 3, 2)
        with torch.no_grad():
            head.linear.weight.copy_(torch.arange(1, 7).float()[:, None])
            head.linear.bias.zero_()
        result = head(torch.tensor([[[1.0], [10.0]]]))
        torch.testing.assert_close(result, torch.tensor(
            [[[1., 2.], [3., 4.], [5., 6.], [10., 20.], [30., 40.], [50., 60.]]]))

    def test_complete_patch_mask_excludes_real_tail_frames(self):
        layout = TokenLayout("1d", True, 8, 2, 3)
        result = complete_patch_frame_mask(torch.tensor([8, 5]), layout)
        self.assertEqual(result.tolist(), [[True] * 6, [True] * 3 + [False] * 3])

    def test_standardization_uses_all_valid_train_tokens_not_label_mask(self):
        cache = {
            "features": torch.tensor([[[1., 10.], [3., 20.], [1000., 1000.]],
                                       [[5., 30.], [2000., 2000.], [3000., 3000.]]]),
            "token_valid": torch.tensor([[True, True, False], [True, False, False]]),
            "supervised": torch.zeros(2, 6, dtype=torch.bool),
        }
        normalizer = fit_token_standardizer(cache, batch_size=1)
        torch.testing.assert_close(normalizer["mean"], torch.tensor([3., 20.]))
        torch.testing.assert_close(normalizer["scale"], torch.tensor([[1., 10.], [3., 20.], [5., 30.]]).std(0, correction=0))
        self.assertEqual(normalizer["fit_tokens"], 3)

    def test_ties_are_grouped_and_absent_classes_are_excluded(self):
        scores = torch.zeros(2)
        self.assertEqual(binary_average_precision(scores, torch.tensor([1, 0])), 0.5)
        self.assertEqual(binary_average_precision(scores, torch.tensor([0, 1])), 0.5)
        self.assertIsNone(binary_average_precision(scores, torch.tensor([0, 0])))
        # Keep a pure outside-subset positive as a negative for subset class 0.
        labels = torch.zeros(1, 3, 120)
        labels[0, 0, 70] = labels[0, 1, 0] = labels[0, 2, 70] = 1
        logits = torch.zeros_like(labels)
        mask = torch.ones(1, 3, dtype=torch.bool)
        whole = FrameMetricAccumulator(120, range(60))
        whole.update(logits, labels, mask)
        expected = whole.compute()
        self.assertAlmostEqual(expected["babel-60"]["frame_map"], 1 / 3)
        self.assertEqual(expected["babel-60"]["supervised_frames"], 3)
        self.assertEqual(expected["babel-120"]["classes_with_positives"], 2)
        divided = FrameMetricAccumulator(120, range(60))
        for index in (2, 1, 0):
            divided.update(logits[:, index:index + 1], labels[:, index:index + 1], mask[:, index:index + 1])
        self.assertEqual(divided.compute(), expected)

    def test_masked_logits_do_not_change_metrics(self):
        logits = torch.zeros(1, 2, 2)
        labels = torch.tensor([[[1., 0.], [0., 1.]]])
        mask = torch.tensor([[True, False]])
        first = FrameMetricAccumulator(2, [0])
        first.update(logits, labels, mask)
        logits[0, 1] = torch.tensor([1e5, -1e5])
        second = FrameMetricAccumulator(2, [0])
        second.update(logits, labels, mask)
        self.assertEqual(first.compute(), second.compute())

    def test_training_writer_logs_actual_timing_keys_without_metadata(self):
        from train import _write_tensorboard_segmentation_probe

        class Writer:
            def __init__(self):
                self.scalars = {}

            def add_scalar(self, name, value, step):
                self.scalars[name] = (value, step)

            def flush(self):
                pass

        writer = Writer()
        summary = {
            "best_epoch": 4,
            "best_val": {name: {"frame_map": 0.3, "bce": 0.1, "micro_f1": 0.2,
                                "classes_with_positives": 60, "supervised_frames": 100}
                         for name in ("babel-120", "babel-60")},
            "timings": {"feature_extraction_or_cache_load_seconds": 1.25,
                        "head_seconds": 2.5, "evaluation_call_seconds": 4.0},
        }
        _write_tensorboard_segmentation_probe(writer, global_step=300, summary=summary,
            state={"best_val_map": 0.4, "best_epoch": 30})
        for tag, value in (("feature_extraction_seconds", 1.25),
                           ("head_training_seconds", 2.5), ("total_seconds", 4.0)):
            self.assertEqual(writer.scalars[f"segmentation_probe/timing/{tag}"], (value, 300))
        self.assertFalse(any("classes_with" in tag or "supervised_frames" in tag
                             for tag in writer.scalars))

    def test_frozen_encoder_complete_patches_resume_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "experiment.segmentation_probe.online.BabelSegmentationDataset", _Dataset
        ):
            root = Path(directory)
            config, options = _config(root)
            torch.manual_seed(5)
            encoder = MotionPatchTransformer1D(6, 8, temporal_patch_size=2,
                embed_dim=12, depth=1, num_heads=3).eval().requires_grad_(False)
            original = copy.deepcopy(encoder.state_dict())
            probe = OnlineSegmentationProbe(config, options, device="cpu")
            rng = torch.get_rng_state().clone()
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("test interruption")
                return train_head_epoch(*args, **kwargs)

            with patch("experiment.segmentation_probe.online.train_head_epoch", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "test interruption"):
                    probe.evaluate(encoder, pretrain_epoch=30)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            output = probe.output_root / "epoch-0030"
            saved = torch.load(output / "latest.pth.tar", map_location="cpu", weights_only=False)
            self.assertEqual(saved["next_epoch"], 2)
            self.assertIn("momentum_buffer", next(iter(saved["optimizer"]["state"].values())))
            cache_path = output / "train-tokens.pt"
            cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            corrupted = copy.deepcopy(cache_payload)
            corrupted["cache"]["features"][0, 0, 0] += 10
            torch.save(corrupted, cache_path)
            with self.assertRaisesRegex(ValueError, "cache checksum mismatch"):
                probe.evaluate(encoder, pretrain_epoch=30)
            corrupted = copy.deepcopy(cache_payload)
            corrupted["cache"]["sample_ids"].reverse()
            corrupted["cache_sha256"] = _cache_digest(corrupted["cache"])
            torch.save(corrupted, cache_path)
            with self.assertRaisesRegex(ValueError, "cache sample order"):
                probe.evaluate(encoder, pretrain_epoch=30)
            torch.save(cache_payload, cache_path)
            with patch.object(probe, "_extract", side_effect=AssertionError("cache not reused")):
                result = probe.evaluate(encoder, pretrain_epoch=30)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertIsNone(result["test"])
            self.assertEqual(result["split_counts"]["val"]["complete_patch_frames"], 18)
            self.assertEqual(result["split_counts"]["val"]["raw_frames"], 20)
            resumed = torch.load(output / "latest.pth.tar", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["next_epoch"], 4)
            self.assertEqual(resumed["normalizer"]["fit_tokens"], 13)

            independent = copy.deepcopy(config)
            independent["logging"]["folder"] = str(root / "control")
            reference_probe = OnlineSegmentationProbe(independent, options, device="cpu")
            reference = reference_probe.evaluate(encoder, pretrain_epoch=30)
            control = torch.load(reference_probe.output_root / "epoch-0030/latest.pth.tar", weights_only=False)
            self.assertEqual(result["best_val"], reference["best_val"])
            for name, value in control["head"].items():
                torch.testing.assert_close(value, resumed["head"][name], rtol=0, atol=0)
            for name, value in encoder.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
            with patch("experiment.segmentation_probe.online.train_head_epoch", side_effect=AssertionError("result not reused")):
                self.assertEqual(probe.evaluate(encoder, pretrain_epoch=30), result)
            with torch.no_grad():
                next(encoder.parameters()).add_(0.01)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                probe.evaluate(encoder, pretrain_epoch=30)


if __name__ == "__main__":
    unittest.main()
