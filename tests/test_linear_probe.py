"""Tests for frozen MotionJEPA linear probing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiment import linear_probe
from experiment.linear_probe import train_probe
from model import MODEL_FACTORIES


def _write_split(
    root: Path,
    split: str,
    values: list[tuple[str, float, str]],
    *,
    num_frames: int,
    motion_dim: int,
    fps: int,
) -> list[dict]:
    motion_root = root / "motions" / split
    motion_root.mkdir(parents=True, exist_ok=True)
    rows = []
    records = []
    for sample_id, value, style in values:
        relative = Path("motions") / split / f"{sample_id}.npy"
        motion = np.full((num_frames, motion_dim), value, dtype=np.float32)
        np.save(root / relative, motion, allow_pickle=False)
        rows.append(f"{sample_id},{relative.as_posix()},{fps},{num_frames}\n")
        records.append(
            {
                "id": sample_id,
                "split": split,
                "metadata": {"style": style},
            }
        )
    index_path = root / f"{split}.txt"
    index_path.write_text("".join(rows), encoding="utf-8")
    manifest = {
        "format": "motion_jepa_npy_v1",
        "split": split,
        "motion_root": f"motions/{split}",
        "num_samples": len(values),
        "dtype": "float32",
        "representation": "motion_jepa_366_v1",
        "fps": fps,
        "num_frames": num_frames,
        "motion_dim": motion_dim,
        "split_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    (root / "motions" / f"{split}.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    return records


def _write_dataset(
    root: Path,
    *,
    num_frames: int = 4,
    motion_dim: int = 6,
    validation_enabled: bool = True,
) -> None:
    fps = 30
    records = []
    records.extend(
        _write_split(
            root,
            "train",
            [
                ("train-a0", -1.0, "A"),
                ("train-a1", -0.8, "A"),
                ("train-b0", 1.0, "B"),
                ("train-b1", 0.8, "B"),
            ],
            num_frames=num_frames,
            motion_dim=motion_dim,
            fps=fps,
        )
    )
    records.extend(
        _write_split(
            root,
            "val",
            [("val-a", -0.9, "A"), ("val-b", 0.9, "B")]
            if validation_enabled
            else [],
            num_frames=num_frames,
            motion_dim=motion_dim,
            fps=fps,
        )
    )
    records.extend(
        _write_split(
            root,
            "test",
            [("test-a", -0.7, "A"), ("test-b", 0.7, "B")],
            num_frames=num_frames,
            motion_dim=motion_dim,
            fps=fps,
        )
    )
    (root / "meta.json").write_text(
        json.dumps(
            {
                "source_dataset": "100STYLE_soma77",
                "representation": "motion_jepa_366_v1",
                "motion_storage": "npy_float32_v1",
                "motion_dim": motion_dim,
                "num_frames": num_frames,
                "fps": fps,
                "validation_enabled": validation_enabled,
                "test_contents": ["FW"] if not validation_enabled else [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "index.json").write_text(
        json.dumps(records) + "\n",
        encoding="utf-8",
    )
    stats = root / "stats"
    stats.mkdir()
    np.save(stats / "mean.npy", np.full(motion_dim, 100.0, dtype=np.float32))
    np.save(stats / "std.npy", np.ones(motion_dim, dtype=np.float32))


def _write_checkpoint(
    path: Path,
    stats_root: Path,
    *,
    model_name: str = "mot_tiny_1d",
    num_frames: int = 4,
    motion_dim: int = 6,
) -> dict[str, torch.Tensor]:
    kwargs = {"in_chans": motion_dim, "num_frames": num_frames}
    if model_name.endswith("_2d"):
        kwargs.update(in_chans=366, num_joints=30)
        motion_dim = 366
    encoder = MODEL_FACTORIES[model_name](**kwargs)
    target = copy.deepcopy(encoder)
    config = {
        "data": {
            "root_path": str(stats_root.parent),
            "stats_path": stats_root.name,
            "num_frames": num_frames,
            "motion_dim": motion_dim,
            "num_joints": 30,
            "fps": 30,
        },
        "meta": {"model_name": model_name, "use_bfloat16": False},
    }
    torch.save(
        {
            "format_version": 1,
            "encoder": encoder.state_dict(),
            "target_encoder": target.state_dict(),
            "config": config,
        },
        path,
    )
    return copy.deepcopy(target.state_dict())


class LinearProbeUnitTest(unittest.TestCase):
    def test_strict_checkpoint_loading_for_1d_and_2d(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats"
            stats.mkdir()
            np.save(stats / "mean.npy", np.zeros(366, dtype=np.float32))
            np.save(stats / "std.npy", np.ones(366, dtype=np.float32))
            for model_name, motion_dim in (("mot_tiny_1d", 6), ("mot_tiny_2d", 366)):
                checkpoint = root / f"{model_name}.pth.tar"
                expected = _write_checkpoint(
                    checkpoint,
                    stats,
                    model_name=model_name,
                    motion_dim=motion_dim,
                )
                encoder, _, info = linear_probe.load_frozen_encoder(
                    checkpoint,
                    "target_encoder",
                    torch.device("cpu"),
                )
                self.assertEqual(info["model_name"], model_name)
                self.assertFalse(encoder.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
                for name, value in encoder.state_dict().items():
                    torch.testing.assert_close(value, expected[name])

                before = copy.deepcopy(encoder.state_dict())
                motion = torch.randn(1, 4, motion_dim)
                with torch.inference_mode():
                    encoder(
                        motion,
                        torch.tensor([30.0]),
                        valid_frames=torch.ones(1, 4, dtype=torch.bool),
                    )
                for name, value in encoder.state_dict().items():
                    torch.testing.assert_close(value, before[name])
                self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))

    def test_pooling_uses_only_valid_tokens(self):
        one_d = torch.tensor([[[1.0], [3.0], [100.0]]])
        pooled_1d = linear_probe.pool_encoder_output(one_d, torch.tensor([2]))
        torch.testing.assert_close(pooled_1d, torch.tensor([[2.0]]))

        two_d = torch.tensor([[[[1.0], [3.0]], [[5.0], [7.0]], [[100.0], [100.0]]]])
        pooled_2d = linear_probe.pool_encoder_output(two_d, torch.tensor([2]))
        torch.testing.assert_close(pooled_2d, torch.tensor([[4.0]]))

        spatial = linear_probe.pool_encoder_output(
            two_d,
            torch.tensor([2]),
            pooling=linear_probe.SPATIAL_FLATTEN_POOLING,
        )
        torch.testing.assert_close(spatial, torch.tensor([[3.0, 5.0]]))
        with self.assertRaisesRegex(ValueError, "requires 2D output"):
            linear_probe.pool_encoder_output(
                one_d,
                torch.tensor([2]),
                pooling=linear_probe.SPATIAL_FLATTEN_POOLING,
            )

    def test_stale_cache_is_rejected(self):
        metadata = {"format_version": 1, "feature_dim": 3}
        payload = {
            "metadata": metadata,
            "features": torch.zeros(2, 3),
            "labels": torch.zeros(2, dtype=torch.long),
            "sample_ids": ["a", "b"],
        }
        linear_probe._validate_feature_cache(payload, metadata)
        with self.assertRaisesRegex(ValueError, "stale"):
            linear_probe._validate_feature_cache(payload, {**metadata, "split": "train"})


class LinearProbeEndToEndTest(unittest.TestCase):
    def test_validation_free_probe_saves_last_epoch_and_tests_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            output = root / "output"
            _write_dataset(dataset_root, validation_enabled=False)
            checkpoint = root / "checkpoint.pth.tar"
            _write_checkpoint(checkpoint, dataset_root / "stats")
            args = argparse.Namespace(
                checkpoint=checkpoint,
                dataset_root=dataset_root,
                output=output,
                checkpoint_key="target_encoder",
                stats_path=None,
                device="cpu",
                feature_batch_size=4,
                batch_size=2,
                num_workers=0,
                epochs=3,
                lr=0.3,
                momentum=0.9,
                weight_decay=0.0,
                seed=0,
                pooling=linear_probe.GLOBAL_MEAN_POOLING,
                recompute_features=False,
                overwrite=False,
            )
            original = train_probe.evaluate_classifier
            test_calls = 0

            def counting_evaluate(classifier, features, labels, **kwargs):
                nonlocal test_calls
                if features.shape[0] == 2:
                    test_calls += 1
                return original(classifier, features, labels, **kwargs)

            from unittest import mock
            with mock.patch.object(train_probe, "evaluate_classifier", side_effect=counting_evaluate):
                summary = train_probe.run(args)
            self.assertEqual(summary["selection"], "fixed_last_epoch")
            self.assertFalse(summary["validation_used"])
            self.assertEqual(summary["best_epoch"], 3)
            self.assertIsNone(summary["best_val"])
            self.assertEqual(summary["test_content"], "FW")
            self.assertEqual(test_calls, 1)
            self.assertFalse((output / "features/val.pt").exists())
            self.assertEqual(summary["head_filename"], "linear-probe-final.pth.tar")
            self.assertFalse((output / "linear-probe-best.pth.tar").exists())
            head = torch.load(output / "linear-probe-final.pth.tar", weights_only=False)
            self.assertEqual(head["selection"], "fixed_last_epoch")
            self.assertEqual(head["epoch"], 3)

    def test_cpu_feature_cache_and_two_epoch_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            output = root / "output"
            _write_dataset(dataset_root)
            pretrain_stats = root / "pretrain-stats"
            pretrain_stats.mkdir()
            np.save(pretrain_stats / "mean.npy", np.zeros(6, dtype=np.float32))
            np.save(pretrain_stats / "std.npy", np.ones(6, dtype=np.float32))
            checkpoint = root / "checkpoint.pth.tar"
            _write_checkpoint(checkpoint, pretrain_stats)
            args = argparse.Namespace(
                checkpoint=checkpoint,
                dataset_root=dataset_root,
                output=output,
                checkpoint_key="target_encoder",
                stats_path=None,
                device="cpu",
                feature_batch_size=4,
                batch_size=2,
                num_workers=0,
                epochs=2,
                lr=0.1,
                momentum=0.9,
                weight_decay=0.0,
                seed=0,
                recompute_features=False,
                overwrite=False,
            )
            summary = train_probe.run(args)
            self.assertEqual(summary["stats_root"], str(pretrain_stats.resolve()))
            self.assertEqual(summary["num_classes"], 2)
            self.assertIn(summary["best_epoch"], (1, 2))
            for name in (
                "metrics.csv",
                "summary.json",
                "linear-probe-best.pth.tar",
                "class-index.json",
                "features/train.pt",
                "features/val.pt",
                "features/test.pt",
            ):
                self.assertTrue((output / name).is_file(), name)

            cache = torch.load(output / "features/train.pt", weights_only=False)
            self.assertEqual(cache["features"].dtype, torch.float32)
            self.assertEqual(cache["features"].shape, (4, 192))
            self.assertEqual(cache["metadata"]["checkpoint_key"], "target_encoder")
            self.assertTrue(torch.isfinite(cache["features"]).all())

            dataset = linear_probe.SingleLabelMotionDataset(
                dataset_root,
                "train",
                num_frames=4,
                fps=30,
                motion_dim=6,
                stats_root=pretrain_stats,
                label_index=linear_probe.load_classification_label_index(dataset_root),
            )
            motion, *_ = dataset[0]
            self.assertAlmostEqual(float(motion[0, 0]), -1.0)

            args.overwrite = True
            reused = train_probe.run(args)
            self.assertEqual(reused["split_counts"], summary["split_counts"])

            args.output = None
            default_output = train_probe.run(args)
            self.assertEqual(default_output["checkpoint"], str(checkpoint.resolve()))
            self.assertTrue(
                (checkpoint.parent / "linear-probe" / "summary.json").is_file()
            )

    def test_2d_probe_preserves_spatial_token_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            _write_dataset(dataset_root, motion_dim=366)
            checkpoint = root / "checkpoint-2d.pth.tar"
            _write_checkpoint(
                checkpoint,
                dataset_root / "stats",
                model_name="mot_tiny_2d",
                motion_dim=366,
            )
            args = argparse.Namespace(
                checkpoint=checkpoint,
                dataset_root=dataset_root,
                output=None,
                checkpoint_key="target_encoder",
                stats_path=None,
                device="cpu",
                feature_batch_size=4,
                batch_size=2,
                num_workers=0,
                epochs=1,
                lr=0.1,
                momentum=0.9,
                weight_decay=0.0,
                seed=0,
                pooling=linear_probe.SPATIAL_FLATTEN_POOLING,
                recompute_features=False,
                overwrite=False,
            )
            summary = train_probe.run(args)
            self.assertEqual(summary["pooling"], linear_probe.SPATIAL_FLATTEN_POOLING)
            self.assertEqual(summary["feature_dim"], 30 * 192)
            cache = torch.load(
                root / "linear-probe-2d/features/train.pt", weights_only=False
            )
            self.assertEqual(cache["features"].shape, (4, 30 * 192))
            self.assertEqual(
                cache["metadata"]["pooling"],
                linear_probe.SPATIAL_FLATTEN_POOLING,
            )


if __name__ == "__main__":
    unittest.main()
