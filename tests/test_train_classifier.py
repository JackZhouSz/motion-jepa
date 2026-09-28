"""Tests for raw-motion and frozen-feature classifier training."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiment.linear_probe.cnn import MotionCNNClassifier
from experiment.linear_probe.linear import RawMotionLinearClassifier
from experiment.linear_probe import train_classifier
from experiment.linear_probe.train_classifier import make_classifier, run
from experiment.linear_probe.transformer import MotionTransformerClassifier
from test_linear_probe import _write_checkpoint, _write_dataset


class ClassifierModelTest(unittest.TestCase):
    def test_raw_linear_is_one_affine_layer_over_masked_temporal_mean(self):
        model = RawMotionLinearClassifier(input_dim=6, num_frames=4, num_classes=2)
        self.assertEqual(
            [module for module in model.modules() if isinstance(module, torch.nn.Linear)],
            [model.head],
        )
        self.assertEqual(model.head.in_features, 6)
        motion = torch.randn(3, 4, 6)
        active = torch.tensor(
            [[True, True, True, True], [True, True, False, False], [False] * 4]
        )
        self.assertEqual(model(motion, active).shape, (3, 2))
        changed = motion.clone()
        changed[1, 2:] = 1.0e6
        torch.testing.assert_close(model(motion, active)[1], model(changed, active)[1])
        expected = model.head(motion[0].mean(dim=0))
        torch.testing.assert_close(model(motion, active)[0], expected)

    def test_default_raw_findings_root_is_unified_component(self):
        self.assertEqual(
            train_classifier.DEFAULT_RAW_FINDINGS_ROOT,
            train_classifier.PROJECT_ROOT
            / "findings/000-100style-classification/classifiers",
        )

    def test_shapes_backward_and_matched_parameter_counts(self):
        models = (
            MotionCNNClassifier(),
            MotionTransformerClassifier(),
        )
        counts = []
        motion = torch.randn(2, 90, 366)
        active = torch.ones(2, 90, dtype=torch.bool)
        for model in models:
            logits = model(motion, active)
            self.assertEqual(logits.shape, (2, 100))
            logits.square().mean().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
        self.assertTrue(all(800_000 <= count <= 1_200_000 for count in counts))
        self.assertLess(max(counts) / min(counts), 1.25)

    def test_invalid_frame_values_do_not_affect_logits(self):
        active = torch.tensor([[True] * 70 + [False] * 20])
        original = torch.randn(1, 90, 366)
        changed = original.clone()
        changed[:, 70:] = torch.randn_like(changed[:, 70:]) * 1000
        for model in (MotionCNNClassifier(), MotionTransformerClassifier()):
            model.eval()
            with torch.inference_mode():
                first = model(original, active)
                second = model(changed, active)
            torch.testing.assert_close(first, second)

    def test_factory_uses_dataset_dimensions(self):
        for name in ("cnn", "transformer", "linear"):
            model, config = make_classifier(
                name, motion_dim=6, num_frames=4, num_classes=2
            )
            logits = model(torch.randn(3, 4, 6), torch.ones(3, 4, dtype=torch.bool))
            self.assertEqual(logits.shape, (3, 2))
            self.assertEqual(config["num_classes"], 2)
            if name == "cnn":
                self.assertEqual(config["widths"], [128, 192, 256])
                self.assertEqual(config["blocks_per_stage"], 1)
            elif name == "transformer":
                self.assertEqual(config["embed_dim"], 128)
                self.assertEqual(config["depth"], 4)
                self.assertEqual(config["num_heads"], 4)

    def test_multiple_jepa_feature_dimensions_and_legacy_alias(self):
        for input_dim in (192, 768):
            motion = torch.randn(2, 4, input_dim)
            active = torch.ones(2, 4, dtype=torch.bool)
            cnn = MotionCNNClassifier(input_dim=input_dim, num_classes=3)
            transformer = MotionTransformerClassifier(
                input_dim=input_dim, num_frames=4, num_classes=3
            )
            self.assertEqual(cnn(motion, active).shape, (2, 3))
            self.assertEqual(transformer(motion, active).shape, (2, 3))
        legacy = MotionCNNClassifier(motion_dim=6, num_classes=2)
        self.assertEqual(legacy(torch.randn(1, 4, 6)).shape, (1, 2))
        with self.assertRaises(ValueError):
            MotionCNNClassifier(motion_dim=6, input_dim=7)

    def test_transformer_uses_a_learnable_cls_token(self):
        model = MotionTransformerClassifier(num_frames=4, motion_dim=6, num_classes=2)
        self.assertEqual(model.cls_token.shape, (1, 1, 128))
        self.assertEqual(model.position_embedding.shape, (1, 5, 128))
        with self.assertRaises(TypeError):
            MotionTransformerClassifier(
                num_frames=4,
                motion_dim=6,
                num_classes=2,
                pooling="cls_token",
            )
        logits = model(
            torch.randn(2, 4, 6),
            torch.tensor([[True, True, False, False], [False] * 4]),
        )
        logits.sum().backward()
        self.assertIsNotNone(model.cls_token.grad)


class SupervisedTrainingTest(unittest.TestCase):
    def test_validation_free_raw_linear_uses_last_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            output_root = root / "output"
            _write_dataset(dataset_root, validation_enabled=False)
            args = argparse.Namespace(
                model="linear",
                input_source="raw",
                dataset_root=dataset_root,
                output_root=output_root,
                findings_root=None,
                device="cpu",
                seed=42,
                epochs=2,
                warmup_epochs=0,
                batch_size=4,
                num_workers=0,
                lr=3.0e-4,
                final_lr=1.0e-6,
                weight_decay=0.05,
                gradient_clip=1.0,
                use_bfloat16=False,
                resume=False,
                overwrite=False,
            )
            summary = run(args)["linear"]
            self.assertEqual(summary["selection"], "fixed_last_epoch")
            self.assertFalse(summary["validation_used"])
            self.assertEqual(summary["best_epoch"], 2)
            self.assertIsNone(summary["best_val"])
            self.assertEqual(summary["head_filename"], "classifier-final.pth.tar")
            run_root = output_root / "linear/seed-42"
            self.assertTrue((run_root / "classifier-final.pth.tar").is_file())
            self.assertFalse((run_root / "classifier-best.pth.tar").exists())

    def test_two_epoch_cpu_training_outputs_and_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            output_root = root / "output"
            findings_root = root / "findings"
            _write_dataset(dataset_root)
            args = argparse.Namespace(
                model="all",
                dataset_root=dataset_root,
                output_root=output_root,
                findings_root=findings_root,
                device="cpu",
                seed=42,
                epochs=2,
                warmup_epochs=0,
                batch_size=4,
                num_workers=0,
                lr=3.0e-4,
                final_lr=1.0e-6,
                weight_decay=0.05,
                gradient_clip=1.0,
                use_bfloat16=False,
                resume=False,
                overwrite=False,
            )
            summaries = run(args)
            self.assertEqual(set(summaries), {"cnn", "transformer"})
            for model_name in summaries:
                run_root = output_root / model_name / "seed-42"
                for name in (
                    "metrics.csv",
                    "summary.json",
                    "classifier-best.pth.tar",
                    "class-index.json",
                ):
                    self.assertTrue((run_root / name).is_file(), f"{model_name}/{name}")
                self.assertFalse((run_root / "classifier-latest.pth.tar").exists())
                self.assertFalse((run_root / "model-config.json").exists())
                best = torch.load(
                    run_root / "classifier-best.pth.tar",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(
                    set(best), {"format_version", "model", "architecture"}
                )
                self.assertNotIn("pooling", best["architecture"])
                architecture = dict(best["architecture"])
                class_name = architecture.pop("name")
                classifier_type = (
                    MotionCNNClassifier
                    if class_name == "MotionCNNClassifier"
                    else MotionTransformerClassifier
                )
                reconstructed = classifier_type(**architecture)
                reconstructed.load_state_dict(best["model"], strict=True)
                self.assertNotIn("model_config", summaries[model_name])
                self.assertNotIn("input_source", summaries[model_name])
                self.assertNotIn("jepa_source", summaries[model_name])
            for name in (
                "README.md",
                "training-curves.png",
                "cnn-metrics.csv",
                "transformer-metrics.csv",
                "results.json",
            ):
                self.assertTrue((findings_root / name).is_file(), name)
            results = json.loads((findings_root / "results.json").read_text())
            self.assertEqual(results["seed"], 42)
            self.assertEqual(set(results["summaries"]), {"cnn", "transformer"})

            completed = summaries["cnn"]
            args.model = "cnn"
            args.output_root = root / "resume-output"
            original_torch_save = train_classifier._atomic_torch_save

            def interrupt_after_first_epoch(value, path):
                original_torch_save(value, path)
                if (
                    path.name == "classifier-latest.pth.tar"
                    and value.get("next_epoch") == 1
                ):
                    raise RuntimeError("simulated interruption")

            with mock.patch.object(
                train_classifier,
                "_atomic_torch_save",
                side_effect=interrupt_after_first_epoch,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run(args)
            interrupted_root = args.output_root / "cnn/seed-42"
            self.assertTrue(
                (interrupted_root / "classifier-latest.pth.tar").is_file()
            )
            latest = torch.load(
                interrupted_root / "classifier-latest.pth.tar",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                set(latest),
                {
                    "format_version",
                    "model",
                    "optimizer",
                    "scheduler",
                    "next_epoch",
                    "best_epoch",
                    "best_accuracy",
                    "rng_state",
                    "signature",
                },
            )
            self.assertEqual(latest["next_epoch"], 1)
            args.resume = True
            resumed = run(args)
            self.assertEqual(resumed["cnn"]["best_epoch"], completed["best_epoch"])
            self.assertEqual(
                resumed["cnn"]["test"],
                completed["test"],
            )
            self.assertFalse(
                (interrupted_root / "classifier-latest.pth.tar").exists()
            )

    def test_frozen_jepa_token_cache_checkpoint_metadata_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            _write_dataset(dataset_root, num_frames=4, motion_dim=6)
            stats_root = root / "pretraining/stats"
            stats_root.mkdir(parents=True)
            np.save(stats_root / "mean.npy", np.zeros(6, dtype=np.float32))
            np.save(stats_root / "std.npy", np.ones(6, dtype=np.float32))
            checkpoint = root / "jepa-run/motion-jepa-1d-latest.pth.tar"
            checkpoint.parent.mkdir()
            source_weights = _write_checkpoint(
                checkpoint,
                stats_root,
                num_frames=4,
                motion_dim=6,
            )
            source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            args = argparse.Namespace(
                model="all",
                input_source="jepa",
                jepa_checkpoint=checkpoint,
                checkpoint_key="target_encoder",
                stats_path=None,
                feature_batch_size=4,
                feature_cache_root=None,
                recompute_features=False,
                dataset_root=dataset_root,
                output_root=None,
                findings_root=None,
                device="cpu",
                seed=42,
                epochs=2,
                warmup_epochs=0,
                batch_size=4,
                num_workers=0,
                lr=3.0e-4,
                final_lr=1.0e-6,
                weight_decay=0.05,
                gradient_clip=1.0,
                use_bfloat16=False,
                resume=False,
                overwrite=False,
            )
            summaries = run(args)
            summary = summaries["cnn"]
            cache_root = checkpoint.parent / "linear-probe/token-features"
            for split, count in (("train", 4), ("val", 2), ("test", 2)):
                payload = torch.load(
                    cache_root / f"{split}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(payload["features"].dtype, torch.bfloat16)
                self.assertEqual(payload["features"].shape, (count, 4, 192))
                self.assertEqual(payload["lengths"].dtype, torch.long)
                self.assertEqual(payload["metadata"]["checkpoint_sha256"], source_hash)
                self.assertEqual(payload["metadata"]["checkpoint_key"], "target_encoder")
            output_root = checkpoint.parent / "linear-probe/classifiers"
            jepa_source = summary["signature"]["jepa_source"]
            self.assertEqual(summary["signature"]["input_source"], "jepa")
            self.assertEqual(jepa_source["model_name"], "mot_tiny_1d")
            self.assertEqual(jepa_source["feature_dim"], 192)
            self.assertEqual(jepa_source["checkpoint_sha256"], source_hash)
            self.assertEqual(jepa_source["stats_root"], str(stats_root.resolve()))
            self.assertNotEqual(
                jepa_source["stats_mean_sha256"],
                hashlib.sha256((dataset_root / "stats/mean.npy").read_bytes()).hexdigest(),
            )
            for model_name in ("cnn", "transformer"):
                model_output = output_root / model_name / "seed-42"
                saved = torch.load(
                    model_output / "classifier-best.pth.tar",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(
                    set(saved), {"format_version", "model", "architecture"}
                )
                self.assertNotIn("jepa_source", saved["architecture"])
                self.assertNotIn("input_source", saved["architecture"])
                self.assertNotIn("num_parameters", saved["architecture"])
                self.assertFalse((model_output / "classifier-latest.pth.tar").exists())
                self.assertFalse((model_output / "model-config.json").exists())
            reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for key, value in source_weights.items():
                torch.testing.assert_close(reloaded["target_encoder"][key], value)

            args.resume = True

            index_path = dataset_root / "index.json"
            original_index = index_path.read_text()
            index_path.write_text(original_index + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                run(args)
            index_path.write_text(original_index, encoding="utf-8")

            np.save(stats_root / "mean.npy", np.ones(6, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "stale"):
                run(args)
            np.save(stats_root / "mean.npy", np.zeros(6, dtype=np.float32))

            changed_checkpoint = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            changed_checkpoint["test_marker"] = True
            torch.save(changed_checkpoint, checkpoint)
            with self.assertRaisesRegex(ValueError, "stale"):
                run(args)


if __name__ == "__main__":
    unittest.main()
