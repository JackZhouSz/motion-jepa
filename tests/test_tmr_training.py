"""Offline end-to-end tests for raw and frozen-JEPA text-motion alignment."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from test_linear_probe import _write_checkpoint, _write_dataset


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_paired_fixture(root: Path) -> Path:
    """Add natural-language, duplicate-caption pairs to the existing NPY fixture."""
    _write_dataset(root, num_frames=4, motion_dim=6)
    metadata_path = root / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_dataset"] = "BONES-SEED/soma_uniform"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    records_path = root / "index.json"
    records = json.loads(records_path.read_text())
    annotations = []
    for record in records:
        sample_id, split = record["id"], record["split"]
        record.update(
            source_id=sample_id,
            start_frame=0,
            end_frame=4,
            motion_path=f"motions/{split}/{sample_id}.npy",
            fps=30,
            length=4,
            motion_dim=6,
        )
        caption = (
            "A person walks." if record["metadata"]["style"] == "A"
            else "A person runs."
        )
        annotations.append({
            "filename": sample_id,
            "events": [
                {"start_time": 0.0, "end_time": 4 / 30, "description": caption},
                {"start_time": 0.0, "end_time": 4 / 30, "description": (
                    "A person walks forward." if record["metadata"]["style"] == "A"
                    else "A person runs quickly."
                )},
            ],
        })
    records_path.write_text(json.dumps(records), encoding="utf-8")
    np.save(root / "stats/mean.npy", np.zeros(6, dtype=np.float32))
    np.save(root / "stats/std.npy", np.ones(6, dtype=np.float32))
    annotations_path = root.parent / "temporal-annotations.jsonl"
    annotations_path.write_text(
        "".join(json.dumps(record) + "\n" for record in annotations), encoding="utf-8"
    )
    return annotations_path


def _write_text_backbone(path: Path) -> None:
    """Save a real, tiny encoder and tokenizer without contacting Hugging Face."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        T5GemmaConfig,
        T5GemmaEncoderModel,
        T5GemmaModuleConfig,
    )

    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "A": 4,
        "person": 5,
        "walks": 6,
        "runs": 7,
        "stands": 8,
        "turns": 9,
        ".": 10,
    }
    backend = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=32,
    )
    tokenizer.save_pretrained(path)
    encoder_config = T5GemmaModuleConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=32,
        layer_types=["full_attention"],
        query_pre_attn_scalar=8,
    )
    config = T5GemmaConfig(
        encoder=encoder_config, is_encoder_decoder=False, vocab_size=32
    )
    model = T5GemmaEncoderModel(config)
    if any("decoder" in name for name, _ in model.named_parameters()):
        raise AssertionError("The fixture must allocate only an encoder")
    model.save_pretrained(path)


@unittest.skipUnless(
    importlib.util.find_spec("transformers") is not None,
    "TMR integration requires its optional Transformers dependency",
)
class TMRTrainingIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self._previous_threads)

    def _training_args(
        self,
        root: Path,
        annotations: Path,
        output: Path,
        *,
        input_source: str = "raw",
        resume: bool = False,
        num_workers: int = 0,
        train_fraction: float = 1.0,
        text_depth: int | None = None,
        motion_depth: int | None = None,
    ):
        from experiment.tmr import train

        command = [
            "--dataset-root", str(root / "dataset"),
            "--annotations-path", str(annotations),
            "--cache-root", str(root / "cache"),
            "--input-source", input_source,
            "--stats-path", str(root / "dataset/stats"),
            "--text-model", str(root / "tiny-text"),
            "--max-text-length", "32",
            "--output-root", str(output),
            "--device", "cpu",
            "--seed", "42",
            "--train-fraction", str(train_fraction),
            "--epochs", "2",
            "--warmup-epochs", "0",
            "--batch-size", "2",
            "--eval-batch-size", "2",
            "--num-workers", str(num_workers),
            "--lr", "0.001",
            "--final-lr", "0.0001",
            "--weight-decay", "0.01",
            "--gradient-clip", "1.0",
            "--embed-dim", "16",
            "--depth", "1",
            "--num-heads", "2",
            "--ff-dim", "32",
            "--dropout", "0.1",
            "--no-use-bfloat16",
        ]
        if input_source == "jepa":
            command += ["--jepa-checkpoint", str(root / "jepa.pth.tar")]
        if text_depth is not None:
            command += ["--text-depth", str(text_depth)]
        if motion_depth is not None:
            command += ["--motion-depth", str(motion_depth)]
        if resume:
            command.append("--resume")
        return train.build_parser().parse_args(command)

    def _assert_completed_run(self, output: Path, summary: dict) -> dict:
        for name in (
            "latest.pth.tar", "best.pth.tar", "metrics.csv", "config.json", "summary.json"
        ):
            self.assertTrue((output / name).is_file(), name)
        with (output / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertIn(summary["best_epoch"], (1, 2))
        self.assertEqual(summary["selection"], "mean_bidirectional_val_r1")
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        latest = torch.load(output / "latest.pth.tar", map_location="cpu", weights_only=False)
        self.assertEqual(latest["next_epoch"], 2)
        self.assertTrue(latest["optimizer"]["state"], "Training must take optimizer steps")
        self.assertTrue(all(
            name.startswith(("text_encoder.", "motion_encoder."))
            for name in latest["model"]
        ), "Training checkpoints must contain only trainable alignment heads")
        for metric in summary["test"]:
            if metric.endswith(("_r1", "_r5", "_r10")):
                self.assertGreaterEqual(summary["test"][metric], 0.0)
                self.assertLessEqual(summary["test"][metric], 1.0)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self.assertEqual(summary["tensorboard_dir"], str(output / "tensorboard"))
        events = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 0}).Reload()
        loss_events = events.Scalars("train/loss")
        self.assertEqual([event.step for event in loss_events], list(range(1, latest["global_step"] + 1)))
        self.assertTrue(all(event.value >= 0 for event in loss_events))
        self.assertEqual(len(events.Scalars("train/learning_rate")), latest["global_step"])
        self.assertEqual(len(events.Scalars("train/epoch_loss")), 2)
        for row, event in zip(rows, events.Scalars("train/epoch_loss")):
            self.assertAlmostEqual(event.value, float(row["train_loss"]), places=6)
        for key, value in summary["test"].items():
            event = events.Scalars(f"test/{key}")[-1]
            self.assertEqual(event.step, latest["global_step"])
            self.assertAlmostEqual(event.value, value, places=6)
        for key in ("t2m_r1", "m2t_r1", "mean_r1"):
            self.assertEqual(len(events.Scalars(f"val/{key}")), 2)
        return latest

    def _assert_deterministic_resume(
        self, root: Path, annotations: Path, num_workers: int, train_fraction: float = 1.0,
        *, text_depth: int | None = None, motion_depth: int | None = None,
        legacy_checkpoint: bool = False,
    ) -> tuple[Path, dict]:
        from experiment.tmr import train

        raw_output = root / f"raw-workers-{num_workers}"
        raw_summary = train.run(self._training_args(
            root, annotations, raw_output, num_workers=num_workers, train_fraction=train_fraction,
            text_depth=text_depth, motion_depth=motion_depth,
        ))
        self.assertEqual(raw_summary["input_source"], "raw")
        uninterrupted = self._assert_completed_run(raw_output, raw_summary)
        self.assertEqual(uninterrupted["config"]["model"]["text_depth"], text_depth or 1)
        self.assertEqual(uninterrupted["config"]["model"]["motion_depth"], motion_depth or 1)

        resumed_output = root / f"resumed-workers-{num_workers}"
        interrupted_args = self._training_args(
            root, annotations, resumed_output, num_workers=num_workers, train_fraction=train_fraction,
            text_depth=text_depth, motion_depth=motion_depth,
        )
        original_save = train._atomic_torch_save

        def interrupt_after_first_epoch(value, path):
            original_save(value, path)
            if path.name == "latest.pth.tar" and value.get("next_epoch") == 1:
                raise RuntimeError("simulated training interruption")

        with mock.patch.object(train, "_atomic_torch_save", side_effect=interrupt_after_first_epoch):
            with self.assertRaisesRegex(RuntimeError, "simulated training interruption"):
                train.run(interrupted_args)
        interrupted = torch.load(
            resumed_output / "latest.pth.tar", map_location="cpu", weights_only=False
        )
        self.assertEqual(interrupted["next_epoch"], 1)
        if legacy_checkpoint:
            # Historical checkpoints contain only the shared depth. Preserve all
            # states while exercising their resume and standalone-evaluation paths.
            for path in (resumed_output / "latest.pth.tar", resumed_output / "best.pth.tar"):
                saved = torch.load(path, map_location="cpu", weights_only=False)
                saved["config"]["model"].pop("text_depth")
                saved["config"]["model"].pop("motion_depth")
                original_save(saved, path)
            from experiment.tmr import evaluate

            evaluate.run(evaluate.build_parser().parse_args([
                "--checkpoint", str(resumed_output / "best.pth.tar"), "--split", "val",
                "--device", "cpu", "--batch-size", "2", "--num-workers", "0",
            ]))
        # Emulate logs written after the last committed checkpoint. Resume must
        # purge these orphan events while keeping that checkpoint's evaluation.
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(log_dir=str(resumed_output / "tensorboard")) as writer:
            writer.add_scalar("train/loss", -999, interrupted["global_step"] + 1)
            writer.add_scalar("val/mean_r1", -999, interrupted["global_step"] + 1)
        resumed_args = self._training_args(
            root, annotations, resumed_output, resume=True, num_workers=num_workers, train_fraction=train_fraction,
            text_depth=text_depth or 1, motion_depth=motion_depth or 1,
        )
        # Explicit head depths describe the same architecture despite a different fallback.
        resumed_args.depth = 6
        resumed_summary = train.run(resumed_args)
        resumed = self._assert_completed_run(resumed_output, resumed_summary)
        self.assertEqual(resumed_summary["test"], raw_summary["test"])
        self.assertEqual(resumed_summary["best_epoch"], raw_summary["best_epoch"])
        for name in uninterrupted["model"]:
            torch.testing.assert_close(
                resumed["model"][name], uninterrupted["model"][name], rtol=0, atol=0
            )

        changed_args = self._training_args(
            root, annotations, resumed_output, resume=True, num_workers=num_workers, train_fraction=train_fraction,
            text_depth=text_depth, motion_depth=motion_depth,
        )
        changed_args.lr *= 2
        with self.assertRaisesRegex(ValueError, "(?i)resume|config|signature|differ"):
            train.run(changed_args)
        changed_args.lr /= 2
        changed_args.text_depth = text_depth or 1
        changed_args.motion_depth = (motion_depth or 1) + 1
        with self.assertRaisesRegex(ValueError, "Resume config"):
            train.run(changed_args)
        if train_fraction != 1:
            changed_args.motion_depth = motion_depth or 1
            changed_args.train_fraction = .5
            with self.assertRaisesRegex(ValueError, "Resume config"):
                train.run(changed_args)
        return raw_output, raw_summary

    def test_invalid_head_depths_fail_before_loading_features(self):
        from experiment.tmr import train

        for flag in ("--depth", "--text-depth", "--motion-depth"):
            for value in ("0", "-1"):
                with self.subTest(flag=flag, value=value):
                    args = train.build_parser().parse_args([flag, value])
                    with mock.patch.object(train, "load_prepared_datasets") as load_features:
                        with self.assertRaisesRegex(ValueError, flag):
                            train.run(args)
                        load_features.assert_not_called()

    def test_fraction_reuses_full_cache_for_both_inputs_resume_and_evaluation(self):
        from experiment.tmr import evaluate, features, train
        from experiment.tmr.dataset import json_digest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = _write_paired_fixture(root / "dataset")
            _write_text_backbone(root / "tiny-text")
            _write_checkpoint(root / "jepa.pth.tar", root / "dataset/stats")
            for source in ("raw", "jepa"):
                features.prepare_caches(
                    dataset_root=root / "dataset", annotations_path=annotations,
                    cache_root=root / "cache", input_source=source,
                    jepa_checkpoint=root / "jepa.pth.tar" if source == "jepa" else None,
                    stats_path=root / "dataset/stats", text_model=str(root / "tiny-text"),
                    max_text_length=32, device="cpu", feature_batch_size=2, text_batch_size=2,
                )
            cached_files = {path: _file_hash(path) for path in (root / "cache").rglob("*") if path.is_file()}
            paired = {}
            with (
                mock.patch.object(features, "load_text_backbone", side_effect=AssertionError("text backbone loaded")),
                mock.patch.object(features, "load_frozen_encoder", side_effect=AssertionError("JEPA backbone loaded")),
            ):
                for source in ("raw", "jepa"):
                    args = self._training_args(root, annotations, root / f"{source}-subset", input_source=source, train_fraction=.75)
                    datasets, provenance = features.load_prepared_datasets(**train.data_configuration(args))
                    self.assertEqual({split: len(dataset) for split, dataset in datasets.items()}, {"train": 3, "val": 2, "test": 2})
                    paired[source] = datasets["train"].sample_ids
                    self.assertEqual(provenance["train_subset"]["sample_ids_sha256"], json_digest(paired[source]))
                    if source == "jepa":
                        selected = torch.cat([datasets["train"].motion_bank[key] for key in paired[source]]).double()
                        torch.testing.assert_close(datasets["train"].motion_mean, selected.mean(0).float())
                        torch.testing.assert_close(datasets["train"].motion_std, selected.std(0, correction=0).float().clamp_min(1e-6))
                        self.assertEqual(provenance["jepa_feature_stats"]["num_tokens"], len(selected))
                        for split in ("val", "test"):
                            dataset = datasets[split]
                            original = dataset.motion_bank[dataset.sample_ids[0]]
                            torch.testing.assert_close(dataset[0]["motion_tokens"], (original - datasets["train"].motion_mean) / datasets["train"].motion_std)
                    else:
                        self.assertIsNone(datasets["train"].motion_mean)
                self.assertEqual(paired["raw"], paired["jepa"])
                self.assertEqual(
                    features.prepare_jepa_statistics(root / "cache", train_fraction=.75, train_subset_seed=42),
                    provenance["jepa_feature_stats"],
                )
                runs = []
                for workers in (0, 2):
                    runs.append(self._assert_deterministic_resume(root, annotations, workers, train_fraction=.75))
                jepa_output = root / "jepa-subset"
                jepa_summary = train.run(self._training_args(root, annotations, jepa_output, input_source="jepa", train_fraction=.75))
                saved = self._assert_completed_run(jepa_output, jepa_summary)
                self.assertEqual(saved["config"]["data"]["train_fraction"], .75)
                runs.append((jepa_output, jepa_summary))
                for output, summary in runs:
                    self.assertEqual(summary["split_counts"], {"train": 3, "val": 2, "test": 2})
                    manifest = json.loads((output / "train-subset.json").read_text())
                    self.assertEqual(manifest["sample_ids"], paired["raw"])
                    evaluation_args = evaluate.build_parser().parse_args([
                        "--checkpoint", str(output / "best.pth.tar"), "--split", "test",
                        "--device", "cpu", "--batch-size", "2", "--num-workers", "0",
                    ])
                    self.assertEqual(evaluate.run(evaluation_args), summary["test"])
            self.assertEqual(cached_files, {path: _file_hash(path) for path in cached_files})

    def test_cache_train_resume_and_standalone_evaluation_for_both_inputs(self):
        from experiment.tmr import evaluate, features, train
        from experiment.tmr.dataset import collate_pairs
        from experiment.tmr.model import AlignmentConfig, TextMotionAlignment
        from transformers import AutoTokenizer, T5GemmaEncoderModel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = _write_paired_fixture(root / "dataset")
            _write_text_backbone(root / "tiny-text")
            _write_checkpoint(root / "jepa.pth.tar", root / "dataset/stats")
            text_hash = _file_hash(root / "tiny-text/model.safetensors")
            jepa_hash = _file_hash(root / "jepa.pth.tar")
            prepared = {}
            for input_source in ("raw", "jepa"):
                features.prepare_caches(
                    dataset_root=root / "dataset",
                    annotations_path=annotations,
                    cache_root=root / "cache",
                    input_source=input_source,
                    jepa_checkpoint=root / "jepa.pth.tar" if input_source == "jepa" else None,
                    stats_path=root / "dataset/stats",
                    text_model=str(root / "tiny-text"),
                    max_text_length=32,
                    device="cpu",
                    feature_batch_size=2,
                    text_batch_size=2,
                    num_workers=0,
                )
                prepared[input_source] = features.load_prepared_datasets(
                    root / "dataset", annotations, root / "cache", input_source,
                    root / "jepa.pth.tar" if input_source == "jepa" else None,
                    "target_encoder", root / "dataset/stats", str(root / "tiny-text"),
                    None, 32,
                )[0]
            for split in ("train", "val", "test"):
                self.assertEqual(
                    prepared["raw"][split].sample_ids, prepared["jepa"][split].sample_ids
                )
                self.assertEqual(
                    prepared["raw"][split].caption_ids, prepared["jepa"][split].caption_ids
                )
                self.assertEqual(prepared["raw"][split].caption_candidates, prepared["jepa"][split].caption_candidates)
                self.assertTrue(all(len(candidates) == 2 for candidates in prepared["raw"][split].caption_candidates))
            self.assertEqual(prepared["raw"]["train"][0]["motion_tokens"].shape, (4, 6))
            self.assertEqual(prepared["jepa"]["train"][0]["motion_tokens"].shape, (4, 192))

            # Cached training and evaluation must work without loading a text backbone.
            with (
                mock.patch.object(T5GemmaEncoderModel, "from_pretrained", side_effect=AssertionError("backbone loaded during cached training")),
                mock.patch.object(AutoTokenizer, "from_pretrained", side_effect=AssertionError("tokenizer loaded during cached training")),
            ):
                raw_runs = []
                # Worker-seed draws must preserve shuffle order across a restart.
                for num_workers in (0, 2):
                    with self.subTest(num_workers=num_workers):
                        raw_runs.append(self._assert_deterministic_resume(
                            root, annotations, num_workers,
                            text_depth=2 if num_workers == 2 else None,
                            motion_depth=1 if num_workers == 2 else None,
                            legacy_checkpoint=num_workers == 0,
                        ))

                jepa_output = root / "jepa-output"
                jepa_summary = train.run(self._training_args(
                    root, annotations, jepa_output, input_source="jepa",
                    text_depth=2, motion_depth=1,
                ))
                self.assertEqual(jepa_summary["input_source"], "jepa")
                saved_jepa = self._assert_completed_run(jepa_output, jepa_summary)
                self.assertIn("jepa_feature_stats", saved_jepa["provenance"])
                self.assertEqual(saved_jepa["config"]["model"]["text_depth"], 2)
                self.assertEqual(saved_jepa["config"]["model"]["motion_depth"], 1)

                quiet_output = root / "no-tensorboard"
                quiet_args = self._training_args(root, annotations, quiet_output)
                quiet_args.tensorboard = False
                quiet_summary = train.run(quiet_args)
                self.assertIsNone(quiet_summary["tensorboard_dir"])
                self.assertFalse((quiet_output / "tensorboard").exists())

                model = TextMotionAlignment(AlignmentConfig(**saved_jepa["config"]["model"]))
                model.load_state_dict(saved_jepa["model"])
                model.eval()
                batch = collate_pairs([prepared["jepa"]["train"][0], prepared["jepa"]["train"][2]])
                with torch.no_grad():
                    motions, texts = model(
                        batch["motion_tokens"], batch["motion_mask"],
                        batch["text_tokens"], batch["text_mask"],
                    )
                torch.testing.assert_close(motions.norm(dim=-1), torch.ones(2))
                torch.testing.assert_close(texts.norm(dim=-1), torch.ones(2))

                for output, summary in [*raw_runs, (jepa_output, jepa_summary)]:
                    evaluation_args = evaluate.build_parser().parse_args([
                        "--checkpoint", str(output / "best.pth.tar"),
                        "--split", "test", "--device", "cpu", "--batch-size", "2",
                        "--num-workers", "0", "--chunk-size", "2",
                    ])
                    self.assertEqual(evaluate.run(evaluation_args), summary["test"])

                # A head trained on unnormalized JEPA tokens must not silently
                # evaluate against the newly standardized feature distribution.
                legacy = dict(saved_jepa)
                legacy["provenance"] = dict(saved_jepa["provenance"])
                legacy["provenance"].pop("jepa_feature_stats")
                legacy_path = root / "legacy-jepa.pth.tar"
                torch.save(legacy, legacy_path)
                legacy_args = evaluate.build_parser().parse_args([
                    "--checkpoint", str(legacy_path), "--split", "test", "--device", "cpu",
                ])
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    evaluate.run(legacy_args)

            self.assertEqual(_file_hash(root / "tiny-text/model.safetensors"), text_hash)
            self.assertEqual(_file_hash(root / "jepa.pth.tar"), jepa_hash)


if __name__ == "__main__":
    unittest.main()
