"""Temporal joins, ragged-cache integrity, and encoder-only text loading."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiment.tmr.dataset import (
    RaggedTokenBank, build_paired_index, collate_pairs, collect_caption_candidates,
)
from experiment.tmr.features import (
    load_prepared_datasets, load_text_backbone, prepare_caches, write_ragged_bank,
)
from model.token_layout import TokenLayout


def _write_dataset(root: Path):
    root.mkdir()
    (root / "motions").mkdir()
    (root / "stats").mkdir()
    np.save(root / "stats/mean.npy", np.full(3, 2, dtype=np.float32))
    np.save(root / "stats/std.npy", np.full(3, 2, dtype=np.float32))
    (root / "meta.json").write_text(json.dumps({
        "representation": "motion_jepa_366_v1", "motion_storage": "npy_float32_v1",
        "motion_dim": 3, "num_frames": 4, "fps": 2,
    }))
    records, annotations = [], []
    for split in ("train", "val", "test"):
        folder = root / "motions" / split
        folder.mkdir()
        lines = []
        count = 3 if split == "train" else 1
        for index in range(count):
            sample_id = f"{split}-{index}"
            relative = f"motions/{split}/{sample_id}.npy"
            length = 2 if index == 1 else 4
            np.save(root / relative, np.full((length, 3), 4 + index, dtype=np.float32))
            records.append({
                "id": sample_id, "source_id": sample_id, "split": split,
                "start_frame": 0, "end_frame": length, "fps": 2,
                "length": length, "motion_dim": 3, "motion_path": relative,
            })
            lines.append(f"{sample_id},{relative},2,{length}\n")
            annotations.append({"filename": sample_id, "events": [{
                "start_time": 3.0 if index == 2 else 0.0,
                "end_time": 4.0 if index == 2 else length / 2,
                "description": "walk forward" if index != 1 else "turn left slowly",
            }]})
        text = "".join(lines)
        (root / f"{split}.txt").write_text(text)
        (root / "motions" / f"{split}.json").write_text(json.dumps({
            "format": "motion_jepa_npy_v1", "split": split,
            "dtype": "float32", "representation": "motion_jepa_366_v1",
            "fps": 2, "num_frames": 4, "motion_dim": 3, "motion_root": f"motions/{split}",
            "num_samples": count, "split_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }))
    (root / "index.json").write_text(json.dumps(records))
    annotation_path = root.parent / "events.jsonl"
    annotation_path.write_text("".join(json.dumps(row) + "\n" for row in annotations))
    return annotation_path


class _Tokenizer:
    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None):
        values = [list(range(1, len(text.split()) + 2)) for text in texts]
        if truncation:
            values = [value[:max_length] for value in values]
        if return_tensors is None:
            return {"input_ids": values}
        length = max(map(len, values))
        ids = torch.zeros((len(values), length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, value in enumerate(values):
            ids[row, :len(value)] = torch.tensor(value)
            mask[row, :len(value)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _TextEncoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=input_ids.float()[..., None] + torch.arange(3))


class _SpatialEncoder(torch.nn.Module):
    embed_dim = 3

    def __init__(self, patch_size=2):
        super().__init__()
        self.token_layout = TokenLayout(
            kind="2d", patchified=True, raw_num_frames=4,
            token_num_frames=4 // patch_size, temporal_patch_size=patch_size,
            raw_num_joints=2, token_num_joints=2,
        )

    def forward(self, motion, fps, valid_frames):
        del fps, valid_frames
        tokens = motion[:, ::self.token_layout.temporal_patch_size]
        return torch.stack((tokens + 1, tokens + 3), dim=2)


def _fake_text_backbone(*args):
    return _TextEncoder(), _Tokenizer(), {
        "model": args[0], "revision": args[1], "resolved_revision": None,
        "tokenizer_sha256": "test-tokenizer", "feature_dim": 3,
        "dtype": "bfloat16", "model_class": "T5GemmaEncoderModel",
    }


class TMRDataTests(unittest.TestCase):
    def test_halfopen_chronological_unique_caption_candidates(self):
        events = [
            {"start_time": 2, "end_time": 3, "description": "outside"},
            {"start_time": 1.2, "end_time": 1.8, "description": "return"},
            {"start_time": 0.5, "end_time": 1.2, "description": "  step   forward "},
            {"start_time": 0.0, "end_time": 0.5, "description": "step forward"},
            {"start_time": 1.8, "end_time": 2, "description": "step forward"},
        ]
        captions, indices = collect_caption_candidates(events, 0, 20, 10)
        self.assertEqual(captions, ["step forward", "return"])
        self.assertEqual(indices, [3, 2, 1, 4])
        self.assertEqual(collect_caption_candidates(events, 30, 40, 10), ([], []))
        # This fragment spans only part of a frame, so no valid frame overlaps.
        self.assertEqual(collect_caption_candidates([{ "start_time": 0, "end_time": .09, "description": "tiny"}], 0, 10, 10), ([], []))

    def test_join_preserves_splits_and_excludes_unannotated_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            paired = build_paired_index(root, annotations)
            self.assertEqual(paired["split_counts"], {"train": 2, "val": 1, "test": 1})
            self.assertEqual(paired["filtered_counts"], {"train": 1, "val": 0, "test": 0})
            self.assertEqual(len(paired["catalog"]), 2)
            self.assertEqual(paired["records"][0]["motion_path"], "motions/train/train-0.npy")
            limited = build_paired_index(root, annotations, 1)
            self.assertEqual(limited["split_counts"]["train"], 1)
            self.assertNotEqual(limited["paired_index_sha256"], paired["paired_index_sha256"])

    def test_cache_roundtrip_and_padding_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            cache = Path(temporary) / "cache"
            with patch("experiment.tmr.features.load_text_backbone", side_effect=_fake_text_backbone) as loader:
                metadata = prepare_caches(dataset_root=root, annotations_path=annotations, cache_root=cache, text_model="fake", device="cpu")
                prepare_caches(dataset_root=root, annotations_path=annotations, cache_root=cache, text_model="fake", device="cpu")
                self.assertEqual(loader.call_count, 1)
            # Neither model loader may be reached during training data loading.
            with patch("experiment.tmr.features.load_text_backbone", side_effect=AssertionError), patch("experiment.tmr.features.load_frozen_encoder", side_effect=AssertionError):
                datasets, actual = load_prepared_datasets(root, annotations, cache, "raw", text_model="fake")
            self.assertEqual(actual, metadata)
            sample = datasets["train"][0]
            self.assertTrue(torch.equal(sample["motion_tokens"], torch.ones(4, 3)))
            self.assertEqual(sample["text_tokens"].dtype, torch.float32)
            batch = collate_pairs([sample, datasets["train"][1]])
            self.assertEqual(batch["motion_mask"].sum(dim=1).tolist(), [4, 2])
            self.assertEqual(batch["text_mask"].sum(dim=1).tolist(), [3, 4])
            self.assertEqual(batch["start_frames"].dtype, torch.long)
            self.assertTrue((batch["motion_tokens"][1, 2:] == 0).all())
            self.assertEqual(np.load(cache / "text/values.npy", mmap_mode="r").dtype, np.uint16)

    def test_training_samples_one_cached_candidate_and_evaluation_is_fixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            items = [json.loads(line) for line in annotations.read_text().splitlines()]
            for item in items:
                if item["filename"] != "train-2":
                    event = item["events"][0]
                    item["events"].extend([
                        {**event, "description": "walk steadily forward"},
                        dict(event),  # Duplicate text must not increase its sampling weight.
                    ])
            annotations.write_text("".join(json.dumps(item) + "\n" for item in items))
            cache = Path(temporary) / "cache"
            with patch("experiment.tmr.features.load_text_backbone", side_effect=_fake_text_backbone):
                prepare_caches(dataset_root=root, annotations_path=annotations, cache_root=cache, text_model="fake", device="cpu")
            datasets, _ = load_prepared_datasets(root, annotations, cache, "raw", text_model="fake")
            train = datasets["train"]
            candidates = train.caption_candidates[0]
            self.assertEqual(len(candidates), 2)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(18)
                samples = [train[0] for _ in range(32)]
                selected = [sample["caption_id"] for sample in samples]
                self.assertEqual(set(selected), set(candidates))
                torch.manual_seed(18)
                self.assertEqual([train[0]["caption_id"] for _ in range(32)], selected)
            for sample in samples:
                torch.testing.assert_close(sample["text_tokens"], train.text_bank[sample["caption_id"]])
            self.assertEqual({len(sample["text_tokens"]) for sample in samples}, {3, 4})
            for split in ("val", "test"):
                self.assertFalse(datasets[split].sample_captions)
                self.assertEqual(len(datasets[split].caption_candidates[0]), 2)
                self.assertEqual({datasets[split][0]["caption_id"] for _ in range(16)}, {datasets[split].caption_ids[0]})
            batch = collate_pairs(samples[:2])
            self.assertEqual(batch["caption_candidate_ids"], [candidates, candidates])
            marker = cache / "raw/prepared.json"
            metadata = json.loads(marker.read_text())
            metadata["format_version"] = 1
            marker.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "individual caption candidates"):
                load_prepared_datasets(root, annotations, cache, "raw", text_model="fake")

    def test_stale_annotations_stats_and_token_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            cache = Path(temporary) / "cache"
            with patch("experiment.tmr.features.load_text_backbone", side_effect=_fake_text_backbone):
                prepare_caches(dataset_root=root, annotations_path=annotations, cache_root=cache, text_model="fake", device="cpu")
            with self.assertRaisesRegex(ValueError, "stale"):
                load_prepared_datasets(root, annotations, cache, "raw", text_model="fake", max_text_length=128)
            previous = annotations.read_text()
            annotations.write_text(previous + "\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                load_prepared_datasets(root, annotations, cache, "raw", text_model="fake")
            annotations.write_text(previous)
            np.save(root / "stats/mean.npy", np.zeros(3, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "stale"):
                load_prepared_datasets(root, annotations, cache, "raw", text_model="fake")

    def test_jepa_spatial_mean_and_patch_lengths_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            cache = Path(temporary) / "cache"
            checkpoint = Path(temporary) / "jepa.pth"
            checkpoint.write_bytes(b"fake weights")
            np.save(root / "motions/val/val-0.npy", np.full((4, 3), 8, dtype=np.float32))
            np.save(root / "motions/test/test-0.npy", np.full((4, 3), 12, dtype=np.float32))
            encoder = _SpatialEncoder()
            info = {
                "num_frames": 4, "fps": 2, "motion_dim": 3, "feature_dim": 3,
                "token_num_frames": 2, "model_name": "fake-spatial",
            }
            with patch("experiment.tmr.features.load_text_backbone", side_effect=_fake_text_backbone), patch("experiment.tmr.features.load_frozen_encoder", return_value=(encoder, {}, info)):
                prepare_caches(
                    dataset_root=root, annotations_path=annotations, cache_root=cache,
                    input_source="jepa", jepa_checkpoint=checkpoint,
                    stats_path=root / "stats", text_model="fake", device="cpu",
                )
            with patch("experiment.tmr.features.load_frozen_encoder", side_effect=AssertionError):
                datasets, metadata = load_prepared_datasets(
                    root, annotations, cache, "jepa", checkpoint,
                    stats_path=root / "stats", text_model="fake",
                )
            self.assertTrue(torch.equal(datasets["train"].motion_bank["train-0"], torch.full((2, 3), 3.0)))
            self.assertEqual(len(datasets["train"][1]["motion_tokens"]), 1)
            # Three valid train tokens (3, 3, 3.5), excluding padding and the
            # captionless train tail. Validation/test have different means.
            mean, std = 19 / 6, np.sqrt(1 / 18)
            np.testing.assert_allclose(np.load(cache / "jepa/stats/mean.npy"), mean)
            np.testing.assert_allclose(np.load(cache / "jepa/stats/std.npy"), std)
            self.assertEqual(metadata["jepa_feature_stats"]["num_tokens"], 3)
            for split, value in (("train", 3), ("val", 5), ("test", 7)):
                torch.testing.assert_close(
                    datasets[split][0]["motion_tokens"],
                    torch.full((2, 3), (value - mean) / std),
                )
            tokens = torch.cat([datasets["train"][i]["motion_tokens"] for i in range(2)])
            torch.testing.assert_close(tokens.mean(0), torch.zeros(3), atol=1e-6, rtol=0)
            torch.testing.assert_close(tokens.std(0, correction=0), torch.ones(3))
            batch = collate_pairs([datasets["train"][0], datasets["train"][1]])
            self.assertTrue((batch["motion_tokens"][1, 1:] == 0).all())
            checkpoint.write_bytes(b"modified weights")
            with self.assertRaisesRegex(ValueError, "stale"):
                load_prepared_datasets(root, annotations, cache, "jepa", checkpoint, text_model="fake")

    def test_jepa_zero_valid_patch_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            checkpoint = Path(temporary) / "jepa.pth"
            checkpoint.write_bytes(b"fake weights")
            info = {"num_frames": 4, "fps": 2, "motion_dim": 3, "feature_dim": 3, "token_num_frames": 1}
            with patch("experiment.tmr.features.load_text_backbone", side_effect=_fake_text_backbone), patch("experiment.tmr.features.load_frozen_encoder", return_value=(_SpatialEncoder(4), {}, info)):
                with self.assertRaisesRegex(ValueError, "zero valid"):
                    prepare_caches(
                        dataset_root=root, annotations_path=annotations, cache_root=Path(temporary) / "cache",
                        input_source="jepa", jepa_checkpoint=checkpoint,
                        stats_path=root / "stats", text_model="fake", device="cpu",
                    )

    def test_motion_validation_happens_before_text_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            arguments = dict(dataset_root=root, annotations_path=annotations,
                             cache_root=Path(temporary) / "cache", text_model="fake", device="cpu")
            with patch("experiment.tmr.features.load_text_backbone", side_effect=AssertionError) as text_loader:
                with self.assertRaisesRegex(ValueError, "jepa-checkpoint"):
                    prepare_caches(**arguments, input_source="jepa")
                with self.assertRaisesRegex(ValueError, "checkpoint_key"):
                    prepare_caches(**arguments, checkpoint_key="invalid")
                np.save(root / "stats/mean.npy", np.zeros(2, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "Statistics must have shape"):
                    prepare_caches(**arguments)
                self.assertEqual(text_loader.call_count, 0)

    def test_empty_jepa_split_and_unsupported_prepared_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            annotations = _write_dataset(root)
            entries = [json.loads(line) for line in annotations.read_text().splitlines()]
            for entry in entries:
                if entry["filename"] == "test-0":
                    entry["events"] = []
            annotations.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
            checkpoint = Path(temporary) / "jepa.pth"
            checkpoint.write_bytes(b"fake weights")
            cache = Path(temporary) / "cache"
            info = {"num_frames": 4, "fps": 2, "motion_dim": 3, "feature_dim": 3, "token_num_frames": 2}
            with patch("experiment.tmr.features.load_text_backbone", side_effect=_fake_text_backbone), patch("experiment.tmr.features.load_frozen_encoder", return_value=(_SpatialEncoder(), {}, info)):
                prepare_caches(dataset_root=root, annotations_path=annotations, cache_root=cache,
                               input_source="jepa", jepa_checkpoint=checkpoint,
                               stats_path=root / "stats", text_model="fake", device="cpu")
            datasets, _ = load_prepared_datasets(root, annotations, cache, "jepa", checkpoint, text_model="fake")
            self.assertEqual(len(datasets["test"]), 0)
            self.assertEqual(RaggedTokenBank(cache / "jepa/test").keys, [])
            marker = cache / "jepa/prepared.json"
            metadata = json.loads(marker.read_text())
            metadata["format_version"] = 999
            marker.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_prepared_datasets(root, annotations, cache, "jepa", checkpoint, text_model="fake")

    def test_text_model_type_and_access_errors_are_actionable(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("Optional Transformers extraction dependencies are unavailable")
        with patch.object(transformers.AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type="bert")):
            with self.assertRaisesRegex(ValueError, "model_type='t5gemma'"):
                load_text_backbone("wrong-model", None, torch.device("cpu"))
        with patch.object(transformers.AutoConfig, "from_pretrained", side_effect=OSError("gated")):
            with self.assertRaisesRegex(RuntimeError, "hf auth login"):
                load_text_backbone("google/test-gated-model", None, torch.device("cpu"))

    def test_incomplete_bank_and_zero_lengths_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bank"
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                RaggedTokenBank(path)
            with self.assertRaisesRegex(ValueError, "positive"):
                write_ragged_bank(path, ["a"], [0], 3, {}, [])
            with self.assertRaisesRegex(ValueError, "nonempty"):
                collate_pairs([{"motion_tokens": torch.zeros(0, 3), "text_tokens": torch.ones(1, 3)}])

    def test_conditionally_generated_checkpoint_loads_encoder_only(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import PreTrainedTokenizerFast, T5GemmaConfig, T5GemmaForConditionalGeneration, T5GemmaModuleConfig
        except ImportError:
            self.skipTest("Optional Transformers extraction dependencies are unavailable")
        module = T5GemmaModuleConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=32, layer_types=["full_attention"],
            query_pre_attn_scalar=8,
        )
        config = T5GemmaConfig(encoder=module.to_dict(), decoder=module.to_dict(), vocab_size=16)
        model = T5GemmaForConditionalGeneration(config).eval()
        backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1, "walk": 2}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary)
            tokenizer.save_pretrained(temporary)
            encoder, loaded_tokenizer, info = load_text_backbone(temporary, None, torch.device("cpu"))
            self.assertFalse(encoder.config.is_encoder_decoder)
            self.assertFalse(hasattr(encoder, "decoder"))
            self.assertFalse(hasattr(encoder, "lm_head"))
            self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
            for name, parameter in encoder.encoder.named_parameters():
                self.assertTrue(torch.equal(parameter, dict(model.model.encoder.named_parameters())[name].bfloat16()))
            inputs = loaded_tokenizer(["walk walk"], return_tensors="pt")
            with torch.inference_mode():
                output = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            self.assertEqual(tuple(output.shape), (1, 2, info["feature_dim"]))


if __name__ == "__main__":
    unittest.main()
