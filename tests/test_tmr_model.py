"""Token-preserving readouts, ambiguity-aware contrastive loss and retrieval."""

from __future__ import annotations

import copy
import math
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from experiment.tmr.losses import symmetric_multi_positive_info_nce
from experiment.tmr.model import AlignmentConfig, TextMotionAlignment
from experiment.tmr.retrieval import evaluate_retrieval


class AlignmentModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        self.config = AlignmentConfig(
            text_dim=12, motion_dim=7, embed_dim=16, depth=2,
            num_heads=4, ff_dim=32, dropout=0.0,
        )

    def test_padding_invariance_and_variable_sequence_lengths(self):
        model = TextMotionAlignment(self.config).eval()
        text = torch.randn(3, 5, 12)
        motion = torch.randn(3, 7, 7)
        text_mask = torch.arange(5)[None, :] < torch.tensor([3, 4, 1])[:, None]
        motion_mask = torch.arange(7)[None, :] < torch.tensor([2, 7, 5])[:, None]
        reference = model(motion, motion_mask, text, text_mask)
        altered = model(
            motion.masked_fill(~motion_mask[..., None], float("inf")), motion_mask,
            text.masked_fill(~text_mask[..., None], float("nan")), text_mask,
        )
        for expected, actual in zip(reference, altered):
            torch.testing.assert_close(expected, actual)
            torch.testing.assert_close(actual.norm(dim=-1), torch.ones(3))
            self.assertEqual(actual.shape, (3, 16))
        # Extra padding must not change token positions or CLS readout.
        extended_text = torch.cat((text, torch.full((3, 4, 12), float("nan"))), dim=1)
        extended_mask = F.pad(text_mask, (0, 4))
        torch.testing.assert_close(reference[1], model.encode_text(extended_text, extended_mask))
        bf16_output = model.encode_text(text.bfloat16(), text_mask)
        self.assertTrue(torch.isfinite(bf16_output).all())

    def test_cls_and_independent_heads_train_with_frozen_external_backbones(self):
        model = TextMotionAlignment(self.config)
        text_backbone = nn.Linear(5, 12).eval().requires_grad_(False)
        motion_backbone = nn.Linear(3, 7).eval().requires_grad_(False)
        frozen_text = copy.deepcopy(text_backbone.state_dict())
        frozen_motion = copy.deepcopy(motion_backbone.state_dict())
        before_cls = model.text_encoder.cls_token.detach().clone()
        text = text_backbone(torch.randn(4, 3, 5))
        motion = motion_backbone(torch.randn(4, 6, 3))
        mask_text = torch.ones(4, 3, dtype=torch.bool)
        mask_motion = torch.ones(4, 6, dtype=torch.bool)
        z_motion, z_text = model(motion, mask_motion, text, mask_text)
        loss = symmetric_multi_positive_info_nce(
            z_motion, z_text, ["a", "b", "c", "d"], ["s1", "s2", "s3", "s4"],
            [0] * 4, [10] * 4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss.backward()
        for encoder in (model.text_encoder, model.motion_encoder):
            self.assertGreater(float(encoder.cls_token.grad.abs().sum()), 0)
            self.assertGreater(float(encoder.input_projection.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before_cls, model.text_encoder.cls_token))
        self.assertNotEqual(
            model.text_encoder.blocks[0].attn.qkv.weight.data_ptr(),
            model.motion_encoder.blocks[0].attn.qkv.weight.data_ptr(),
        )
        for backbone, before in ((text_backbone, frozen_text), (motion_backbone, frozen_motion)):
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in backbone.parameters()))
            for key, value in backbone.state_dict().items():
                torch.testing.assert_close(value, before[key])

    def test_empty_valid_sequence_and_bad_features_are_rejected(self):
        model = TextMotionAlignment(self.config)
        text = torch.randn(2, 3, 12)
        active = torch.ones(2, 3, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "valid token"):
            model.encode_text(text, torch.zeros_like(active))
        with self.assertRaisesRegex(ValueError, "valid_mask"):
            model.encode_text(text, torch.ones(2, 4, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "finite"):
            model.encode_text(text.masked_fill(active[..., None], float("nan")), active)
        with self.assertRaisesRegex(ValueError, "Expected tokens"):
            model.encode_text(torch.randn(2, 3, 4), active)
        with self.assertRaisesRegex(ValueError, "valid token"):
            model.encode_motion(torch.empty(2, 0, 7), torch.empty(2, 0, dtype=torch.bool))

    def test_config_validation_and_state_roundtrip(self):
        for changes in (
            {"depth": 0}, {"embed_dim": 15}, {"num_heads": 3},
            {"text_dim": -1}, {"dropout": 1.0}, {"dropout": float("nan")},
        ):
            values = vars(self.config) | changes
            with self.assertRaises(ValueError):
                AlignmentConfig(**values)
        model = TextMotionAlignment(self.config).eval()
        restored = TextMotionAlignment(AlignmentConfig(**vars(self.config))).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        tokens, active = torch.randn(2, 4, 7), torch.ones(2, 4, dtype=torch.bool)
        torch.testing.assert_close(model.encode_motion(tokens, active), restored.encode_motion(tokens, active))


class ContrastiveLossTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_diagonal_reduces_to_symmetric_cross_entropy(self):
        torch.manual_seed(18)
        motion = F.normalize(torch.randn(4, 5), dim=-1).requires_grad_()
        text = F.normalize(torch.randn(4, 5), dim=-1).requires_grad_()
        loss = symmetric_multi_positive_info_nce(
            motion, text, list("abcd"), list("wxyz"), [0] * 4, [10] * 4, temperature=0.2,
        )
        scores = motion @ text.T / 0.2
        expected = (F.cross_entropy(scores, torch.arange(4)) + F.cross_entropy(scores.T, torch.arange(4))) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(motion.grad).all())
        self.assertTrue(torch.isfinite(text.grad).all())

    def test_duplicate_captions_sum_all_positive_probability_mass(self):
        embeddings = torch.zeros(3, 2)
        loss = symmetric_multi_positive_info_nce(
            embeddings, embeddings, ["same", "same", "other"], ["s1", "s2", "s3"],
            [0] * 3, [10] * 3,
        )
        expected = (2 * math.log(3 / 2) + math.log(3)) / 3
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_candidate_captions_are_positives_even_when_another_candidate_was_sampled(self):
        embeddings = torch.zeros(3, 2, requires_grad=True)
        loss = symmetric_multi_positive_info_nce(
            embeddings, embeddings, ["a", "b", "c"], ["s1", "s2", "s3"],
            [0] * 3, [10] * 3, caption_candidate_ids=[["a", "b"], ["b"], ["c"]],
        )
        # Motion 0 accepts a/b, text b accepts motions 0/1; other positives are diagonal.
        expected = (math.log(3 / 2) + 2 * math.log(3)) / 3
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        with self.assertRaisesRegex(ValueError, "sampled caption"):
            symmetric_multi_positive_info_nce(
                embeddings, embeddings, ["a", "b", "c"], ["s1", "s2", "s3"],
                [0] * 3, [10] * 3, caption_candidate_ids=[["b"], ["b"], ["c"]],
            )

    def test_overlap_negatives_excluded_and_touching_intervals_retained(self):
        embeddings = torch.zeros(3, 2)
        overlap_loss = symmetric_multi_positive_info_nce(
            embeddings, embeddings, list("abc"), ["s", "s", "other"],
            [0, 5, 0], [10, 15, 10],
        )
        expected = (2 * math.log(2) + math.log(3)) / 3
        self.assertAlmostEqual(float(overlap_loss), expected, places=6)
        touching_loss = symmetric_multi_positive_info_nce(
            embeddings[:2], embeddings[:2], list("ab"), ["s", "s"], [0, 10], [10, 20],
        )
        self.assertAlmostEqual(float(touching_loss), math.log(2), places=6)
        all_ambiguous = symmetric_multi_positive_info_nce(
            embeddings[:2], embeddings[:2], list("ab"), ["s", "s"], [0, 5], [10, 15],
        )
        self.assertEqual(float(all_ambiguous), 0)

    def test_positive_precedence_and_no_negative_batches_have_finite_gradients(self):
        for captions, sources, starts, ends in (
            (["same", "same"], ["s", "s"], [0, 5], [10, 15]),
            (["a", "b"], ["s", "s"], [0, 5], [10, 15]),
            (["single"], ["s"], [0], [10]),
        ):
            size = len(captions)
            motion = torch.randn(size, 6, requires_grad=True)
            text = torch.randn(size, 6, requires_grad=True)
            loss = symmetric_multi_positive_info_nce(motion, text, captions, sources, starts, ends)
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(motion.grad).all())
            self.assertTrue(torch.isfinite(text.grad).all())

    def test_loss_float32_inside_autocast_and_validation(self):
        embeddings = torch.eye(2).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = symmetric_multi_positive_info_nce(
                embeddings, embeddings, list("ab"), list("st"), [0, 0], [10, 10],
            )
        self.assertEqual(loss.dtype, torch.float32)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "temperature"):
                symmetric_multi_positive_info_nce(
                    embeddings, embeddings, list("ab"), list("st"), [0, 0], [10, 10], temperature,
                )
        with self.assertRaisesRegex(ValueError, "integer frame"):
            symmetric_multi_positive_info_nce(
                embeddings, embeddings, list("ab"), list("st"), [0.5, 1], [10, 10],
            )
        with self.assertRaisesRegex(ValueError, "start < end"):
            symmetric_multi_positive_info_nce(
                embeddings, embeddings, list("ab"), list("st"), [0, 10], [10, 10],
            )


def _dense_reference(motion, text, motion_ids, text_ids):
    scores = motion @ text.T
    motion_ranks, text_ranks = [], []
    candidates = [{value} if isinstance(value, str) else set(value) for value in motion_ids]
    for i, captions in enumerate(candidates):
        order = scores[i].argsort(descending=True, stable=True).tolist()
        motion_ranks.append(next(j + 1 for j, index in enumerate(order) if text_ids[index] in captions))
    for i, caption in enumerate(text_ids):
        order = scores[:, i].argsort(descending=True, stable=True).tolist()
        text_ranks.append(next(j + 1 for j, index in enumerate(order) if caption in candidates[index]))
    result = {}
    for prefix, ranks in (("m2t", motion_ranks), ("t2m", text_ranks)):
        values = torch.tensor(ranks, dtype=torch.float32)
        for k in (1, 5, 10):
            result[f"{prefix}_r{k}"] = float((values <= k).float().mean())
        result[f"{prefix}_medr"] = float(values.quantile(0.5))
    result["mean_r1"] = (result["m2t_r1"] + result["t2m_r1"]) / 2
    return result


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_chunked_matches_complete_gallery_dense_reference(self):
        torch.manual_seed(56)
        motion = F.normalize(torch.randn(17, 7), dim=-1)
        text = F.normalize(torch.randn(11, 7), dim=-1)
        text_ids = [f"caption-{i}" for i in range(11)]
        motion_ids = [text_ids[i % 11] for i in range(17)]
        expected = _dense_reference(motion, text, motion_ids, text_ids)
        for chunk in (1, 2, 5, 512):
            actual = evaluate_retrieval(motion, text, motion_ids, text_ids, chunk_size=chunk)
            self.assertEqual(actual, expected)

    def test_duplicates_and_candidate_index_ties(self):
        # All similarities tie: deterministic ordering is the candidate index.
        motion, text = torch.ones(4, 2), torch.ones(2, 2)
        motion_ids, text_ids = ["b", "a", "b", "a"], ["a", "b"]
        for chunk in (1, 2, 3):
            metrics = evaluate_retrieval(motion, text, motion_ids, text_ids, chunk)
            self.assertEqual(metrics["t2m_r1"], 0.5)
            self.assertEqual(metrics["m2t_r1"], 0.5)
            self.assertEqual(metrics["t2m_medr"], 1.5)
            self.assertEqual(metrics["m2t_medr"], 1.5)
            self.assertEqual(metrics["mean_r1"], 0.5)

    def test_multiple_captions_per_motion_match_dense_gallery_and_ties(self):
        motion_ids = [["b", "a"], ["b", "c"], ["c"]]
        text_ids = ["a", "b", "c"]
        for motion, text in (
            (F.normalize(torch.randn(3, 7), dim=-1), F.normalize(torch.randn(3, 7), dim=-1)),
            (torch.ones(3, 2), torch.ones(3, 2)),
        ):
            expected = _dense_reference(motion, text, motion_ids, text_ids)
            for chunk in (1, 2, 512):
                self.assertEqual(evaluate_retrieval(motion, text, motion_ids, text_ids, chunk), expected)

    def test_any_correct_caption_motion_is_positive(self):
        motion = torch.tensor([[0.1, 0.9], [1.0, 0.0], [0.0, 1.0]])
        text = torch.eye(2)
        metrics = evaluate_retrieval(motion, text, ["a", "a", "b"], ["a", "b"], 1)
        self.assertEqual(metrics["t2m_r1"], 1)
        self.assertAlmostEqual(metrics["m2t_r1"], 2 / 3, places=6)

    def test_unique_gallery_positive_presence_and_finite_validation(self):
        embeddings = torch.eye(2)
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_retrieval(embeddings, embeddings, ["a", "a"], ["a", "a"])
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate_retrieval(embeddings, embeddings, ["a", "a"], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_retrieval(embeddings * float("nan"), embeddings, ["a", "b"], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            evaluate_retrieval(embeddings, embeddings, ["a", "b"], ["a", "b"], chunk_size=0)


if __name__ == "__main__":
    unittest.main()
