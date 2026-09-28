"""Tests for dependency-free TensorBoard metric emission."""

from __future__ import annotations

import unittest

from train import (
    _write_tensorboard_interval,
    _write_tensorboard_linear_probe,
    _write_tensorboard_online_metrics,
)


class _Writer:
    def __init__(self):
        self.scalars = {}
        self.flush_count = 0

    def add_scalar(self, name, value, step):
        self.scalars[name] = (value, step)

    def flush(self):
        self.flush_count += 1


class TensorBoardLoggingTest(unittest.TestCase):
    def test_interval_values_are_emitted_at_the_global_step(self):
        writer = _Writer()
        _write_tensorboard_interval(
            writer,
            global_step=100,
            epoch=2,
            loss=0.25,
            learning_rate=1.0e-4,
            weight_decay=0.04,
            time_ms=12.5,
            memory_mib=1024.0,
            grad_first=0.1,
            grad_last=0.2,
            grad_average=0.15,
        )
        self.assertEqual(writer.scalars["train/loss"], (0.25, 100))
        self.assertEqual(writer.scalars["train/iteration_time_ms"], (12.5, 100))
        self.assertEqual(writer.scalars["train/epoch"], (2.0, 100))
        self.assertEqual(writer.flush_count, 1)

    def test_none_writer_is_a_noop(self):
        _write_tensorboard_interval(
            None,
            global_step=1,
            epoch=1,
            loss=1.0,
            learning_rate=1.0,
            weight_decay=1.0,
            time_ms=1.0,
            memory_mib=0.0,
            grad_first=0.0,
            grad_last=0.0,
            grad_average=0.0,
        )

    def test_linear_probe_values_are_emitted_at_the_pretraining_step(self):
        writer = _Writer()
        _write_tensorboard_linear_probe(
            writer,
            global_step=300,
            summary={
                "best_epoch": 17,
                "best_val": {"top1_accuracy": 0.8},
                "test": {"top1_accuracy": 0.75},
            },
            best_val_top1=0.82,
        )
        self.assertEqual(
            writer.scalars["linear_probe/val_top1_accuracy"], (0.8, 300)
        )
        self.assertEqual(
            writer.scalars["linear_probe/test_top1_accuracy"], (0.75, 300)
        )
        self.assertEqual(
            writer.scalars["linear_probe/best_val_top1_accuracy"], (0.82, 300)
        )
        self.assertEqual(
            writer.scalars["linear_probe/probe_best_epoch"], (17.0, 300)
        )
        self.assertEqual(writer.flush_count, 1)

    def test_nested_online_metrics_are_flattened(self):
        writer = _Writer()
        _write_tensorboard_online_metrics(
            writer,
            global_step=50,
            summary={
                "num_samples": 16,
                "representation": {"body": {"rankme": 3.5}},
                "heldout_jepa": {"prediction_gain": 0.4},
            },
        )
        self.assertEqual(
            writer.scalars["online_metrics/representation/body/rankme"],
            (3.5, 50),
        )
        self.assertEqual(
            writer.scalars["online_metrics/heldout_jepa/prediction_gain"],
            (0.4, 50),
        )


if __name__ == "__main__":
    unittest.main()
