"""Checkpoint state restoration tests."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from helper import init_opt
from mask import MaskCollator1D
from model import MotionTransformer1D, MotionTransformerPredictor1D
from train import _load_checkpoint, _save_checkpoint
from utils.schedulers import LinearMomentumSchedule


class CheckpointTest(unittest.TestCase):
    def test_full_state_restores_rng_schedules_and_masks(self):
        torch.manual_seed(7)
        np.random.seed(7)
        random.seed(7)
        encoder = MotionTransformer1D(6, 4, embed_dim=12, depth=1, num_heads=3)
        predictor = MotionTransformerPredictor1D(
            4, 12, 12, depth=1, num_heads=3
        )
        target = copy.deepcopy(encoder)
        optimizer, scaler, lr, wd = init_opt(
            encoder,
            predictor,
            iterations_per_epoch=2,
            start_lr=1.0e-4,
            ref_lr=1.0e-3,
            warmup=0,
            num_epochs=2,
            wd=0.04,
            final_wd=0.4,
        )
        momentum = LinearMomentumSchedule(0.9, 1.0, 4)
        collator = MaskCollator1D(
            4, (0.75, 0.75), (0.25, 0.25), nenc=1, npred=1
        )
        batch = [(torch.zeros(4, 6), 60)]
        lr.step()
        wd.step()
        momentum.step()
        collator(batch)
        saved_encoder = copy.deepcopy(encoder.state_dict())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth.tar"
            _save_checkpoint(
                path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target,
                optimizer=optimizer,
                scaler=scaler,
                lr_scheduler=lr,
                wd_scheduler=wd,
                momentum_scheduler=momentum,
                mask_collator=collator,
                next_epoch=1,
                global_step=2,
                loss=0.5,
                world_size=1,
                rank=0,
                config={"test": True},
                linear_probe_latest={"pretrain_epoch": 1, "test": {"top1_accuracy": 0.7}},
                best_probe_val_top1=0.8,
                best_probe_epoch=1,
                online_metrics_latest={"pretrain_epoch": 1, "rankme": 3.0},
            )
            expected_random = (random.random(), np.random.rand(), torch.rand(()))
            expected_lr = lr.step()
            expected_masks = collator(batch)[1:]

            for parameter in encoder.parameters():
                parameter.data.add_(10)
            random.random()
            np.random.rand()
            torch.rand(())
            lr.step()
            collator(batch)
            linear_probe_state = {}
            online_metrics_state = {}
            next_epoch, global_step = _load_checkpoint(
                path,
                device=torch.device("cpu"),
                encoder=encoder,
                predictor=predictor,
                target_encoder=target,
                optimizer=optimizer,
                scaler=scaler,
                lr_scheduler=lr,
                wd_scheduler=wd,
                momentum_scheduler=momentum,
                mask_collator=collator,
                rank=0,
                world_size=1,
                linear_probe_state=linear_probe_state,
                online_metrics_state=online_metrics_state,
            )
            self.assertEqual((next_epoch, global_step), (1, 2))
            self.assertEqual(linear_probe_state["best_val_top1"], 0.8)
            self.assertEqual(linear_probe_state["best_epoch"], 1)
            self.assertEqual(linear_probe_state["latest"]["pretrain_epoch"], 1)
            self.assertEqual(online_metrics_state["latest"]["rankme"], 3.0)
            for name, value in encoder.state_dict().items():
                torch.testing.assert_close(value, saved_encoder[name])
            actual_random = (random.random(), np.random.rand(), torch.rand(()))
            self.assertEqual(actual_random[0], expected_random[0])
            self.assertEqual(actual_random[1], expected_random[1])
            torch.testing.assert_close(actual_random[2], expected_random[2])
            self.assertEqual(lr.step(), expected_lr)
            actual_masks = collator(batch)[1:]
            for expected_group, actual_group in zip(expected_masks, actual_masks):
                for expected, actual in zip(expected_group, actual_group):
                    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
