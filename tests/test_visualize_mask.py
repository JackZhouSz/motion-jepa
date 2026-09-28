"""Mask visualization tests across raw and patchified layouts."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from visualize_mask import mask_statistics, render_mask_png, sample_masks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MaskVisualizationTest(unittest.TestCase):
    def _config(self, name: str) -> dict:
        path = PROJECT_ROOT / "configs" / name
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["data"]["num_frames"] = 12
        return config

    def test_active_trajectory_patch_layout_samples_and_renders(self):
        cases = (
            ("mjepa_patch_2d_tiny_coarse7.yaml", "2d", True, 4, 8),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config_name, kind, patchified, frames, joints in cases:
                sampled = sample_masks(
                    self._config(config_name), seed=3, valid_length=11
                )
                self.assertEqual(sampled.layout.kind, kind)
                self.assertEqual(sampled.layout.patchified, patchified)
                self.assertEqual(sampled.layout.token_num_frames, frames)
                expected_shape = (frames,) if joints is None else (frames, joints)
                for mask in sampled.contexts + sampled.targets:
                    self.assertEqual(mask.shape, expected_shape)
                output = root / f"{config_name}.png"
                stats = render_mask_png(output, sampled, seed=3)
                self.assertTrue(output.is_file())
                with Image.open(output) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 500)
                    self.assertGreater(image.height, 300)
                self.assertEqual(stats["kind"], kind)
                self.assertEqual(stats["patchified"], patchified)
                self.assertEqual(stats["context_target_overlap_cells"], 0)
                self.assertEqual(
                    stats["valid_token_frames"], 11 if not patchified else 3
                )

    def test_seed_and_sample_selection_are_deterministic(self):
        config = self._config("mjepa_patch_2d_tiny_coarse7.yaml")
        first = sample_masks(
            config, seed=9, valid_length=12, batch_size=2, sample_index=1
        )
        second = sample_masks(
            config, seed=9, valid_length=12, batch_size=2, sample_index=1
        )
        for left, right in zip(
            first.contexts + first.targets, second.contexts + second.targets
        ):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(mask_statistics(first), mask_statistics(second))

    def test_cli_writes_patch_2d_png(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mask.png"
            completed = subprocess.run(
                [
                    str(Path("/opt/conda/envs/kimodo/bin/python")),
                    str(PROJECT_ROOT / "visualize_mask.py"),
                    "--config",
                    str(PROJECT_ROOT / "configs/mjepa_patch_2d_tiny_coarse7.yaml"),
                    "--output",
                    str(output),
                    "--seed",
                    "4",
                    "--valid-length",
                    "89",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(output.is_file())
            self.assertIn("Saved mask visualization", completed.stdout)
            self.assertIn('"token_num_joints": 8', completed.stdout)


if __name__ == "__main__":
    unittest.main()
