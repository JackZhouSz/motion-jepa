"""Regression tests for the independent MotionJEPA motion package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from motion_rep import MotionJEPAMotionRep
from skeleton import SOMASkeleton30, parse_bvh_motion
from visualization import SOMASkin


class MotionJEPATest(unittest.TestCase):
    def test_bvh_parser_channel_order_scale_and_fps(self):
        bvh = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
}
MOTION
Frames: 2
Frame Time: 0.008333
100 200 300 0 0 0
110 220 330 90 0 0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.bvh"
            path.write_text(bvh, encoding="utf-8")
            rotations, positions, fps = parse_bvh_motion(path)
        self.assertEqual(tuple(rotations.shape), (2, 1, 3, 3))
        np.testing.assert_allclose(positions.numpy(), [[1, 2, 3], [1.1, 2.2, 3.3]], atol=1e-6)
        self.assertAlmostEqual(fps, 120.00480019200768)

    def test_golden_representation_and_decode(self):
        fixture = Path(__file__).with_name("assets") / "motion_jepa_golden.npz"
        with np.load(fixture, allow_pickle=False) as data:
            local = torch.from_numpy(data["local_rotations"])
            root = torch.from_numpy(data["root_positions"])
            expected = torch.from_numpy(data["features"])
        representation = MotionJEPAMotionRep(SOMASkeleton30(), fps=120)
        actual = representation(local, root, to_canonicalize=True)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        decoded = representation.inverse(actual)
        self.assertEqual(tuple(decoded["posed_joints"].shape), (len(local), 30, 3))
        self.assertTrue(torch.isfinite(decoded["posed_joints"]).all())

    def test_soma_skin_frame(self):
        fixture = Path(__file__).with_name("assets") / "motion_jepa_golden.npz"
        with np.load(fixture, allow_pickle=False) as data:
            local = torch.from_numpy(data["local_rotations"])
            root = torch.from_numpy(data["root_positions"])
        representation = MotionJEPAMotionRep(SOMASkeleton30(), fps=120)
        output = representation.skeleton.expand_output(
            representation.inverse(
                representation(local, root, to_canonicalize=True)
            )
        )
        skin = SOMASkin(representation.skeleton)
        vertices = skin.pose(output["global_rot_mats"][0], output["posed_joints"][0])
        self.assertEqual(vertices.shape, (18056, 3))
        self.assertEqual(skin.faces.shape, (36108, 3))
        self.assertTrue(np.isfinite(vertices).all())


if __name__ == "__main__":
    unittest.main()
