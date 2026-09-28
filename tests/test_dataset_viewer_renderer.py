from __future__ import annotations

import numpy as np
import torch

from visualization.dataset_viewer import MotionRenderer


class _FakeSkin:
    def __init__(self) -> None:
        self.vertices = torch.zeros((2, 3), dtype=torch.float32)
        self.calls: list[int] = []

    def pose(self, rotations: torch.Tensor, positions: torch.Tensor) -> np.ndarray:
        frame = int(positions[0, 0].item())
        self.calls.append(frame)
        return np.full((2, 3), frame, dtype=np.float32)


class _FakeMesh:
    def __init__(self) -> None:
        self.visible = False
        self.vertices = np.full((2, 3), -1, dtype=np.float32)


class _FakeScene:
    def __init__(self) -> None:
        self.meshes: list[_FakeMesh] = []

    def add_mesh_simple(self, _name: str, **kwargs) -> _FakeMesh:
        mesh = _FakeMesh()
        mesh.vertices = kwargs["vertices"]
        mesh.visible = kwargs["visible"]
        self.meshes.append(mesh)
        return mesh


class _FakeClient:
    def __init__(self) -> None:
        self.scene = _FakeScene()


class _FakeSkeletonRenderer:
    def update(self, points: np.ndarray, contact_indices=None) -> None:
        self.points = points
        self.contact_indices = contact_indices


def _renderer() -> MotionRenderer:
    renderer = MotionRenderer.__new__(MotionRenderer)
    renderer.motion = {
        "global_rot_mats": torch.eye(3).repeat(3, 77, 1, 1),
        "posed_joints": torch.zeros((3, 77, 3), dtype=torch.float32),
        "foot_contacts": torch.zeros((3, 4), dtype=torch.bool),
    }
    renderer.motion["posed_joints"][:, 0, 0] = torch.arange(3)
    renderer.skin = _FakeSkin()
    renderer.mesh = _FakeMesh()
    renderer.shaded_skeleton = _FakeSkeletonRenderer()
    renderer.foot_indices = np.arange(4)
    renderer.frame = 0
    renderer._mesh_visible = False
    renderer._mesh_vertices = None
    return renderer


def test_hidden_mesh_skips_skinning_and_vertex_updates() -> None:
    renderer = _renderer()

    renderer.set_frame(1, show_contacts=False)

    assert renderer.skin.calls == []
    np.testing.assert_array_equal(renderer.mesh.vertices, np.full((2, 3), -1))


def test_showing_mesh_caches_every_frame_once() -> None:
    renderer = _renderer()
    renderer.set_frame(1, show_contacts=False)

    renderer.set_mesh_visible(True)
    assert renderer.skin.calls == [0, 1, 2]
    np.testing.assert_array_equal(renderer.mesh.vertices, np.full((2, 3), 1))

    renderer.set_frame(2, show_contacts=False)
    renderer.set_frame(0, show_contacts=False)
    assert renderer.skin.calls == [0, 1, 2]
    np.testing.assert_array_equal(renderer.mesh.vertices, np.zeros((2, 3)))

    renderer.set_mesh_visible(False)
    renderer.set_frame(2, show_contacts=False)
    assert renderer.skin.calls == [0, 1, 2]
    np.testing.assert_array_equal(renderer.mesh.vertices, np.zeros((2, 3)))


def test_hidden_mesh_is_created_lazily() -> None:
    renderer = _renderer()
    renderer.mesh = None
    renderer.client = _FakeClient()
    renderer.skin.faces = np.zeros((1, 3), dtype=np.int32)
    renderer._mesh_opacity = 0.9

    renderer.set_frame(2, show_contacts=False)
    assert renderer.client.scene.meshes == []
    assert renderer.skin.calls == []

    renderer.set_mesh_visible(True)
    assert len(renderer.client.scene.meshes) == 1
    assert renderer.skin.calls == [0, 1, 2]
    np.testing.assert_array_equal(renderer.mesh.vertices, np.full((2, 3), 2))

    renderer.set_mesh_visible(False)
    renderer.set_frame(2, show_contacts=False)
    assert renderer.skin.calls == [0, 1, 2]
    np.testing.assert_array_equal(renderer.mesh.vertices, np.full((2, 3), 2))
