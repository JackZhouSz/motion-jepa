"""Export validation and synchronization do not require a live browser or GPU."""

from contextlib import nullcontext
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from experiment.prediction.visualize import ComparisonViewer, _NamespacedScene, load_results


def _export(tmp_path, **overrides):
    arrays = dict(
        sample_ids=np.array(["first", "second"]), lengths=np.array([4, 2]), fps=np.array(30),
        target_motion=np.zeros((2, 4, 366), dtype=np.float32),
        reconstructed_motion=np.zeros((2, 4, 366), dtype=np.float32),
        target_joints=np.zeros((2, 4, 30, 3), dtype=np.float32),
        reconstructed_joints=np.zeros((2, 4, 30, 3), dtype=np.float32),
    )
    arrays.update(overrides)
    path = tmp_path / "reconstructions.npz"
    np.savez(path, **arrays)
    return path


def test_export_load_preserves_ids_lengths_and_raw_data(tmp_path):
    raw = np.ones((2, 4, 366), dtype=np.float32) * 1.7
    results = load_results(_export(tmp_path, reconstructed_motion=raw))
    assert len(results) == 2
    assert results.fps == 30
    assert results.sample_ids.tolist() == ["first", "second"]
    assert results.lengths.tolist() == [4, 2]
    np.testing.assert_array_equal(results.reconstructed_motion, raw)


@pytest.mark.parametrize("overrides,message", [
    ({"lengths": np.array([4, 5])}, "lengths must"),
    ({"fps": np.array(0)}, "fps must"),
    ({"fps": np.array([30, 30])}, "fps must"),
    ({"reconstructed_motion": np.zeros((2, 3, 366), dtype=np.float32)}, "reconstructed_motion must"),
    ({"target_joints": np.full((2, 4, 30, 3), np.nan, dtype=np.float32)}, "target_joints must"),
])
def test_invalid_export_is_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        load_results(_export(tmp_path, **overrides))


class _Scene:
    def __init__(self):
        self.calls = []
        self.world_axes = SimpleNamespace(visible=True)

    def set_up_direction(self, _direction):
        pass

    def _add(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return SimpleNamespace(visible=True)

    add_frame = _add
    add_label = _add
    add_grid = _add
    add_mesh_simple = _add
    add_batched_meshes_simple = _add


def test_existing_renderer_paths_are_namespaced():
    scene = _Scene()
    target = _NamespacedScene(scene, "/prediction/target")
    reconstructed = _NamespacedScene(scene, "/prediction/reconstruction")
    target.add_mesh_simple("/motion_jepa/mesh", vertices=np.zeros((1, 3)))
    reconstructed.add_mesh_simple("/motion_jepa/mesh", vertices=np.zeros((1, 3)))
    assert [name for name, _ in scene.calls] == [
        "/prediction/target/motion_jepa/mesh", "/prediction/reconstruction/motion_jepa/mesh",
    ]


class _Handle:
    def __init__(self, value=None):
        self._value = value
        self.updates = []
        self.clicks = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self._value == value:
            return
        self._value = value
        for callback in self.updates:
            callback(None)

    def on_update(self, callback):
        self.updates.append(callback)
        return callback

    def on_click(self, callback):
        self.clicks.append(callback)
        return callback

    def click(self):
        for callback in self.clicks:
            callback(None)


class _Gui:
    def __init__(self):
        self.handles = {}

    def add_folder(self, *_args, **_kwargs):
        return nullcontext()

    def _add(self, label, value=None):
        handle = _Handle(value)
        self.handles[label] = handle
        return handle

    def add_dropdown(self, label, _options, initial_value):
        return self._add(label, initial_value)

    def add_markdown(self, content):
        return SimpleNamespace(content=content)

    def add_slider(self, label, **kwargs):
        handle = self._add(label, kwargs["initial_value"])
        handle.max = kwargs["max"]
        return handle

    def add_button(self, label):
        return self._add(label)

    def add_button_group(self, label, options):
        return self._add(label, options[0])

    def add_checkbox(self, label, initial_value):
        return self._add(label, initial_value)


class _Server:
    def __init__(self, **_kwargs):
        self.scene = _Scene()
        self.stopped = False

    def on_client_connect(self, _callback):
        pass

    def on_client_disconnect(self, _callback):
        pass

    def stop(self):
        self.stopped = True


class _Renderer:
    def __init__(self, client, motion, *_args):
        self.length = motion["length"]
        self.frame = 0
        self.cleared = False
        self.shaded_skeleton = SimpleNamespace(visible=True)
        self.updated = threading.Event()
        client.scene.add_mesh_simple("/motion_jepa/mesh")

    def clear(self):
        self.cleared = True

    def set_frame(self, frame, show_contacts):
        if self.cleared:
            raise RuntimeError("Removed renderer was updated")
        self.frame = frame
        self.contacts = show_contacts
        if frame > 0:
            self.updated.set()

    def set_mesh_visible(self, value):
        self.mesh_visible = value


def test_both_renderers_share_playback_and_sample_changes(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "viser", SimpleNamespace(ViserServer=_Server))
    monkeypatch.setitem(sys.modules, "visualization.dataset_viewer", SimpleNamespace(MotionRenderer=_Renderer))
    monkeypatch.setattr("experiment.prediction.visualize._decode_motion", lambda motion, fps: {"length": len(motion)})
    viewer = ComparisonViewer(_export(tmp_path), host="localhost", port=0)
    client = SimpleNamespace(client_id=1, scene=_Scene(), camera=SimpleNamespace(), gui=_Gui())
    try:
        viewer._on_connect(client)
        session = viewer.sessions[1]
        parent_positions = dict((name, kwargs["position"]) for name, kwargs in client.scene.calls if name in (
            "/prediction/target", "/prediction/reconstruction",
        ))
        assert parent_positions == {
            "/prediction/target": (-1.5, 0.0, 0.0), "/prediction/reconstruction": (1.5, 0.0, 0.0),
        }
        session.gui["frame"].value = 3
        assert [renderer.frame for renderer in session.renderers] == [3, 3]
        old_renderers = session.renderers
        session.gui["sample"].value = viewer.labels[1]
        assert all(renderer.cleared for renderer in old_renderers)
        assert session.gui["frame"].max == 1
        assert [renderer.frame for renderer in session.renderers] == [0, 0]
        client.gui.handles["Play / Pause"].click()
        assert session.renderers[0].updated.wait(1.0)
        assert session.renderers[1].updated.wait(1.0)
        with viewer.lock:
            assert [renderer.frame for renderer in session.renderers] == [session.gui["frame"].value] * 2
        session.gui["sample"].value = viewer.labels[0]
        assert not session.playing
        assert [renderer.length for renderer in session.renderers] == [4, 4]
        session.gui["contacts"].value = True
        assert all(renderer.contacts for renderer in session.renderers)
    finally:
        viewer.close()
    assert viewer.server.stopped
    assert not viewer._playback_thread.is_alive()
