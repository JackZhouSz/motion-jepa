"""Generation exports and draw selection remain independent of model loading."""

from contextlib import nullcontext
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from experiment.generation.visualize import GenerationComparisonViewer, load_results


def _export(tmp_path, **overrides):
    generated = np.zeros((2, 3, 4, 366), dtype=np.float32)
    generated[:, 1] = 1
    generated[:, 2] = 2
    arrays = dict(
        sample_ids=np.array(["first", "second"]), lengths=np.array([4, 2]), fps=np.array(30),
        target_motion=np.zeros((2, 4, 366), dtype=np.float32), generated_motion=generated,
        target_joints=np.zeros((2, 4, 30, 3), dtype=np.float32),
        generated_joints=np.zeros((2, 3, 4, 30, 3), dtype=np.float32),
        noise_seeds=np.array([[0, 1, 2], [3, 4, 2 ** 64 - 1]], dtype=np.uint64),
        sampling_json=np.array(json.dumps({"steps": 32, "guidance_scale": 1.0})),
    )
    arrays.update(overrides)
    path = tmp_path / "generations.npz"
    np.savez(path, **arrays)
    return path


def test_generation_export_retains_draws_and_uint64_seeds(tmp_path):
    results = load_results(_export(tmp_path))
    assert len(results) == 2
    assert results.num_draws == 3
    assert results.lengths.tolist() == [4, 2]
    assert results.fps == 30
    assert int(results.noise_seeds[1, 2]) == 2 ** 64 - 1
    np.testing.assert_array_equal(results.generated_motion[0, 2], np.full((4, 366), 2))
    assert json.loads(results.sampling_json)["steps"] == 32


@pytest.mark.parametrize("overrides,message", [
    ({"lengths": np.array([4, 5])}, "lengths must"),
    ({"lengths": np.array([4, 0])}, "lengths must"),
    ({"fps": np.array([30, 30])}, "fps must"),
    ({"fps": np.array(0)}, "fps must"),
    ({"generated_motion": np.zeros((2, 0, 4, 366), dtype=np.float32)}, "generated_motion must"),
    ({"generated_joints": np.zeros((2, 2, 4, 30, 3), dtype=np.float32)}, "generated_joints must"),
    ({"target_joints": np.full((2, 4, 30, 3), np.nan, dtype=np.float32)}, "target_joints must"),
    ({"noise_seeds": np.zeros((2, 3), dtype=np.float32)}, "noise_seeds must"),
    ({"noise_seeds": np.full((2, 3), -1, dtype=np.int64)}, "noise_seeds must"),
    ({"sampling_json": np.array("[]")}, "sampling_json must"),
])
def test_invalid_generation_export_is_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        load_results(_export(tmp_path, **overrides))


def test_missing_required_fields_are_reported(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, sample_ids=np.array(["missing"]))
    with pytest.raises(ValueError, match="missing fields"):
        load_results(path)


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
        self.marker = motion["marker"]
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


def test_draw_switch_preserves_shared_frame_and_browser_independence(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "viser", SimpleNamespace(ViserServer=_Server))
    monkeypatch.setitem(sys.modules, "visualization.dataset_viewer", SimpleNamespace(MotionRenderer=_Renderer))
    monkeypatch.setattr("experiment.generation.visualize._decode_motion", lambda motion, fps: {
        "length": len(motion), "marker": float(motion[0, 0]),
    })
    viewer = GenerationComparisonViewer(_export(tmp_path), host="localhost", port=0)
    clients = [SimpleNamespace(client_id=i, scene=_Scene(), camera=SimpleNamespace(), gui=_Gui()) for i in (1, 2)]
    try:
        for client in clients:
            viewer._on_connect(client)
        session, other = (viewer.sessions[client.client_id] for client in clients)
        session.gui["frame"].value = 3
        old = session.renderers
        session.gui["draw"].value = viewer.draw_labels[2]
        assert all(renderer.cleared for renderer in old)
        assert [renderer.frame for renderer in session.renderers] == [3, 3]
        assert session.renderers[1].marker == 2
        assert other.renderers[1].marker == 0
        assert "Draw 2" in session.gui["info"].content
        session.gui["sample"].value = viewer.labels[1]
        assert session.gui["frame"].max == 1
        assert [renderer.length for renderer in session.renderers] == [2, 2]
        assert session.renderers[1].marker == 2
        assert str(2 ** 64 - 1) in session.gui["info"].content
        clients[0].gui.handles["Play / Pause"].click()
        assert session.renderers[0].updated.wait(1.0)
        assert session.renderers[1].updated.wait(1.0)
        with viewer.lock:
            assert [renderer.frame for renderer in session.renderers] == [session.gui["frame"].value] * 2
        session.gui["draw"].value = viewer.draw_labels[1]
        assert not session.playing
        assert session.renderers[1].marker == 1
        assert other.renderers[1].marker == 0
        session.gui["contacts"].value = True
        session.gui["mesh"].value = True
        session.gui["skeleton"].value = False
        assert all(renderer.contacts and renderer.mesh_visible for renderer in session.renderers)
        assert not any(renderer.shaded_skeleton.visible for renderer in session.renderers)
        generated_paths = [name for name, _ in clients[0].scene.calls if name.endswith("/motion_jepa/mesh")]
        assert "/prediction/target/motion_jepa/mesh" in generated_paths
        assert "/prediction/reconstruction/motion_jepa/mesh" in generated_paths
    finally:
        viewer.close()
    assert viewer.server.stopped
    assert not viewer._playback_thread.is_alive()
