"""Renderer replacement must serialize with playback and GUI callbacks."""

from __future__ import annotations

import threading
import time
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dataset.visualize import ProcessedDatasetViewer
from experiment.tmr.visualize_dataset import TMRDatasetViewer, TMRMotionEntry
from visualization.dataset_viewer import MotionJEPADatasetViewer


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
            callback(SimpleNamespace(client_id=None, target=self))

    def on_update(self, callback):
        self.updates.append(callback)
        return callback

    def on_click(self, callback):
        self.clicks.append(callback)
        return callback

    def click(self):
        for callback in self.clicks:
            callback(SimpleNamespace(client_id=1, target=self))


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

    def add_button(self, label):
        return self._add(label)

    def add_button_group(self, label, options):
        return self._add(label, options[0])

    def add_number(self, label, initial_value, **_kwargs):
        return self._add(label, initial_value)

    def add_slider(self, label, *args, **kwargs):
        handle = self._add(label, kwargs.get("initial_value", args[-1] if args else 0))
        handle.max = kwargs.get("max", args[1] if args else 1)
        return handle

    def add_checkbox(self, label, initial_value):
        return self._add(label, initial_value)


class _Server:
    def __init__(self, **_kwargs):
        self.scene = SimpleNamespace(world_axes=SimpleNamespace(), set_up_direction=lambda _: None)

    def on_client_connect(self, _callback):
        pass

    def on_client_disconnect(self, _callback):
        pass

    def stop(self):
        pass


class _Renderer:
    def __init__(self, _client, motion, *_args, **_kwargs):
        self.length = motion["length"]
        self.frame = 0
        self.removed = False
        self.joints = SimpleNamespace(visible=True)
        self.bones = SimpleNamespace(visible=True)
        self.updated = threading.Event()
        self.clear_started = threading.Event()
        self.clear_release = None

    def clear(self):
        self.removed = True
        self.clear_started.set()
        if self.clear_release is not None and not self.clear_release.wait(2.0):
            raise RuntimeError("Timed out waiting to finish renderer replacement")

    def set_frame(self, frame, show_contacts):
        if self.removed:
            raise RuntimeError("Cannot assign to 'batched_positions' on a removed BatchedMeshHandle.")
        self.frame = max(0, min(frame, self.length - 1))
        if self.frame > 0:
            self.updated.set()

    def set_mesh_visible(self, _visible):
        if self.removed:
            raise RuntimeError("Removed renderer")

    def set_mesh_opacity(self, _opacity):
        if self.removed:
            raise RuntimeError("Removed renderer")


@contextmanager
def _viewer(kind):
    root = Path("/fixture")
    entries = [TMRMotionEntry(
        id=f"sample-{index}", path=root / f"sample-{index}.npy", actual_length=8 - index,
        caption=f"caption-{index}", captions=(f"caption-{index}",),
        source_id=f"source-{index}", start_frame=0, end_frame=8 - index, fps=30,
        caption_id=f"text-{index}",
    ) for index in range(2)]
    module = "dataset.visualize" if kind is ProcessedDatasetViewer else "visualization.dataset_viewer"
    errors = []
    with (
        patch.dict("sys.modules", {"viser": SimpleNamespace(ViserServer=_Server)}),
        patch(f"{module}.read_dataset_fps", return_value=30),
        patch(f"{module}.discover_entries", return_value=entries),
        patch(f"{module}.load_motion", side_effect=lambda entry, **_: ({"length": entry.actual_length}, 30)),
        patch(f"{module}.MotionRenderer", _Renderer),
        patch("experiment.tmr.visualize_dataset.load_cached_entries", return_value=(root, entries)),
        patch("threading.excepthook", side_effect=lambda args: errors.append(args.exc_value)),
    ):
        kwargs = dict(split="train", limit=2, host="localhost", port=0, mesh=False)
        if kind is ProcessedDatasetViewer:
            kwargs["normalized"] = False
        viewer = kind(root, **kwargs)
        client = SimpleNamespace(
            client_id=1, camera=SimpleNamespace(), gui=_Gui(),
            scene=SimpleNamespace(add_grid=lambda *_args, **_kwargs: None),
        )
        viewer._on_connect(client)
        try:
            yield viewer, client, viewer.sessions[1], errors
        finally:
            viewer.close()


class DatasetViewerPlaybackTest(unittest.TestCase):
    kinds = (MotionJEPADatasetViewer, TMRDatasetViewer, ProcessedDatasetViewer)

    def test_play_pause_works_after_repeated_sample_changes(self):
        for kind in self.kinds:
            with self.subTest(viewer=kind.__name__), _viewer(kind) as (viewer, client, session, errors):
                play = client.gui.handles["Play / Pause"]
                dropdown = session.gui["dropdown"]
                for index in (1, 0, 1):
                    play.click()
                    self.assertTrue(session.renderer.updated.wait(1.0))
                    previous = session.renderer
                    dropdown.value = (
                        viewer.labels[index] if kind is ProcessedDatasetViewer else viewer.entries[index].label
                    )
                    self.assertTrue(previous.removed)
                    self.assertFalse(session.playing)
                    self.assertEqual(session.renderer.frame, 0)
                    play.click()
                    self.assertTrue(session.renderer.updated.wait(1.0))
                    play.click()
                    self.assertFalse(session.playing)
                    paused_frame = session.renderer.frame
                    time.sleep(0.05)
                    self.assertEqual(session.renderer.frame, paused_frame)
                    session.gui["frame"].value = 3
                    self.assertEqual(session.renderer.frame, 3)
                    self.assertTrue(viewer._playback_thread.is_alive())
                self.assertEqual(errors, [])

    def test_removed_renderer_is_unavailable_to_playback_and_slider_callbacks(self):
        for kind in self.kinds:
            with self.subTest(viewer=kind.__name__), _viewer(kind) as (viewer, client, session, errors):
                previous = session.renderer
                previous.clear_release = threading.Event()
                play = client.gui.handles["Play / Pause"]
                play.click()
                self.assertTrue(previous.updated.wait(1.0))
                label = viewer.labels[1] if kind is ProcessedDatasetViewer else viewer.entries[1].label
                switch = threading.Thread(target=lambda: setattr(session.gui["dropdown"], "value", label))
                callback_started = threading.Event()
                callback_finished = threading.Event()

                def move_slider():
                    callback_started.set()
                    session.gui["frame"].value = 5
                    callback_finished.set()

                slider = threading.Thread(target=move_slider)
                switch.start()
                try:
                    self.assertTrue(previous.clear_started.wait(1.0))
                    slider.start()
                    self.assertTrue(callback_started.wait(1.0))
                    self.assertFalse(callback_finished.wait(0.1))
                    self.assertEqual(errors, [])
                finally:
                    previous.clear_release.set()
                    switch.join(timeout=1.0)
                    if slider.ident is not None:
                        slider.join(timeout=1.0)
                self.assertFalse(switch.is_alive())
                self.assertFalse(slider.is_alive())
                self.assertTrue(callback_finished.is_set())
                self.assertEqual(errors, [])
                self.assertTrue(viewer._playback_thread.is_alive())
                session.renderer.updated.clear()
                play.click()
                self.assertTrue(session.renderer.updated.wait(1.0))


if __name__ == "__main__":
    unittest.main()
