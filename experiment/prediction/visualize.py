"""CPU-only, synchronized comparison viewer for exported reconstructions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReconstructionResults:
    sample_ids: np.ndarray
    lengths: np.ndarray
    fps: int
    target_motion: np.ndarray
    reconstructed_motion: np.ndarray
    target_joints: np.ndarray
    reconstructed_joints: np.ndarray

    def __len__(self) -> int:
        return len(self.sample_ids)


def load_results(path: Path) -> ReconstructionResults:
    """Read the evaluator's NPZ without requiring torch, a JEPA model, or GPU."""
    required = (
        "sample_ids", "lengths", "fps", "target_motion", "reconstructed_motion",
        "target_joints", "reconstructed_joints",
    )
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Reconstruction export is missing fields: {sorted(missing)}")
        arrays = {key: archive[key] for key in required}
    ids = arrays["sample_ids"]
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "US":
        raise ValueError("sample_ids must be a nonempty vector of strings.")
    lengths = arrays["lengths"]
    if lengths.shape != ids.shape or lengths.dtype.kind not in "iu":
        raise ValueError("lengths must be an integer vector matching sample_ids.")
    fps = arrays["fps"]
    if fps.ndim != 0 or fps.dtype.kind not in "iuf" or not np.isfinite(fps) or fps <= 0 or float(fps) != int(fps):
        raise ValueError("fps must be a positive integer scalar.")
    motion = arrays["target_motion"]
    if motion.ndim != 3 or motion.shape[0] != len(ids) or motion.shape[2] != 366:
        raise ValueError("target_motion must have shape [N, T, 366].")
    if np.any(lengths <= 0) or np.any(lengths > motion.shape[1]):
        raise ValueError("lengths must be positive and no larger than the exported frame count.")
    for key in ("target_motion", "reconstructed_motion", "target_joints", "reconstructed_joints"):
        value = arrays[key]
        expected = motion.shape if key.endswith("motion") else (*motion.shape[:2], 30, 3)
        if value.shape != expected or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite floating-point data with shape {expected}.")
    return ReconstructionResults(
        sample_ids=ids.astype(str), lengths=lengths.astype(np.int64), fps=int(fps),
        target_motion=arrays["target_motion"], reconstructed_motion=arrays["reconstructed_motion"],
        target_joints=arrays["target_joints"], reconstructed_joints=arrays["reconstructed_joints"],
    )


class _NamespacedScene:
    """Prefix the existing renderer's fixed names under a translated scene frame."""

    def __init__(self, scene: Any, prefix: str):
        self.scene = scene
        self.prefix = prefix.rstrip("/")

    def add_mesh_simple(self, name: str, **kwargs):
        return self.scene.add_mesh_simple(self.prefix + "/" + name.lstrip("/"), **kwargs)

    def add_batched_meshes_simple(self, name: str, **kwargs):
        return self.scene.add_batched_meshes_simple(self.prefix + "/" + name.lstrip("/"), **kwargs)


def _decode_motion(motion: np.ndarray, fps: int) -> dict:
    # Deferred imports keep export inspection independent of model/viewer dependencies.
    import torch

    from motion_rep import MotionJEPAMotionRep
    from skeleton import SOMASkeleton30

    skeleton = SOMASkeleton30()
    representation = MotionJEPAMotionRep(skeleton, fps)
    with torch.inference_mode():
        return skeleton.expand_output(
            representation.inverse(torch.from_numpy(np.asarray(motion, dtype=np.float32)))
        )


@dataclass
class _Session:
    client: Any
    index: int = 0
    renderers: tuple[Any, ...] = ()
    playing: bool = False
    speed: float = 1.0
    next_frame_time: float = 0.0
    updating_frame: bool = False
    gui: dict[str, Any] = field(default_factory=dict)


class ComparisonViewer:
    """Each connected browser controls one synchronized target/reconstruction pair."""

    def __init__(self, results_path: Path, *, host="0.0.0.0", port=8080, mesh=False):
        self.results = load_results(results_path)
        try:
            import viser
            from visualization.dataset_viewer import MotionRenderer
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Reconstruction visualization requires the motion-jepa environment with viser and trimesh."
            ) from error
        self.renderer_type = MotionRenderer
        self.default_mesh = bool(mesh)
        self.labels = tuple(f"{index}: {sample_id}" for index, sample_id in enumerate(self.results.sample_ids))
        self.sessions: dict[int, _Session] = {}
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self.server = viser.ViserServer(
            host=host, port=port, label="MotionJEPA Reconstruction", enable_camera_keyboard_controls=False,
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self._on_connect)
        self.server.on_client_disconnect(self._on_disconnect)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def _load_sample(self, session: _Session, index: int) -> None:
        # Serialize decoding, renderer replacement, playback, and GUI updates.
        with self.lock:
            session.playing = False
            for renderer in session.renderers:
                renderer.clear()
            session.renderers = ()
            length = int(self.results.lengths[index])
            renderers = []
            try:
                for prefix, features in (
                    ("/prediction/target", self.results.target_motion),
                    ("/prediction/reconstruction", self.results.reconstructed_motion),
                ):
                    motion = _decode_motion(features[index, :length], self.results.fps)
                    renderer_client = SimpleNamespace(scene=_NamespacedScene(session.client.scene, prefix))
                    renderers.append(self.renderer_type(
                        renderer_client, motion, bool(session.gui["mesh"].value), bool(session.gui["skeleton"].value),
                    ))
            except Exception:
                for renderer in renderers:
                    renderer.clear()
                raise
            session.renderers = tuple(renderers)
            session.index = index
            session.next_frame_time = time.monotonic()
            session.gui["frame"].max = max(0, length - 1)
            session.gui["info"].content = (
                f"**{self.results.sample_ids[index]}**\n\n"
                f"{length} frames · {self.results.fps} FPS\n\n"
                "Left: original · Right: reconstructed"
            )
            self._set_frame(session, 0)

    def _set_frame(self, session: _Session, frame: int) -> None:
        with self.lock:
            if not session.renderers or session.updating_frame:
                return
            session.updating_frame = True
            try:
                frame = int(np.clip(frame, 0, session.renderers[0].length - 1))
                for renderer in session.renderers:
                    renderer.set_frame(frame, bool(session.gui["contacts"].value))
                if session.gui["frame"].value != frame:
                    session.gui["frame"].value = frame
            finally:
                session.updating_frame = False

    def _on_connect(self, client) -> None:
        client.camera.position = np.array([0.0, 2.0, 8.0])
        client.camera.look_at = np.array([0.0, 1.0, 0.0])
        client.scene.add_grid("/ground", width=20, height=20, plane="xz", infinite_grid=True)
        for prefix, offset, label in (
            ("/prediction/target", -1.5, "Original"),
            ("/prediction/reconstruction", 1.5, "Reconstructed"),
        ):
            client.scene.add_frame(prefix, position=(offset, 0.0, 0.0), show_axes=False)
            client.scene.add_label(prefix + "/label", text=label, position=(0.0, 2.2, 0.0))
        session = _Session(client)
        with self.lock:
            self.sessions[client.client_id] = session
        with client.gui.add_folder("Reconstruction", expand_by_default=True):
            sample = client.gui.add_dropdown("Sample", self.labels, initial_value=self.labels[0])
            info = client.gui.add_markdown("")
        with client.gui.add_folder("Playback", expand_by_default=True):
            frame = client.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            play = client.gui.add_button("Play / Pause")
            previous = client.gui.add_button("Previous frame")
            next_ = client.gui.add_button("Next frame")
            speed = client.gui.add_button_group("Speed", ("0.5x", "1x", "2x"))
            speed.value = "1x"
        with client.gui.add_folder("Display", expand_by_default=True):
            mesh = client.gui.add_checkbox("Show Mesh", initial_value=self.default_mesh)
            skeleton = client.gui.add_checkbox("Show Skeleton", initial_value=not self.default_mesh)
            contacts = client.gui.add_checkbox("Show Foot Contacts", initial_value=False)
        session.gui = dict(sample=sample, info=info, frame=frame, mesh=mesh, skeleton=skeleton, contacts=contacts)
        self._load_sample(session, 0)

        @sample.on_update
        def _(_event):
            self._load_sample(session, self.labels.index(sample.value))

        @frame.on_update
        def _(_event):
            self._set_frame(session, int(frame.value))

        @play.on_click
        def _(_event):
            with self.lock:
                session.playing = not session.playing
                session.next_frame_time = time.monotonic()

        @previous.on_click
        def _(_event):
            self._set_frame(session, int(frame.value) - 1)

        @next_.on_click
        def _(_event):
            self._set_frame(session, int(frame.value) + 1)

        @speed.on_click
        def _(_event):
            with self.lock:
                session.speed = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[speed.value]

        @mesh.on_update
        def _(_event):
            with self.lock:
                for renderer in session.renderers:
                    renderer.set_mesh_visible(bool(mesh.value))

        @skeleton.on_update
        def _(_event):
            with self.lock:
                for renderer in session.renderers:
                    renderer.shaded_skeleton.visible = bool(skeleton.value)

        @contacts.on_update
        def _(_event):
            self._set_frame(session, int(frame.value))

    def _on_disconnect(self, client) -> None:
        with self.lock:
            session = self.sessions.pop(client.client_id, None)
            if session:
                session.playing = False
                for renderer in session.renderers:
                    renderer.clear()
                session.renderers = ()

    def _playback_loop(self) -> None:
        while not self._stop.wait(1.0 / 240.0):
            now = time.monotonic()
            with self.lock:
                for session in list(self.sessions.values()):
                    if not session.playing or not session.renderers or now < session.next_frame_time:
                        continue
                    renderer = session.renderers[0]
                    self._set_frame(session, (renderer.frame + 1) % renderer.length)
                    session.next_frame_time = now + 1.0 / (self.results.fps * session.speed)

    def close(self) -> None:
        self._stop.set()
        if threading.current_thread() is not self._playback_thread:
            self._playback_thread.join(timeout=1.0)
        with self.lock:
            for session in list(self.sessions.values()):
                self._on_disconnect(session.client)
        self.server.stop()


def visualize_results(results_path: Path, *, host="0.0.0.0", port=8080, mesh=False) -> None:
    """Serve a comparison until interrupted; no training or encoder loading occurs."""
    viewer = ComparisonViewer(results_path, host=host, port=port, mesh=mesh)
    try:
        while not viewer._stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
