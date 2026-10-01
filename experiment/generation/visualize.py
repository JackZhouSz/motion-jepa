"""CPU-only comparison of a reference motion and a selected generated draw."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiment.prediction.visualize import (
    ComparisonViewer,
    _decode_motion,
    _NamespacedScene,
)


@dataclass(frozen=True)
class GenerationResults:
    sample_ids: np.ndarray
    lengths: np.ndarray
    fps: int
    target_motion: np.ndarray
    generated_motion: np.ndarray
    target_joints: np.ndarray
    generated_joints: np.ndarray
    noise_seeds: np.ndarray
    sampling_json: str | None = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def num_draws(self) -> int:
        return int(self.generated_motion.shape[1])


def load_results(path: Path) -> GenerationResults:
    """Validate an evaluator/sample NPZ without loading a model or GPU."""
    required = (
        "sample_ids", "lengths", "fps", "target_motion", "generated_motion",
        "target_joints", "generated_joints", "noise_seeds",
    )
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Generation export is missing fields: {sorted(missing)}")
        arrays = {key: archive[key] for key in required}
        sampling = archive["sampling_json"] if "sampling_json" in archive.files else None
    ids = arrays["sample_ids"]
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "US":
        raise ValueError("sample_ids must be a nonempty vector of strings.")
    lengths = arrays["lengths"]
    if lengths.shape != ids.shape or lengths.dtype.kind not in "iu":
        raise ValueError("lengths must be an integer vector matching sample_ids.")
    fps = arrays["fps"]
    if fps.ndim != 0 or fps.dtype.kind not in "iuf" or not np.isfinite(fps) or fps <= 0 or float(fps) != int(fps):
        raise ValueError("fps must be a positive integer scalar.")
    target = arrays["target_motion"]
    generated = arrays["generated_motion"]
    if target.ndim != 3 or target.shape[0] != len(ids) or target.shape[2] != 366:
        raise ValueError("target_motion must have shape [N, T, 366].")
    if generated.ndim != 4 or generated.shape[0] != len(ids) or generated.shape[1] < 1 or generated.shape[2:] != target.shape[1:]:
        raise ValueError("generated_motion must have shape [N, K, T, 366] with K >= 1.")
    if np.any(lengths <= 0) or np.any(lengths > target.shape[1]):
        raise ValueError("lengths must be positive and no larger than the exported frame count.")
    expected = {
        "target_motion": target.shape,
        "generated_motion": generated.shape,
        "target_joints": (*target.shape[:2], 30, 3),
        "generated_joints": (*generated.shape[:3], 30, 3),
    }
    for key, shape in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite floating-point data with shape {shape}.")
    seeds = arrays["noise_seeds"]
    if seeds.shape != generated.shape[:2] or seeds.dtype.kind not in "iu" or np.any(seeds < 0):
        raise ValueError("noise_seeds must be a nonnegative integer array with shape [N, K].")
    sampling_json = None
    if sampling is not None:
        if sampling.ndim != 0 or sampling.dtype.kind not in "US":
            raise ValueError("sampling_json must be a scalar JSON string.")
        sampling_json = sampling.item()
        if isinstance(sampling_json, bytes):
            sampling_json = sampling_json.decode("utf-8")
        try:
            metadata = json.loads(sampling_json)
        except (TypeError, ValueError) as error:
            raise ValueError("sampling_json must contain a JSON object.") from error
        if not isinstance(metadata, dict):
            raise ValueError("sampling_json must contain a JSON object.")
    return GenerationResults(
        sample_ids=ids.astype(str), lengths=lengths.astype(np.int64), fps=int(fps),
        target_motion=target, generated_motion=generated,
        target_joints=arrays["target_joints"], generated_joints=arrays["generated_joints"],
        noise_seeds=seeds, sampling_json=sampling_json,
    )


class GenerationComparisonViewer(ComparisonViewer):
    """Reuse reconstruction playback and rendering, adding per-browser draw selection."""

    def __init__(self, results_path: Path, *, host="0.0.0.0", port=8080, mesh=False):
        self.results = load_results(results_path)
        try:
            import viser
            from visualization.dataset_viewer import MotionRenderer
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Generation visualization requires the motion-jepa environment with viser and trimesh."
            ) from error
        self.renderer_type = MotionRenderer
        self.default_mesh = bool(mesh)
        self.labels = tuple(f"{index}: {sample_id}" for index, sample_id in enumerate(self.results.sample_ids))
        self.draw_labels = tuple(f"Draw {index}" for index in range(self.results.num_draws))
        self.sessions = {}
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self.server = viser.ViserServer(
            host=host, port=port, label="MotionJEPA Conditional Generation", enable_camera_keyboard_controls=False,
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self._on_connect)
        self.server.on_client_disconnect(self._on_disconnect)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def _load_sample(self, session, index: int, *, frame: int = 0) -> None:
        with self.lock:
            session.playing = False
            draw_index = self.draw_labels.index(session.gui["draw"].value) if "draw" in session.gui else 0
            for renderer in session.renderers:
                renderer.clear()
            session.renderers = ()
            length = int(self.results.lengths[index])
            renderers = []
            try:
                for prefix, features in (
                    ("/prediction/target", self.results.target_motion[index]),
                    ("/prediction/reconstruction", self.results.generated_motion[index, draw_index]),
                ):
                    motion = _decode_motion(features[:length], self.results.fps)
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
                f"{length} frames · {self.results.fps} FPS · Draw {draw_index}\n\n"
                f"Noise seed: `{int(self.results.noise_seeds[index, draw_index])}`\n\n"
                "Left: original · Right: generated"
            )
            self._set_frame(session, frame)

    def _on_connect(self, client) -> None:
        super()._on_connect(client)
        client.scene.add_label("/prediction/reconstruction/label", text="Generated", position=(0.0, 2.2, 0.0))
        session = self.sessions[client.client_id]
        with client.gui.add_folder("Generated draws", expand_by_default=True):
            draw = client.gui.add_dropdown("Draw", self.draw_labels, initial_value=self.draw_labels[0])
        session.gui["draw"] = draw

        @draw.on_update
        def _(_event):
            # Keep the selected frame to compare alternative motions at that instant.
            self._load_sample(session, session.index, frame=int(session.gui["frame"].value))


def visualize_results(results_path: Path, *, host="0.0.0.0", port=8080, mesh=False) -> None:
    viewer = GenerationComparisonViewer(results_path, host=host, port=port, mesh=mesh)
    try:
        while not viewer._stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


__all__ = ["GenerationResults", "GenerationComparisonViewer", "load_results", "visualize_results"]
