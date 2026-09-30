"""Visualize processed MotionJEPA clips selected by their NPY path."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization import (  # noqa: E402
    MotionEntry,
    MotionRenderer,
    discover_entries,
    load_motion,
    read_dataset_fps,
)


@dataclass
class ViewerSession:
    client: Any
    dropdown: Any
    renderer: MotionRenderer | None = None
    playing: bool = False
    speed: float = 1.0
    fps: int = 60
    next_frame_time: float = 0.0
    gui: dict[str, Any] | None = None


def _display_path(root: Path, entry: MotionEntry) -> str:
    """Return a unique user-facing processed data path."""
    try:
        database = entry.path.relative_to(root).as_posix()
    except ValueError:
        database = entry.path.as_posix()
    return database


class ProcessedDatasetViewer:
    """Full motion viewer whose dropdown selects an NPY sample path."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        limit: int,
        host: str,
        port: int,
        mesh: bool,
        normalized: bool,
    ) -> None:
        try:
            import viser
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Dataset visualization requires `pip install viser trimesh`."
            ) from error

        self.root = root
        self.fps = read_dataset_fps(root)
        self.mesh = bool(mesh)
        self.normalized = bool(normalized)
        self.stats_root = root / "stats" if root.is_dir() else None
        self.entries = discover_entries(root, split, limit)
        self.labels = [_display_path(root, entry) for entry in self.entries]
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Processed motion paths must be unique.")
        self.entry_by_label = dict(zip(self.labels, self.entries))
        self.sessions: dict[int, ViewerSession] = {}
        # Programmatic slider changes synchronously invoke their callbacks.
        self.lock = threading.RLock()
        self.closed = False

        self.server = viser.ViserServer(
            host=host,
            port=port,
            label="MotionJEPA Processed Dataset",
            enable_camera_keyboard_controls=False,
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self._on_connect)
        self.server.on_client_disconnect(self._on_disconnect)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def _load(self, session: ViewerSession, label: str) -> None:
        entry = self.entry_by_label[label]
        motion, fps = load_motion(
            entry,
            fps=self.fps,
            normalized=self.normalized,
            stats_root=self.stats_root,
        )
        with self.lock:
            session.playing = False
            previous_renderer = session.renderer
            session.renderer = None
            if previous_renderer is not None:
                previous_renderer.clear()
            session.renderer = MotionRenderer(
                session.client,
                motion,
                show_mesh=(
                    bool(session.gui["mesh"].value) if session.gui else self.mesh
                ),
                show_skeleton=(
                    bool(session.gui["skeleton"].value)
                    if session.gui
                    else not self.mesh
                ),
            )
            session.fps = fps
            session.next_frame_time = time.monotonic()
            if session.gui:
                session.renderer.set_mesh_opacity(float(session.gui["opacity"].value))
                session.gui["frame"].max = max(0, session.renderer.length - 1)
                session.gui["frame"].value = 0
                session.gui["fps"].value = fps
                session.gui["info"].content = (
                    f"### {entry.id}\n`{label}`\n\n{entry.caption}"
                )

    def _on_connect(self, client: Any) -> None:
        client.camera.position = np.array([2.7, 1.9, 7.7])
        client.camera.look_at = np.array([0.0, 1.0, 0.0])
        client.scene.add_grid(
            "/ground",
            width=20,
            height=20,
            plane="xz",
            position=(0.0, 0.0, 0.0),
            infinite_grid=True,
        )
        with client.gui.add_folder("Dataset", expand_by_default=True):
            dropdown = client.gui.add_dropdown(
                "Processed data path",
                self.labels,
                initial_value=self.labels[0],
            )
            info = client.gui.add_markdown("")
        with client.gui.add_folder("Playback", expand_by_default=True):
            fps_handle = client.gui.add_number("FPS", initial_value=60, disabled=True)
            frame = client.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            play = client.gui.add_button("Play / Pause")
            previous = client.gui.add_button("Previous")
            next_ = client.gui.add_button("Next")
            speed = client.gui.add_button_group("Speed", ("0.5x", "1x", "2x"))
            speed.value = "1x"
        with client.gui.add_folder("Display", expand_by_default=True):
            mesh = client.gui.add_checkbox("Show Mesh", initial_value=self.mesh)
            opacity = client.gui.add_slider("Mesh Opacity", 0.0, 1.0, 0.01, 0.9)
            skeleton = client.gui.add_checkbox(
                "Show Skeleton", initial_value=not self.mesh
            )
            contacts = client.gui.add_checkbox(
                "Show Foot Contacts", initial_value=False
            )
        session = ViewerSession(client=client, dropdown=dropdown)
        session.gui = {
            "dropdown": dropdown,
            "info": info,
            "fps": fps_handle,
            "frame": frame,
            "mesh": mesh,
            "opacity": opacity,
            "skeleton": skeleton,
            "contacts": contacts,
        }
        with self.lock:
            self.sessions[client.client_id] = session
        self._load(session, dropdown.value)

        def set_frame(value: int) -> None:
            with self.lock:
                if session.renderer is None:
                    return
                session.renderer.set_frame(value, show_contacts=bool(contacts.value))
                if frame.value != session.renderer.frame:
                    frame.value = session.renderer.frame

        @dropdown.on_update
        def _(_event: Any) -> None:
            self._load(session, dropdown.value)

        @frame.on_update
        def _(_event: Any) -> None:
            set_frame(int(frame.value))

        @play.on_click
        def _(_event: Any) -> None:
            with self.lock:
                session.playing = not session.playing
                session.next_frame_time = time.monotonic()

        @previous.on_click
        def _(_event: Any) -> None:
            with self.lock:
                session.playing = False
                set_frame(int(frame.value) - 1)

        @next_.on_click
        def _(_event: Any) -> None:
            with self.lock:
                session.playing = False
                set_frame(int(frame.value) + 1)

        @speed.on_click
        def _(_event: Any) -> None:
            with self.lock:
                session.speed = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[speed.value]

        @mesh.on_update
        def _(_event: Any) -> None:
            with self.lock:
                if session.renderer is not None:
                    session.renderer.set_mesh_visible(bool(mesh.value))

        @opacity.on_update
        def _(_event: Any) -> None:
            with self.lock:
                if session.renderer is not None:
                    session.renderer.set_mesh_opacity(float(opacity.value))

        @skeleton.on_update
        def _(_event: Any) -> None:
            with self.lock:
                if session.renderer is not None:
                    session.renderer.joints.visible = bool(skeleton.value)
                    session.renderer.bones.visible = bool(skeleton.value)

        @contacts.on_update
        def _(_event: Any) -> None:
            set_frame(int(frame.value))

    def _on_disconnect(self, client: Any) -> None:
        with self.lock:
            session = self.sessions.pop(client.client_id, None)
            if session is not None:
                session.playing = False
                session.renderer = None

    def close(self) -> None:
        with self.lock:
            self.closed = True
        if threading.current_thread() is not self._playback_thread:
            self._playback_thread.join(timeout=1.0)

    def _playback_loop(self) -> None:
        while not self.closed:
            now = time.monotonic()
            with self.lock:
                sessions = list(self.sessions.values())
            for session in sessions:
                with self.lock:
                    renderer = session.renderer
                    if self.closed:
                        return
                    if (
                        renderer is None
                        or not session.playing
                        or session.gui is None
                        or now < session.next_frame_time
                    ):
                        continue
                    next_frame = (
                        0 if renderer.frame >= renderer.length - 1 else renderer.frame + 1
                    )
                    renderer.set_frame(
                        next_frame,
                        show_contacts=bool(session.gui["contacts"].value),
                    )
                    session.gui["frame"].value = next_frame
                    session.next_frame_time = now + 1.0 / max(
                        float(session.fps) * session.speed,
                        1.0,
                    )
            time.sleep(1.0 / 240.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize processed MotionJEPA NPY clips using a path dropdown."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "dataset/bones-seed-processed",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum dropdown entries; use 0 to list all NPY samples.",
    )
    parser.add_argument("--host", default=os.environ.get("SERVER_NAME", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SERVER_PORT", "6006")),
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Render the skinned mesh instead of the skeleton.",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Undo normalization with the dataset statistics before decoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = ProcessedDatasetViewer(
        args.input,
        split=args.split,
        limit=args.limit,
        host=args.host,
        port=args.port,
        mesh=args.mesh,
        normalized=args.normalized,
    )
    print(f"Found {len(viewer.entries)} NPY motion sample(s) under {args.input}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    main()
