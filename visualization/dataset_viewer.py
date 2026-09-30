"""Independent viser dataset browser for NPY-backed MotionJEPA motions."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from motion_rep import MotionJEPAMotionRep
from skeleton import SOMASkeleton30, SOMASkeleton77

from .shaded_skeleton import ShadedSkeletonRenderer
from .soma_skin import SOMASkin


@dataclass(frozen=True)
class MotionEntry:
    id: str
    path: Path
    actual_length: int
    caption: str = ""

    @property
    def label(self) -> str:
        return self.path.as_posix()


def discover_entries(root: Path, split: str, limit: int) -> list[MotionEntry]:
    if not root.is_dir():
        raise FileNotFoundError(f"MotionJEPA NPY dataset root does not exist: {root}")
    records: dict[str, dict[str, Any]] = {}
    index_records: list[dict[str, Any]] = []
    index_path = root / "index.json"
    if index_path.is_file():
        index_records = json.loads(index_path.read_text(encoding="utf-8"))
        records = {record["id"]: record for record in index_records}
    split_path = root / f"{split}.txt"
    entries: list[MotionEntry] = []
    if split_path.is_file():
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise ValueError(f"Malformed split row in {split_path}: {line!r}")
            item_id, relative_path, fps_text, length_text = fields
            try:
                sample_fps = int(fps_text)
                actual_length = int(length_text)
            except ValueError as error:
                raise ValueError(f"Malformed NPY locator in {split_path}: {line!r}") from error
            if sample_fps <= 0 or actual_length <= 0:
                raise ValueError(f"Malformed NPY locator in {split_path}: {line!r}")
            path = root / relative_path
            record = records.get(item_id, {})
            caption = (record.get("captions") or [""])[0]
            entries.append(MotionEntry(item_id, path, actual_length, caption))
    else:
        if index_records:
            for record in index_records:
                if record.get("split") != split or not record.get("motion_path"):
                    continue
                caption = (record.get("captions") or [""])[0]
                entries.append(
                    MotionEntry(
                        str(record["id"]),
                        root / str(record["motion_path"]),
                        int(record["length"]),
                        caption,
                    )
                )
    if limit > 0:
        entries = entries[:limit]
    if not entries:
        raise FileNotFoundError(f"No MotionJEPA samples found under {root}")
    return entries


def read_dataset_fps(root: Path) -> int:
    path = root / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"No MotionJEPA NPY metadata found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("motion_storage") != "npy_float32_v1":
        raise ValueError(f"Dataset is not NPY-backed: {path}")
    fps = int(metadata["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid dataset FPS in {path}: {fps}")
    return fps


def load_motion(
    entry: MotionEntry,
    *,
    fps: int,
    normalized: bool = False,
    stats_root: Path | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    features = np.load(entry.path, allow_pickle=False)
    expected_shape = (entry.actual_length, MotionJEPAMotionRep.FEATURE_DIM)
    if features.shape != expected_shape or features.dtype != np.float32:
        raise ValueError(
            f"NPY motion must be float32{expected_shape}, got "
            f"{features.dtype}{features.shape}: {entry.label}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"Motion contains non-finite values: {entry.label}")
    if normalized:
        if stats_root is None:
            raise ValueError("Normalized features require a stats directory.")
        mean_path = stats_root / "mean.npy"
        std_path = stats_root / "std.npy"
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                f"Expected both normalization files: {mean_path} and {std_path}"
            )
        mean = np.load(mean_path, allow_pickle=False).astype(np.float32, copy=False)
        std = np.load(std_path, allow_pickle=False).astype(np.float32, copy=False)
        expected_shape = (MotionJEPAMotionRep.FEATURE_DIM,)
        if mean.shape != expected_shape or std.shape != expected_shape:
            raise ValueError(
                f"Normalization statistics must have shape {expected_shape}, got "
                f"mean={mean.shape}, std={std.shape}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"Normalization statistics contain non-finite values: {stats_root}")
        std = np.where(std < 1.0e-6, 1.0, std)
        features = features * std + mean
    representation = MotionJEPAMotionRep(SOMASkeleton30(), fps)
    decoded = representation.inverse(torch.from_numpy(features))
    return representation.skeleton.expand_output(decoded), fps


class MotionRenderer:
    def __init__(self, client, motion: dict[str, torch.Tensor], show_mesh: bool, show_skeleton: bool):
        self.client = client
        self.motion = motion
        self.skeleton = SOMASkeleton77()
        self.skin = SOMASkin(self.skeleton)
        self.parents = self.skeleton.parents.numpy()
        self.foot_indices = np.asarray(
            [
                self.skeleton.name_to_index["LeftFoot"],
                self.skeleton.name_to_index["LeftToeBase"],
                self.skeleton.name_to_index["LeftToeEnd"],
                self.skeleton.name_to_index["RightFoot"],
                self.skeleton.name_to_index["RightToeBase"],
                self.skeleton.name_to_index["RightToeEnd"],
            ]
        )
        self.frame = 0
        self._mesh_visible = bool(show_mesh)
        self._mesh_vertices: np.ndarray | None = None
        self._mesh_opacity = 0.9
        self.mesh = None
        if self._mesh_visible:
            self._create_mesh(self._ensure_mesh_cache()[0])
        first_joints = motion["posed_joints"][0].detach().cpu().numpy()
        self.shaded_skeleton = ShadedSkeletonRenderer(
            client.scene,
            first_joints,
            skeleton=self.skeleton,
            visible=show_skeleton,
        )
        # Retain these public handle attributes for callers of MotionRenderer.
        self.joints = self.shaded_skeleton.joints
        self.bones = self.shaded_skeleton.bones

    @property
    def length(self) -> int:
        return int(self.motion["posed_joints"].shape[0])

    def _contact_indices(self, frame: int, show_contacts: bool) -> np.ndarray | None:
        if not show_contacts:
            return None
        contacts = self.motion["foot_contacts"][frame].detach().cpu().numpy().astype(bool)
        return self.foot_indices[contacts]

    def _ensure_mesh_cache(self) -> np.ndarray:
        if self._mesh_vertices is None:
            rotations = self.motion["global_rot_mats"]
            positions = self.motion["posed_joints"]
            with torch.inference_mode():
                first = self.skin.pose(rotations[0], positions[0])
                self._mesh_vertices = np.empty(
                    (self.length, *first.shape),
                    dtype=np.float32,
                )
                self._mesh_vertices[0] = first
                for frame in range(1, self.length):
                    self._mesh_vertices[frame] = self.skin.pose(
                        rotations[frame], positions[frame]
                    )
        return self._mesh_vertices

    def _create_mesh(self, vertices: np.ndarray) -> None:
        self.mesh = self.client.scene.add_mesh_simple(
            "/motion_jepa/mesh",
            vertices=vertices,
            faces=self.skin.faces,
            color=(152, 189, 255),
            opacity=self._mesh_opacity,
            side="double",
            visible=True,
        )

    def set_mesh_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible and self.mesh is None:
            self._create_mesh(self._ensure_mesh_cache()[self.frame])
        elif visible and not self._mesh_visible:
            self.mesh.vertices = self._ensure_mesh_cache()[self.frame]
        self._mesh_visible = visible
        if self.mesh is not None:
            self.mesh.visible = visible

    def set_mesh_opacity(self, opacity: float) -> None:
        self._mesh_opacity = float(opacity)
        if self.mesh is not None:
            self.mesh.opacity = self._mesh_opacity

    def _skeleton_geometry(self, frame: int, show_contacts: bool):
        """Compatibility helper returning joint positions and colors."""
        points = self.motion["posed_joints"][frame].detach().cpu().numpy()
        colors = np.full((len(points), 3), (255, 235, 0), dtype=np.uint8)
        contact_indices = self._contact_indices(frame, show_contacts)
        if contact_indices is not None:
            colors[contact_indices] = (255, 0, 0)
        segments = np.stack(
            [[points[int(self.parents[joint])], points[joint]] for joint in range(1, len(points))],
            axis=0,
        )
        return points, segments, colors

    def set_frame(self, frame: int, show_contacts: bool) -> None:
        self.frame = int(np.clip(frame, 0, self.length - 1))
        positions = self.motion["posed_joints"][self.frame]
        if self._mesh_visible:
            assert self.mesh is not None
            self.mesh.vertices = self._ensure_mesh_cache()[self.frame]
        points = positions.detach().cpu().numpy()
        self.shaded_skeleton.update(
            points,
            contact_indices=self._contact_indices(self.frame, show_contacts),
        )

    def clear(self) -> None:
        if self.mesh is not None:
            self.mesh.remove()
        self.shaded_skeleton.remove()


@dataclass
class ViewerSession:
    client: Any
    entry_index: int = 0
    renderer: MotionRenderer | None = None
    playing: bool = False
    speed: float = 1.0
    fps: int = 30
    next_frame_time: float = 0.0
    gui: dict[str, Any] | None = None


class MotionJEPADatasetViewer:
    def __init__(
        self,
        root: Path,
        *,
        split: str = "train",
        limit: int = 500,
        sample_index: int = 0,
        sample_id: str | None = None,
        host: str = "0.0.0.0",
        port: int = 6006,
        mesh: bool = False,
        normalized: bool = False,
        entries: list[MotionEntry] | None = None,
        label: str = "MotionJEPA Dataset Viewer",
    ):
        try:
            import viser
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("MotionJEPA visualization requires `pip install viser trimesh`.") from error
        self.viser = viser
        self.root = root
        self.fps = read_dataset_fps(root)
        self.normalized = normalized
        self.stats_root = root / "stats" if root.is_dir() else None
        self.entries = discover_entries(root, split, limit) if entries is None else list(entries)
        if entries is not None and limit > 0:
            self.entries = self.entries[:limit]
        if not self.entries:
            raise FileNotFoundError(f"No MotionJEPA samples found under {root}")
        self.initial_index = max(0, min(sample_index, len(self.entries) - 1))
        if sample_id is not None:
            matches = [index for index, entry in enumerate(self.entries) if entry.id == sample_id]
            if not matches:
                raise KeyError(f"Unknown sample id: {sample_id}")
            self.initial_index = matches[0]
        self.default_mesh = mesh
        self.sessions: dict[int, ViewerSession] = {}
        self.lock = threading.Lock()
        self.closed = False
        self.server = viser.ViserServer(
            host=host,
            port=port,
            label=label,
            enable_camera_keyboard_controls=False,
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self._on_connect)
        self.server.on_client_disconnect(self._on_disconnect)
        threading.Thread(target=self._playback_loop, daemon=True).start()

    def _load_entry(self, session: ViewerSession, index: int) -> None:
        entry = self.entries[index]
        motion, fps = load_motion(
            entry,
            fps=self.fps,
            normalized=self.normalized,
            stats_root=self.stats_root,
        )
        if session.renderer is not None:
            session.renderer.clear()
        gui = session.gui or {}
        session.renderer = MotionRenderer(
            session.client,
            motion,
            bool(gui.get("mesh").value) if "mesh" in gui else self.default_mesh,
            bool(gui.get("skeleton").value) if "skeleton" in gui else not self.default_mesh,
        )
        session.entry_index = index
        session.fps = fps
        session.playing = False
        session.next_frame_time = time.monotonic()
        if gui:
            gui["frame"].max = max(0, session.renderer.length - 1)
            gui["frame"].value = 0
            gui["fps"].value = fps
            gui["info"].content = f"### {entry.id}\n`{entry.label}`\n\n{entry.caption}"

    def _on_connect(self, client) -> None:
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
        session = ViewerSession(client=client, entry_index=self.initial_index)
        self.sessions[client.client_id] = session
        labels = [entry.label for entry in self.entries]
        with client.gui.add_folder("Dataset", expand_by_default=True):
            dropdown = client.gui.add_dropdown("Motion", labels, initial_value=labels[self.initial_index])
            info = client.gui.add_markdown("")
        with client.gui.add_folder("Playback", expand_by_default=True):
            fps_handle = client.gui.add_number("FPS", initial_value=30, disabled=True)
            frame = client.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            play = client.gui.add_button("Play / Pause")
            previous = client.gui.add_button("Previous")
            next_ = client.gui.add_button("Next")
            speed = client.gui.add_button_group("Speed", ("0.5x", "1x", "2x"))
            speed.value = "1x"
        with client.gui.add_folder("Display", expand_by_default=True):
            mesh = client.gui.add_checkbox("Show Mesh", initial_value=self.default_mesh)
            opacity = client.gui.add_slider("Mesh Opacity", 0.0, 1.0, 0.01, 0.9)
            skeleton = client.gui.add_checkbox("Show Skeleton", initial_value=not self.default_mesh)
            contacts = client.gui.add_checkbox("Show Foot Contacts", initial_value=False)
        session.gui = {
            "dropdown": dropdown,
            "info": info,
            "fps": fps_handle,
            "frame": frame,
            "mesh": mesh,
            "skeleton": skeleton,
            "contacts": contacts,
        }
        self._load_entry(session, self.initial_index)

        def set_frame(value: int) -> None:
            if session.renderer is None:
                return
            session.renderer.set_frame(value, bool(contacts.value))
            if frame.value != session.renderer.frame:
                frame.value = session.renderer.frame

        @dropdown.on_update
        def _(_event):
            self._load_entry(session, labels.index(dropdown.value))

        @frame.on_update
        def _(_event):
            set_frame(int(frame.value))

        @play.on_click
        def _(_event):
            session.playing = not session.playing

        @previous.on_click
        def _(_event):
            set_frame(int(frame.value) - 1)

        @next_.on_click
        def _(_event):
            set_frame(int(frame.value) + 1)

        @speed.on_click
        def _(_event):
            session.speed = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[speed.value]

        @mesh.on_update
        def _(_event):
            if session.renderer:
                session.renderer.set_mesh_visible(bool(mesh.value))

        @opacity.on_update
        def _(_event):
            if session.renderer:
                session.renderer.set_mesh_opacity(float(opacity.value))

        @skeleton.on_update
        def _(_event):
            if session.renderer:
                session.renderer.joints.visible = bool(skeleton.value)
                session.renderer.bones.visible = bool(skeleton.value)

        @contacts.on_update
        def _(_event):
            set_frame(int(frame.value))

    def _on_disconnect(self, client) -> None:
        self.sessions.pop(client.client_id, None)

    def close(self) -> None:
        self.closed = True

    def _playback_loop(self) -> None:
        while not self.closed:
            now = time.monotonic()
            for session in list(self.sessions.values()):
                renderer = session.renderer
                if renderer is None or not session.playing or session.gui is None:
                    continue
                period = 1.0 / max(float(session.fps) * session.speed, 1.0)
                if now >= session.next_frame_time:
                    next_frame = 0 if renderer.frame >= renderer.length - 1 else renderer.frame + 1
                    renderer.set_frame(next_frame, bool(session.gui["contacts"].value))
                    session.gui["frame"].value = next_frame
                    session.next_frame_time = now + period
            time.sleep(1.0 / 240.0)
