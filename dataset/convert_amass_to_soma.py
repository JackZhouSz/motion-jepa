"""Convert AMASS stage-II motions to fixed-identity SOMA77 BVH files.

AMASS identity is deliberately discarded for retargeting: both the source
SMPL layer and SOMA fitting layer use the neutral model with zero betas.
Source beta and gender are retained only in the conversion manifest.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_rep.geometry import velocity  # noqa: E402
from skeleton import SOMASkeleton77  # noqa: E402

TARGET_FPS = 30
ROTATION_ORDER = "ZXY"
Z_UP_TO_Y_UP = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=torch.float32
)
FPS_KEYS = ("mocap_frame_rate", "mocap_framerate")
MANIFEST_NAME = "conversion_manifest.jsonl"
ERRORS_NAME = "errors.jsonl"
_NVRTC_BUILTINS_HANDLE: Any | None = None

# These joints are constrained by the body-only SMPL topology. Everything
# else stays at the repository's fixed relaxed/neutral SOMA77 pose.
FITTED_BODY_JOINTS = frozenset(
    {
        "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    }
)


class DiscardSequence(ValueError):
    """A valid input intentionally excluded by the FPS policy."""


def preload_nvrtc_builtins_for_cuda(device: str) -> None:
    """Configure pip CUDA wheels whose NVRTC library has no sibling runpath."""
    global _NVRTC_BUILTINS_HANDLE
    if not str(device).startswith("cuda") or _NVRTC_BUILTINS_HANDLE is not None:
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or spec.submodule_search_locations is None:
        return
    cuda_major = str(torch.version.cuda or "").split(".", maxsplit=1)[0]
    candidates: list[Path] = []
    for root in spec.submodule_search_locations:
        candidates.extend(
            path
            for path in Path(root).glob(f"cu{cuda_major}/lib/libnvrtc-builtins.so.*")
            if ".alt." not in path.name
        )
    if candidates:
        _NVRTC_BUILTINS_HANDLE = ctypes.CDLL(
            str(sorted(candidates)[-1]), mode=ctypes.RTLD_GLOBAL
        )


@dataclass(frozen=True)
class AMASSSequence:
    poses: np.ndarray
    trans: np.ndarray
    source_fps_raw: float
    source_fps: int
    frame_step: int
    source_num_frames: int
    source_betas: np.ndarray
    source_gender: str
    surface_model_type: str | None


@dataclass(frozen=True)
class FitOptions:
    body_iters: int = 2
    finger_iters: int = 0
    full_iters: int = 1
    lie_iters: int = 3
    lie_lambda: float = 1e-1
    autograd_iters: int = 0
    autograd_lr: float = 5e-3

    def as_kwargs(self) -> dict[str, int | float]:
        return {
            "body_iters": self.body_iters,
            "finger_iters": self.finger_iters,
            "full_iters": self.full_iters,
            "lie_iters": self.lie_iters,
            "lie_lambda": self.lie_lambda,
            "autograd_iters": self.autograd_iters,
            "autograd_lr": self.autograd_lr,
        }


def round_fps(fps: float) -> int:
    """Round positive FPS with the preprocessing pipeline's half-up rule."""
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise DiscardSequence(f"invalid source FPS: {fps!r}")
    return int(math.floor(value + 0.5))


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar string metadata, got {array.shape}")
    return str(array.flat[0])


def load_amass_sequence(path: str | Path) -> AMASSSequence:
    """Load, validate, and stride-sample one AMASS stage-II file."""
    path = Path(path)
    try:
        context = np.load(path, allow_pickle=True)
    except Exception as error:
        raise ValueError(f"cannot load NPZ: {error}") from error
    with context as data:
        missing = [key for key in ("poses", "trans") if key not in data]
        if missing:
            raise ValueError(f"missing required arrays: {', '.join(missing)}")
        poses_full = np.asarray(data["poses"])
        trans_full = np.asarray(data["trans"])
        if poses_full.ndim != 2 or poses_full.shape[1] < 66:
            raise ValueError(f"poses must be (T, >=66), got {poses_full.shape}")
        if len(poses_full) == 0:
            raise ValueError("poses must contain at least one frame")
        if trans_full.shape != (len(poses_full), 3):
            raise ValueError(f"trans must be ({len(poses_full)}, 3), got {trans_full.shape}")
        if not np.isfinite(poses_full[:, :66]).all() or not np.isfinite(trans_full).all():
            raise ValueError("poses/trans contain NaN or infinity")

        fps_key = next((key for key in FPS_KEYS if key in data), None)
        if fps_key is None:
            raise ValueError(f"missing FPS metadata; expected one of {FPS_KEYS}")
        source_fps_raw = float(np.asarray(data[fps_key]).reshape(()))
        source_fps = round_fps(source_fps_raw)
        if source_fps % TARGET_FPS:
            raise DiscardSequence(
                f"rounded source FPS {source_fps} is not a multiple of {TARGET_FPS}"
            )
        step = source_fps // TARGET_FPS
        poses = np.zeros((len(poses_full), 72), dtype=np.float32)
        poses[:, :66] = poses_full[:, :66]
        poses = np.ascontiguousarray(poses[::step])
        trans = np.ascontiguousarray(trans_full[::step], dtype=np.float32)
        betas = (
            np.asarray(data["betas"], dtype=np.float32).reshape(-1).copy()
            if "betas" in data else np.empty(0, dtype=np.float32)
        )
        if not np.isfinite(betas).all():
            raise ValueError("betas contain NaN or infinity")
        gender = _scalar_string(data["gender"]) if "gender" in data else "unknown"
        model_type = (
            _scalar_string(data["surface_model_type"])
            if "surface_model_type" in data else None
        )
    return AMASSSequence(
        poses, trans, source_fps_raw, source_fps, step, len(poses_full), betas, gender,
        model_type,
    )


def _tensor_digest(hasher: Any, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    hasher.update(str(array.dtype).encode())
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes())


def skeleton_fingerprint(skeleton: SOMASkeleton77) -> str:
    hasher = hashlib.sha256()
    hasher.update("\0".join(skeleton.names).encode())
    for tensor in (skeleton.parents, skeleton.neutral_joints, skeleton.relaxed_hands):
        _tensor_digest(hasher, tensor)
    return hasher.hexdigest()


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _snapshot_fitting_skeleton(inv: Any) -> dict[str, Any]:
    soma = inv.soma
    view = soma.public_rig_view() if hasattr(soma, "public_rig_view") else None
    parents = view.joint_parent_ids if view is not None else soma.joint_parent_ids
    bind = view.bind_transforms_world if view is not None else soma._cached_bind_transforms_world
    weights = view.skinning_weights if view is not None else soma.skinning_weights
    return {
        "joint_names": tuple(map(str, inv.joint_names)),
        "parents": parents.detach().cpu().clone(),
        "bind": bind.detach().cpu().clone(),
        "rest_shape": soma._cached_rest_shape.detach().cpu().clone(),
        "weights": weights.detach().cpu().clone(),
    }


def assert_skeleton_unchanged(expected: Mapping[str, Any], inv: Any) -> None:
    actual = _snapshot_fitting_skeleton(inv)
    if expected["joint_names"] != actual["joint_names"]:
        raise RuntimeError("SOMA fitting joint names changed")
    for key in ("parents", "bind", "rest_shape", "weights"):
        if not torch.equal(expected[key], actual[key]):
            raise RuntimeError(
                f"SOMA fitting skeleton {key!r} changed; max diff "
                f"{(expected[key] - actual[key]).abs().max().item()}"
            )


def body_only_soma77(
    fitted_relative: torch.Tensor,
    fitted_names: Sequence[str],
    skeleton: SOMASkeleton77,
) -> torch.Tensor:
    name_to_index = {str(name): index for index, name in enumerate(fitted_names)}
    missing = sorted(FITTED_BODY_JOINTS.difference(name_to_index))
    if missing:
        raise ValueError(f"fitted rig is missing joints: {missing}")
    relaxed = skeleton.relaxed_hands.to(fitted_relative)
    output = relaxed.unsqueeze(0).expand(len(fitted_relative), -1, -1, -1).clone()
    for index, name in enumerate(skeleton.names):
        if name in FITTED_BODY_JOINTS:
            output[:, index] = fitted_relative[:, name_to_index[name]]
    return output


def _matrix_to_quaternion_wxyz(matrices: torch.Tensor) -> torch.Tensor:
    shape = matrices.shape[:-2]
    xyzw = Rotation.from_matrix(matrices.cpu().numpy().reshape(-1, 3, 3)).as_quat()
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)
    return torch.from_numpy(wxyz.reshape(*shape, 4).astype(np.float32))


def _quaternion_wxyz_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    shape = quaternions.shape[:-1]
    wxyz = quaternions.cpu().numpy().reshape(-1, 4)
    xyzw = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=-1)
    matrices = Rotation.from_quat(xyzw).as_matrix().astype(np.float32)
    return torch.from_numpy(matrices.reshape(*shape, 3, 3))


def detect_contacts(
    local_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SOMASkeleton77,
    velocity_threshold: float = 0.15,
    relative_height_threshold: float = 0.10,
) -> torch.Tensor:
    """Return [left foot, left toe, right foot, right toe] contacts."""
    _, positions, _ = skeleton.fk(local_rotations, root_positions)
    positions = positions.unsqueeze(0)
    speeds = torch.linalg.norm(velocity(positions, TARGET_FPS), dim=-1)
    indices = skeleton.left_foot_joint_idx + skeleton.right_foot_joint_idx
    effectors = positions[:, :, indices]
    floors = effectors[..., 1].amin(dim=1, keepdim=True)
    return (
        (speeds[:, :, indices] < velocity_threshold)
        & (effectors[..., 1] < floors + relative_height_threshold)
    ).to(torch.float32)


def _working_rig(skeleton: SOMASkeleton77, ground_offset: float) -> list[Any]:
    neutral = skeleton.neutral_joints.cpu().numpy()
    parents = skeleton.parents.cpu().numpy()
    tags = {
        "Hips": "Hips", "Head": "Head", "LeftHand": "LeftHand",
        "RightHand": "RightHand", "LeftFoot": "LeftFoot", "RightFoot": "RightFoot",
    }
    lowest = float(neutral[:, 1].min())
    rig = []
    for index, name in enumerate(skeleton.names):
        parent = int(parents[index])
        if parent < 0:
            translation = neutral[index] + [0.0, -lowest + ground_offset, 0.0]
            parent_name = None
        else:
            translation = neutral[index] - neutral[parent]
            parent_name = skeleton.names[parent]
        rig.append(
            SimpleNamespace(
                name=name,
                parent=parent_name,
                t_pose_translation=np.asarray(translation, dtype=float).tolist(),
                t_pose_rotation=[0.0, 0.0, 0.0, 1.0],
                retarget_tag=tags.get(name),
            )
        )
    return rig


def apply_motion_correction(
    local_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SOMASkeleton77,
    velocity_threshold: float = 0.15,
    relative_height_threshold: float = 0.10,
    contact_threshold: float = 0.5,
    root_margin: float = 0.04,
    ground_offset: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    contacts = detect_contacts(
        local_rotations,
        root_positions,
        skeleton,
        velocity_threshold=velocity_threshold,
        relative_height_threshold=relative_height_threshold,
    )
    if not bool((contacts >= contact_threshold).any()):
        return local_rotations, root_positions, contacts, "no_contacts"
    try:
        from motion_correction.motion_postprocess import correct_motion
    except ImportError as error:
        raise RuntimeError(
            "install the vendored extension with "
            "`pip install -e third_party/motion_correction`, or use --no-motion-correction"
        ) from error
    roots = root_positions.cpu().float().unsqueeze(0).clone()
    quaternions = _matrix_to_quaternion_wxyz(local_rotations).unsqueeze(0).clone()
    masks = {
        name: torch.zeros(len(local_rotations), dtype=torch.float32)
        for name in ("Root", "FullBody", "LeftHand", "RightHand", "LeftFoot", "RightFoot")
    }
    correct_motion(
        roots, quaternions, contacts, roots.clone(), quaternions.clone(), masks,
        contact_threshold, root_margin, _working_rig(skeleton, ground_offset), False,
    )
    corrected = _quaternion_wxyz_to_matrix(quaternions[0])
    if not torch.isfinite(corrected).all() or not torch.isfinite(roots).all():
        raise RuntimeError("MotionCorrection produced non-finite values")
    return corrected, roots[0], contacts, "applied"


def hierarchy_lines(skeleton: SOMASkeleton77) -> list[str]:
    children: list[list[int]] = [[] for _ in skeleton.names]
    for child, parent in enumerate(skeleton.parents.tolist()):
        if parent >= 0:
            children[int(parent)].append(child)
    neutral = skeleton.neutral_joints.cpu().numpy()

    def emit(index: int, depth: int) -> list[str]:
        indent = "  " * depth
        parent = int(skeleton.parents[index])
        offset = neutral[index] if parent < 0 else neutral[index] - neutral[parent]
        offset = offset * 100.0
        lines = [f"{indent}{'ROOT' if depth == 0 else 'JOINT'} {skeleton.names[index]}", f"{indent}{{"]
        lines.append(f"{indent}  OFFSET {offset[0]:.9f} {offset[1]:.9f} {offset[2]:.9f}")
        if depth == 0:
            lines.append(
                f"{indent}  CHANNELS 6 Xposition Yposition Zposition "
                "Zrotation Xrotation Yrotation"
            )
        else:
            lines.append(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation")
        for child in children[index]:
            lines.extend(emit(child, depth + 1))
        if not children[index]:
            lines.extend(
                [f"{indent}  End Site", f"{indent}  {{",
                 f"{indent}    OFFSET 0.000000000 0.000000000 0.000000000",
                 f"{indent}  }}"]
            )
        lines.append(f"{indent}}}")
        return lines
    return ["HIERARCHY", *emit(0, 0)]


def bvh_text(
    local_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SOMASkeleton77,
) -> str:
    expected = (len(root_positions), len(skeleton.names), 3, 3)
    if tuple(local_rotations.shape) != expected:
        raise ValueError(f"expected rotations {expected}, got {tuple(local_rotations.shape)}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        euler = Rotation.from_matrix(
            local_rotations.cpu().numpy().reshape(-1, 3, 3)
        ).as_euler(ROTATION_ORDER, degrees=True)
    euler = euler.reshape(len(root_positions), len(skeleton.names), 3)
    roots_cm = root_positions.cpu().numpy() * 100.0
    if not np.isfinite(euler).all() or not np.isfinite(roots_cm).all():
        raise ValueError("BVH contains non-finite values")
    lines = hierarchy_lines(skeleton)
    lines.extend(["MOTION", f"Frames: {len(roots_cm)}", f"Frame Time: {1 / TARGET_FPS:.12f}"])
    for frame in range(len(roots_cm)):
        values = [*roots_cm[frame], *euler[frame].reshape(-1)]
        lines.append(" ".join(f"{float(value):.9f}" for value in values))
    return "\n".join(lines) + "\n"


def write_bvh_atomic(
    path: str | Path,
    local_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: SOMASkeleton77,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(bvh_text(local_rotations, root_positions, skeleton))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class FixedIdentityConverter:
    """Long-lived zero-beta source and target models."""

    def __init__(
        self,
        smpl_model_path: str | Path,
        data_root: str | Path | None,
        device: str,
        fit_options: FitOptions,
    ) -> None:
        requested_device = torch.device(device if torch.cuda.is_available() else "cpu")
        preload_nvrtc_builtins_for_cuda(str(requested_device))
        from soma import SMPLLayer, SOMALayer, get_assets_dir
        from soma.fitting.pose_inversion import PoseInversion

        self.device = requested_device
        self.fit_options = fit_options
        self.data_root = Path(data_root) if data_root else Path(get_assets_dir())
        self.model_path = Path(smpl_model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        kwargs = dict(gender="neutral", num_betas=10, device=self.device, mode="warp")
        self.source = SMPLLayer(self.data_root, model_path=self.model_path, **kwargs)
        if self.source._v_template.shape[0] != 6890 or self.source.num_joints != 24:
            raise ValueError(
                "SMPL model must contain 6,890 vertices and 24 joints; got "
                f"{self.source._v_template.shape[0]} vertices and "
                f"{self.source.num_joints} joints"
            )
        if self.source.num_identity_coeffs < 10:
            raise ValueError("SMPL model must provide at least ten beta components")
        self.target = SOMALayer(
            self.data_root,
            identity_model_type="smpl",
            identity_model_kwargs={
                "model_path": str(self.model_path), "gender": "neutral", "num_betas": 10,
            },
            device=self.device,
            mode="warp",
            enable_procedural_transforms=False,
        )
        zeros = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        self.source.prepare_identity(zeros)
        # py-soma-x 0.3.0's low-LOD PoseInversion path uses the stale relative
        # import ``soma.fitting.body`` although SOMALayer lives in
        # ``soma.body``. Keep the workaround local and harmless once upstream
        # ships that module or fixes the import.
        try:
            importlib.import_module("soma.fitting.body")
        except ModuleNotFoundError as error:
            if error.name != "soma.fitting.body":
                raise
            sys.modules["soma.fitting.body"] = importlib.import_module("soma.body")
        self.inv = PoseInversion(self.target, low_lod=True)
        self.inv.prepare_identity(zeros)
        self.snapshot = _snapshot_fitting_skeleton(self.inv)

    def convert(
        self, sequence: AMASSSequence, batch_size: int
    ) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
        from soma.geometry.rig_utils import remove_joint_orient_local

        poses = torch.from_numpy(sequence.poses).to(self.device)
        trans = torch.from_numpy(sequence.trans).to(self.device)
        basis = Z_UP_TO_Y_UP.to(self.device)
        rotations, roots, errors = [], [], []
        for start in range(0, len(poses), batch_size):
            end = min(start + batch_size, len(poses))
            with torch.inference_mode():
                vertices = self.source.pose(
                    poses[start:end].reshape(-1, 24, 3),
                    pose2rot=True,
                    apply_correctives=True,
                    absolute_pose=False,
                    global_translation=trans[start:end],
                )["vertices"] @ basis.T
            result = self.inv.fit(vertices, **self.fit_options.as_kwargs())
            rotations.append(result["rotations"].detach().cpu())
            roots.append(result["root_translation"].detach().cpu())
            errors.append(result["per_vertex_error"].detach().cpu())
        rotations_tensor = torch.cat(rotations)
        soma = self.inv.soma
        relative = remove_joint_orient_local(
            rotations_tensor.to(soma._t_pose_orient.device),
            soma._t_pose_orient,
            soma._t_pose_orient_parent_T,
        ).cpu()
        names = list(map(str, self.inv.joint_names))
        if names[0] == "Root":
            names, relative = names[1:], relative[:, 1:]
        roots_tensor, errors_tensor = torch.cat(roots), torch.cat(errors)
        if not all(torch.isfinite(value).all() for value in (relative, roots_tensor, errors_tensor)):
            raise RuntimeError("PoseInversion produced non-finite values")
        assert_skeleton_unchanged(self.snapshot, self.inv)
        return relative, names, roots_tensor, errors_tensor

    def assert_unchanged(self) -> None:
        assert_skeleton_unchanged(self.snapshot, self.inv)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()


def discover_stageii(input_dir: Path, limit: int | None = None) -> list[Path]:
    sources = sorted(input_dir.rglob("*_stageii.npz"))
    return sources if limit is None else sources[:limit]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smpl-model-path", type=Path, default=Path("skeleton/assets/smpl/SMPL_NEUTRAL.pkl"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--motion-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-velocity", type=float, default=0.15)
    parser.add_argument("--contact-height", type=float, default=0.10)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--root-margin", type=float, default=0.04)
    parser.add_argument("--ground-offset", type=float, default=0.02)
    parser.add_argument("--body-iters", type=int, default=2)
    parser.add_argument("--finger-iters", type=int, default=0)
    parser.add_argument("--full-iters", type=int, default=1)
    parser.add_argument("--lie-iters", type=int, default=3)
    parser.add_argument("--lie-lambda", type=float, default=1e-1)
    parser.add_argument("--autograd-iters", type=int, default=0)
    parser.add_argument("--autograd-lr", type=float, default=5e-3)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("input and output directories must differ")
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("batch size and limit must be positive")
    iterations = (
        args.body_iters, args.finger_iters, args.full_iters,
        args.lie_iters, args.autograd_iters,
    )
    if any(value < 0 for value in iterations) or not any(iterations):
        raise ValueError("at least one nonnegative fitting iteration must be enabled")
    if args.contact_velocity < 0 or args.contact_height < 0:
        raise ValueError("contact velocity and height thresholds must be nonnegative")
    if not 0 <= args.contact_threshold <= 1:
        raise ValueError("contact threshold must be in [0, 1]")
    if args.root_margin < 0 or args.ground_offset < 0:
        raise ValueError("root margin and ground offset must be nonnegative")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    sources = discover_stageii(args.input_dir, args.limit)
    if not sources:
        raise FileNotFoundError(f"no *_stageii.npz under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    skeleton = SOMASkeleton77()
    fingerprint = skeleton_fingerprint(skeleton)
    options = FitOptions(
        args.body_iters, args.finger_iters, args.full_iters, args.lie_iters,
        args.lie_lambda, args.autograd_iters, args.autograd_lr,
    )
    converter = FixedIdentityConverter(
        args.smpl_model_path, args.data_root, args.device, options
    )
    model_digest = file_sha256(args.smpl_model_path)
    manifest, error_log = (
        args.output_dir / MANIFEST_NAME, args.output_dir / ERRORS_NAME
    )
    counts = {"converted": 0, "skipped": 0, "discarded": 0, "failed": 0}

    for source in tqdm(sources, unit="file", dynamic_ncols=True):
        relative_path = source.relative_to(args.input_dir)
        output = args.output_dir / relative_path.with_suffix(".bvh")
        common = {
            "source_relpath": relative_path.as_posix(),
            "output_relpath": output.relative_to(args.output_dir).as_posix(),
        }
        if args.skip_existing and output.is_file():
            counts["skipped"] += 1
            continue
        try:
            converter.assert_unchanged()
            sequence = load_amass_sequence(source)
            relative, names, roots, errors = converter.convert(sequence, args.batch_size)
            rotations = body_only_soma77(relative, names, skeleton)
            correction, contact_frames = "disabled", 0
            if args.motion_correction:
                rotations, roots, contacts, correction = apply_motion_correction(
                    rotations,
                    roots,
                    skeleton,
                    velocity_threshold=args.contact_velocity,
                    relative_height_threshold=args.contact_height,
                    contact_threshold=args.contact_threshold,
                    root_margin=args.root_margin,
                    ground_offset=args.ground_offset,
                )
                contact_frames = int(
                    (contacts >= args.contact_threshold).any(dim=-1).sum()
                )
            write_bvh_atomic(output, rotations, roots, skeleton)
            converter.assert_unchanged()
            append_jsonl(
                manifest,
                {
                    **common,
                    "status": "converted",
                    "source_fps_raw": sequence.source_fps_raw,
                    "source_fps": sequence.source_fps,
                    "target_fps": TARGET_FPS,
                    "source_num_frames": sequence.source_num_frames,
                    "frame_step": sequence.frame_step,
                    "output_num_frames": len(sequence.poses),
                    "source_betas": sequence.source_betas.tolist(),
                    "source_gender": sequence.source_gender,
                    "surface_model_type": sequence.surface_model_type,
                    "target_identity": "smpl_neutral_beta0",
                    "target_betas": [0.0] * 10,
                    "smpl_model_sha256": model_digest,
                    "skeleton_sha256": fingerprint,
                    "mean_vertex_error_m": float(errors.mean()),
                    "max_vertex_error_m": float(errors.max()),
                    "motion_correction": correction,
                    "motion_correction_settings": {
                        "contact_velocity_mps": args.contact_velocity,
                        "contact_relative_height_m": args.contact_height,
                        "contact_threshold": args.contact_threshold,
                        "root_margin_m": args.root_margin,
                        "ground_offset_m": args.ground_offset,
                    },
                    "contact_frames": contact_frames,
                    "fit_options": options.as_kwargs(),
                },
            )
            counts["converted"] += 1
        except DiscardSequence as error:
            append_jsonl(error_log, {**common, "status": "discarded", "error": str(error)})
            counts["discarded"] += 1
        except Exception as error:
            append_jsonl(
                error_log,
                {**common, "status": "failed", "error": f"{type(error).__name__}: {error}",
                 "traceback": traceback.format_exc(limit=12)},
            )
            counts["failed"] += 1
    print(json.dumps(counts, sort_keys=True))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
