from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from dataset.convert_amass_to_soma import (
    DiscardSequence,
    FITTED_BODY_JOINTS,
    apply_motion_correction,
    body_only_soma77,
    bvh_text,
    detect_contacts,
    discover_stageii,
    load_amass_sequence,
    round_fps,
    skeleton_fingerprint,
    write_bvh_atomic,
)
from skeleton import SOMASkeleton77
from skeleton.bvh import parse_bvh_motion


def _write_amass(path: Path, *, frames: int = 9, fps: float = 120.0, **extra) -> None:
    values = {
        "poses": np.arange(frames * 165, dtype=np.float32).reshape(frames, 165) / 1000,
        "trans": np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) / 100,
        "mocap_frame_rate": np.asarray(fps),
        "betas": np.linspace(-1, 1, 16, dtype=np.float32),
        "gender": np.asarray("female"),
        "surface_model_type": np.asarray("smplx"),
    }
    values.update(extra)
    np.savez(path, **values)


def test_amass_loading_uses_body_pose_and_exact_stride(tmp_path: Path) -> None:
    source = tmp_path / "motion_stageii.npz"
    _write_amass(source)

    sequence = load_amass_sequence(source)

    assert sequence.source_fps == 120
    assert sequence.source_fps_raw == 120.0
    assert sequence.frame_step == 4
    assert sequence.source_num_frames == 9
    assert sequence.poses.shape == (3, 72)
    np.testing.assert_array_equal(sequence.poses[:, :66], np.load(source)["poses"][::4, :66])
    np.testing.assert_array_equal(sequence.poses[:, 66:], 0)
    np.testing.assert_array_equal(sequence.trans, np.load(source)["trans"][::4])
    assert sequence.source_gender == "female"
    assert sequence.surface_model_type == "smplx"


def test_amass_loading_supports_alternate_fps_key(tmp_path: Path) -> None:
    source = tmp_path / "motion_stageii.npz"
    _write_amass(
        source,
        mocap_frame_rate=np.asarray(120.0),
        mocap_framerate=np.asarray(60.0),
    )
    with np.load(source) as original:
        values = {key: original[key] for key in original.files if key != "mocap_frame_rate"}
    np.savez(source, **values)
    assert load_amass_sequence(source).frame_step == 2


@pytest.mark.parametrize("fps, expected", [(29.5, 30), (59.5, 60), (119.6, 120)])
def test_fps_is_rounded_half_up(fps: float, expected: int) -> None:
    assert round_fps(fps) == expected


def test_non_multiple_fps_is_discarded(tmp_path: Path) -> None:
    source = tmp_path / "motion_stageii.npz"
    _write_amass(source, fps=100.0)
    with pytest.raises(DiscardSequence, match="not a multiple"):
        load_amass_sequence(source)


def test_discovery_excludes_stagei_and_preserves_limit(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    for relative in ("z_stageii.npz", "nested/a_stageii.npz", "a_stagei.npz", "other.npz"):
        (tmp_path / relative).touch()
    assert [path.name for path in discover_stageii(tmp_path)] == [
        "a_stageii.npz",
        "z_stageii.npz",
    ]
    assert [path.name for path in discover_stageii(tmp_path, 1)] == ["a_stageii.npz"]


def test_corrupt_and_nonfinite_inputs_are_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing_stageii.npz"
    np.savez(missing, poses=np.zeros((2, 165), dtype=np.float32), mocap_frame_rate=60)
    with pytest.raises(ValueError, match="missing required"):
        load_amass_sequence(missing)

    nonfinite = tmp_path / "nonfinite_stageii.npz"
    trans = np.zeros((2, 3), dtype=np.float32)
    trans[0, 0] = np.nan
    np.savez(
        nonfinite,
        poses=np.zeros((2, 165), dtype=np.float32),
        trans=trans,
        mocap_frame_rate=60,
    )
    with pytest.raises(ValueError, match="NaN or infinity"):
        load_amass_sequence(nonfinite)


def test_body_only_projection_keeps_fixed_relaxed_joints() -> None:
    skeleton = SOMASkeleton77()
    fitted_names = list(skeleton.names)
    fitted = torch.eye(3).repeat(2, len(fitted_names), 1, 1)
    angle = Rotation.from_euler("X", 15, degrees=True).as_matrix().astype(np.float32)
    fitted[:, fitted_names.index("LeftArm")] = torch.from_numpy(angle)
    fitted[:, fitted_names.index("Jaw")] = torch.from_numpy(angle)

    projected = body_only_soma77(fitted, fitted_names, skeleton)

    assert torch.equal(projected[:, skeleton.names.index("LeftArm")], fitted[:, fitted_names.index("LeftArm")])
    assert torch.equal(
        projected[:, skeleton.names.index("Jaw")],
        skeleton.relaxed_hands[skeleton.names.index("Jaw")].expand(2, 3, 3),
    )
    for index, name in enumerate(skeleton.names):
        if name not in FITTED_BODY_JOINTS:
            assert torch.equal(projected[:, index], skeleton.relaxed_hands[index].expand(2, 3, 3))


def test_bvh_round_trip_has_fixed_soma77_hierarchy(tmp_path: Path) -> None:
    skeleton = SOMASkeleton77()
    frames = 4
    rotations = skeleton.relaxed_hands.expand(frames, -1, -1, -1).clone()
    turn = Rotation.from_euler("ZXY", [13.0, -7.0, 19.0], degrees=True).as_matrix()
    rotations[1, skeleton.names.index("LeftArm")] = torch.from_numpy(turn).float()
    roots = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.2, -0.3], [0.2, 0.2, -0.1], [0.4, 0.1, 0.0]],
        dtype=torch.float32,
    )
    output = tmp_path / "nested" / "motion.bvh"

    write_bvh_atomic(output, rotations, roots, skeleton)
    parsed_rotations, parsed_roots, fps = parse_bvh_motion(output)

    assert fps == pytest.approx(30.0, abs=1e-8)
    assert parsed_rotations.shape == (frames, 77, 3, 3)
    torch.testing.assert_close(parsed_rotations.float(), rotations, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(parsed_roots, roots, atol=1e-7, rtol=1e-7)
    assert bvh_text(rotations, roots, skeleton).count("CHANNELS") == 77
    assert skeleton_fingerprint(skeleton) == skeleton_fingerprint(SOMASkeleton77())


def test_contacts_and_motion_correction_are_finite() -> None:
    skeleton = SOMASkeleton77()
    frames = 5
    rotations = skeleton.relaxed_hands.expand(frames, -1, -1, -1).clone()
    roots = torch.zeros(frames, 3)
    contacts = detect_contacts(rotations, roots, skeleton)
    assert contacts.shape == (1, frames, 4)
    assert torch.equal(contacts, torch.ones_like(contacts))

    corrected_rotations, corrected_roots, returned_contacts, status = apply_motion_correction(
        rotations, roots, skeleton
    )
    assert status == "applied"
    assert torch.isfinite(corrected_rotations).all()
    assert torch.isfinite(corrected_roots).all()
    assert torch.equal(returned_contacts, contacts)
    _, before, _ = skeleton.fk(rotations, roots)
    _, after, _ = skeleton.fk(corrected_rotations, corrected_roots)
    effectors = skeleton.left_foot_joint_idx + skeleton.right_foot_joint_idx
    assert after[:, effectors, 1].amin() >= before[:, effectors, 1].amin()
    before_drift = torch.linalg.norm(torch.diff(before[:, effectors], dim=0), dim=-1)
    after_drift = torch.linalg.norm(torch.diff(after[:, effectors], dim=0), dim=-1)
    assert after_drift.max() <= before_drift.max()
