from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from dataset import preprocess_amass
from dataset.convert_amass_to_soma import write_bvh_atomic
from skeleton import SOMASkeleton77
from visualization.dataset_viewer import discover_entries, load_motion


def _write_bvh(path: Path, frames: int) -> None:
    skeleton = SOMASkeleton77()
    rotations = skeleton.relaxed_hands.expand(frames, -1, -1, -1).clone()
    roots = torch.zeros(frames, 3)
    roots[:, 0] = torch.linspace(0.0, 1.0, frames)
    write_bvh_atomic(path, rotations, roots, skeleton)


def _conversion_record(relative: str, frames: int) -> dict:
    return {
        "status": "converted",
        "source_relpath": relative.replace(".bvh", ".npz"),
        "output_relpath": relative,
        "source_fps_raw": 120.0,
        "source_fps": 120,
        "source_num_frames": frames * 4,
        "frame_step": 4,
        "output_num_frames": frames,
        "source_gender": "neutral",
        "surface_model_type": "smplx",
        "target_identity": "smpl_neutral_beta0",
        "mean_vertex_error_m": 0.004,
        "max_vertex_error_m": 0.1,
        "motion_correction": "applied",
    }


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _args(input_dir: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=input_dir,
        output=output,
        manifest=input_dir / "conversion_manifest.jsonl",
        num_frames=90,
        fps=30,
        overlap=0.5,
        workers=1,
        chunksize=8,
        limit=None,
        overwrite=False,
    )


def test_manifest_discovery_preserves_paths_and_limit(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    paths = ("collection/z_stageii.bvh", "collection/a_stageii.bvh")
    for relative in paths:
        destination = input_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
    manifest = input_dir / "conversion_manifest.jsonl"
    _write_manifest(manifest, [_conversion_record(path, 90) for path in paths])

    sources = preprocess_amass.discover_sources(input_dir, manifest)
    limited = preprocess_amass.discover_sources(input_dir, manifest, limit=1)

    assert [source.bvh_relpath for source in sources] == sorted(paths)
    assert sources[0].id == "collection/a_stageii"
    assert [source.id for source in limited] == ["collection/a_stageii"]


def test_manifest_rejects_duplicates_missing_and_unexpected_bvhs(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    manifest = input_dir / "conversion_manifest.jsonl"
    record = _conversion_record("a_stageii.bvh", 90)
    _write_manifest(manifest, [record, record])
    with pytest.raises(ValueError, match="duplicate"):
        preprocess_amass.discover_sources(input_dir, manifest)

    _write_manifest(manifest, [record])
    with pytest.raises(ValueError, match="missing BVHs"):
        preprocess_amass.discover_sources(input_dir, manifest)

    (input_dir / "a_stageii.bvh").touch()
    (input_dir / "extra.bvh").touch()
    with pytest.raises(ValueError, match="absent from manifest"):
        preprocess_amass.discover_sources(input_dir, manifest)


def test_complete_window_policy() -> None:
    assert preprocess_amass.complete_windows(89, 90, 45) == ()
    assert preprocess_amass.complete_windows(90, 90, 45) == ((0, 90),)
    assert preprocess_amass.complete_windows(135, 90, 45) == (
        (0, 90),
        (45, 135),
    )
    assert preprocess_amass.complete_windows(136, 90, 45) == (
        (0, 90),
        (45, 135),
    )


@pytest.mark.parametrize(
    ("rotations", "roots", "fps", "error"),
    [
        (torch.eye(3).repeat(90, 76, 1, 1), torch.zeros(90, 3), 30.0, "SOMA77"),
        (torch.eye(3).repeat(90, 77, 1, 1), torch.zeros(90, 3), 29.0, "must be 30 FPS"),
        (
            torch.full((90, 77, 3, 3), float("nan")),
            torch.zeros(90, 3),
            30.0,
            "NaN or infinity",
        ),
    ],
)
def test_motion_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotations: torch.Tensor,
    roots: torch.Tensor,
    fps: float,
    error: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    preprocess_amass._init_worker(str(tmp_path), str(output), 90, 30, 45)
    monkeypatch.setattr(
        preprocess_amass,
        "parse_bvh_motion",
        lambda _path: (rotations, roots, fps),
    )
    source = preprocess_amass.SourceMotion(
        "motion",
        "motion.bvh",
        _conversion_record("motion.bvh", 90),
    )
    result = preprocess_amass._convert_one(source)
    assert result["ok"] is False
    assert error in result["error"]


def test_preprocess_output_loads_in_existing_viewer(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output = tmp_path / "processed"
    input_dir.mkdir()
    long_relative = "subset/long_stageii.bvh"
    short_relative = "subset/short_stageii.bvh"
    _write_bvh(input_dir / long_relative, 135)
    _write_bvh(input_dir / short_relative, 89)
    _write_manifest(
        input_dir / "conversion_manifest.jsonl",
        [
            _conversion_record(long_relative, 135),
            _conversion_record(short_relative, 89),
        ],
    )

    preprocess_amass.preprocess(_args(input_dir, output))

    entries = discover_entries(output, "train", limit=0)
    assert [entry.id for entry in entries] == [
        "subset/long_stageii_0000",
        "subset/long_stageii_0001",
    ]
    assert not (output / "val.txt").read_text(encoding="utf-8")
    assert not (output / "test.txt").read_text(encoding="utf-8")
    motion, fps = load_motion(entries[0], fps=30)
    assert fps == 30
    assert motion["local_rot_mats"].shape == (90, 77, 3, 3)
    assert torch.isfinite(motion["local_rot_mats"]).all()
    metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert metadata["split_policy"] == "all_train_without_babel"
    assert metadata["tail_policy"] == "discard_incomplete"
    assert metadata["source_rotations_already_tpose_relative"] is True
    assert metadata["num_discovered_source_motions"] == 2
    assert metadata["num_discarded_short_sources"] == 1
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index[0]["captions"] == []
    assert index[0]["source_amass_relpath"] == "subset/long_stageii.npz"
    assert index[0]["start_frame"] == 0
    assert index[1]["start_frame"] == 45
