"""Paths to assets packaged with MotionJEPA."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = PACKAGE_ROOT / "assets"
SKELETONS_ROOT = ASSETS_ROOT / "skeletons"
SKELETON_METADATA_PATH = ASSETS_ROOT / "motion_jepa_skeleton_metadata.npz"


def skeleton_asset_path(*parts: str) -> Path:
    return SKELETONS_ROOT.joinpath(*parts)


__all__ = [
    "ASSETS_ROOT",
    "PACKAGE_ROOT",
    "SKELETONS_ROOT",
    "SKELETON_METADATA_PATH",
    "skeleton_asset_path",
]
