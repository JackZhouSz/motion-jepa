"""Factory helpers for MotionJEPA skeleton variants."""

from .asset_paths import skeleton_asset_path
from .definitions import SOMASkeleton30, SOMASkeleton77


def build_skeleton(nbjoints: int):
    if nbjoints == 30:
        return SOMASkeleton30(skeleton_asset_path("somaskel30"))
    if nbjoints == 77:
        return SOMASkeleton77(skeleton_asset_path("somaskel77"))
    raise ValueError(f"Unsupported MotionJEPA skeleton joint count: {nbjoints}")


__all__ = ["build_skeleton"]
