"""Structured masking for MotionJEPA."""

from .collators import MaskCollator1D, MaskCollator2D
from .collators_v2 import MaskCollator1DV2, PatchMaskCollator1DV2
from .body_region_collator import (
    PatchBodyRegionSegmentMaskCollator2D,
    PatchRandomBodySegmentMaskCollator2D,
    PatchRandomSpatialSegmentMaskCollator2D,
)
from .patch_collators import PatchMaskCollator1D, PatchMaskCollator2D

__all__ = [
    "MaskCollator1D",
    "MaskCollator1DV2",
    "MaskCollator2D",
    "PatchBodyRegionSegmentMaskCollator2D",
    "PatchRandomBodySegmentMaskCollator2D",
    "PatchRandomSpatialSegmentMaskCollator2D",
    "PatchMaskCollator1D",
    "PatchMaskCollator1DV2",
    "PatchMaskCollator2D",
]
