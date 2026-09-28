"""Public Motion-JEPA model variants and named factories."""

from .motion_transformer_1d import (
    MotionTransformer1D,
    MotionTransformerPredictor1D,
    
    mot_tiny_1d,
    mot_small_1d,
    mot_base_1d,
    mot_large_1d,
    mot_huge_1d,
    mot_giant_1d,

    mot_predictor_tiny_1d,
    mot_predictor_small_1d,
    mot_predictor_base_1d,
    mot_predictor_large_1d,
    mot_predictor_huge_1d,
    mot_predictor_giant_1d,
)
from .motion_transformer_2d import (
    MotionFeatureTokenizer2D,
    MotionTransformer2D,
    MotionTransformerPredictor2D,
    mot_tiny_2d,
    mot_small_2d,
    mot_base_2d,
    mot_large_2d,
    mot_huge_2d,
    mot_giant_2d,
    mot_predictor_tiny_2d,
    mot_predictor_small_2d,
    mot_predictor_base_2d,
    mot_predictor_large_2d,
    mot_predictor_huge_2d,
    mot_predictor_giant_2d,
)
from .motion_patch_transformer_1d import (
    MotionPatchTransformer1D,
    MotionPatchTransformerPredictor1D,
    mot_patch_tiny_1d,
    mot_patch_small_1d,
    mot_patch_base_1d,
    mot_patch_large_1d,
    mot_patch_huge_1d,
    mot_patch_giant_1d,
    mot_predictor_patch_tiny_1d,
    mot_predictor_patch_small_1d,
    mot_predictor_patch_base_1d,
    mot_predictor_patch_large_1d,
    mot_predictor_patch_huge_1d,
    mot_predictor_patch_giant_1d,
)
from .motion_patch_transformer_2d import (
    BODY_TOKEN_OFFSET,
    MotionPatchTransformer2D,
    MotionPatchTransformerPredictor2D,
    SPATIAL_GROUPINGS,
    get_spatial_grouping,
    spatial_patch_signature,
    TRAJECTORY_FIELDS,
    TRAJECTORY_TOKEN_INDEX,
    mot_patch_tiny_2d,
    mot_patch_small_2d,
    mot_patch_base_2d,
    mot_patch_large_2d,
    mot_patch_huge_2d,
    mot_patch_giant_2d,
    mot_predictor_patch_tiny_2d,
    mot_predictor_patch_small_2d,
    mot_predictor_patch_base_2d,
    mot_predictor_patch_large_2d,
    mot_predictor_patch_huge_2d,
    mot_predictor_patch_giant_2d,
)
from .token_layout import TokenLayout

MODEL_FACTORIES = {
    factory.__name__: factory
    for factory in (
        mot_tiny_1d,
        mot_small_1d,
        mot_base_1d,
        mot_large_1d,
        mot_huge_1d,
        mot_giant_1d,

        mot_patch_tiny_1d,
        mot_patch_small_1d,
        mot_patch_base_1d,
        mot_patch_large_1d,
        mot_patch_huge_1d,
        mot_patch_giant_1d,

        mot_tiny_2d,
        mot_small_2d,
        mot_base_2d,
        mot_large_2d,
        mot_huge_2d,
        mot_giant_2d,

        mot_patch_tiny_2d,
        mot_patch_small_2d,
        mot_patch_base_2d,
        mot_patch_large_2d,
        mot_patch_huge_2d,
        mot_patch_giant_2d,
    )
}

PREDICTOR_FACTORIES = {
    factory.__name__: factory
    for factory in (
        mot_predictor_tiny_1d,
        mot_predictor_small_1d,
        mot_predictor_base_1d,
        mot_predictor_large_1d,
        mot_predictor_huge_1d,
        mot_predictor_giant_1d,

        mot_predictor_patch_tiny_1d,
        mot_predictor_patch_small_1d,
        mot_predictor_patch_base_1d,
        mot_predictor_patch_large_1d,
        mot_predictor_patch_huge_1d,
        mot_predictor_patch_giant_1d,

        mot_predictor_tiny_2d,
        mot_predictor_small_2d,
        mot_predictor_base_2d,
        mot_predictor_large_2d,
        mot_predictor_huge_2d,
        mot_predictor_giant_2d,

        mot_predictor_patch_tiny_2d,
        mot_predictor_patch_small_2d,
        mot_predictor_patch_base_2d,
        mot_predictor_patch_large_2d,
        mot_predictor_patch_huge_2d,
        mot_predictor_patch_giant_2d,
    )
}

PATCH_MODEL_NAMES = frozenset(
    factory.__name__
    for factory in (
        mot_patch_tiny_1d,
        mot_patch_small_1d,
        mot_patch_base_1d,
        mot_patch_large_1d,
        mot_patch_huge_1d,
        mot_patch_giant_1d,
        mot_patch_tiny_2d,
        mot_patch_small_2d,
        mot_patch_base_2d,
        mot_patch_large_2d,
        mot_patch_huge_2d,
        mot_patch_giant_2d,
    )
)
PATCH_PREDICTOR_NAMES = frozenset(
    factory.__name__
    for factory in (
        mot_predictor_patch_tiny_1d,
        mot_predictor_patch_small_1d,
        mot_predictor_patch_base_1d,
        mot_predictor_patch_large_1d,
        mot_predictor_patch_huge_1d,
        mot_predictor_patch_giant_1d,
        mot_predictor_patch_tiny_2d,
        mot_predictor_patch_small_2d,
        mot_predictor_patch_base_2d,
        mot_predictor_patch_large_2d,
        mot_predictor_patch_huge_2d,
        mot_predictor_patch_giant_2d,
    )
)
MODEL_KINDS = {
    name: ("2d" if factory in {
        mot_tiny_2d, mot_small_2d, mot_base_2d,
        mot_large_2d, mot_huge_2d, mot_giant_2d,
        mot_patch_tiny_2d, mot_patch_small_2d, mot_patch_base_2d,
        mot_patch_large_2d, mot_patch_huge_2d, mot_patch_giant_2d,
    } else "1d")
    for name, factory in MODEL_FACTORIES.items()
}
PREDICTOR_KINDS = {
    name: ("2d" if factory in {
        mot_predictor_tiny_2d, mot_predictor_small_2d, mot_predictor_base_2d,
        mot_predictor_large_2d, mot_predictor_huge_2d, mot_predictor_giant_2d,
        mot_predictor_patch_tiny_2d, mot_predictor_patch_small_2d,
        mot_predictor_patch_base_2d, mot_predictor_patch_large_2d,
        mot_predictor_patch_huge_2d, mot_predictor_patch_giant_2d,
    } else "1d")
    for name, factory in PREDICTOR_FACTORIES.items()
}

__all__ = [
    "MODEL_FACTORIES",
    "MODEL_KINDS",
    "PATCH_MODEL_NAMES",
    "PATCH_PREDICTOR_NAMES",
    "PREDICTOR_KINDS",
    "MotionFeatureTokenizer2D",
    "MotionTransformer1D",
    "MotionTransformer2D",
    "MotionTransformerPredictor1D",
    "MotionTransformerPredictor2D",
    "MotionPatchTransformer1D",
    "MotionPatchTransformerPredictor1D",
    "MotionPatchTransformer2D",
    "MotionPatchTransformerPredictor2D",
    "BODY_TOKEN_OFFSET",
    "SPATIAL_GROUPINGS",
    "get_spatial_grouping",
    "spatial_patch_signature",
    "TRAJECTORY_FIELDS",
    "TRAJECTORY_TOKEN_INDEX",
    "TokenLayout",
    *MODEL_FACTORIES,
    *PREDICTOR_FACTORIES,
]
