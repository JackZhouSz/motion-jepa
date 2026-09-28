"""Frozen-feature linear probing and classification dataset utilities."""

from .dataset import (
    ClassificationLabelIndex,
    ClassificationTokenDataset,
    MultiLabelIndex,
    MultiLabelMotionDataset,
    SingleLabelIndex,
    SingleLabelMotionDataset,
    build_classification_datasets,
    load_classification_label_index,
)
from .cnn import MotionCNNClassifier
from .features import (
    CACHE_FORMAT_VERSION,
    GLOBAL_MEAN_POOLING,
    SPLITS,
    SPATIAL_FLATTEN_POOLING,
    Metrics,
    _validate_feature_cache,
    build_cache_metadata,
    extract_features,
    load_frozen_encoder,
    load_or_extract_split,
    pool_encoder_output,
    resolve_device,
    resolve_pretraining_stats,
)
from .linear import RawMotionLinearClassifier
from .transformer import MotionTransformerClassifier

__all__ = [
    "CACHE_FORMAT_VERSION",
    "GLOBAL_MEAN_POOLING",
    "SPLITS",
    "SPATIAL_FLATTEN_POOLING",
    "Metrics",
    "MotionCNNClassifier",
    "RawMotionLinearClassifier",
    "MotionTransformerClassifier",
    "ClassificationLabelIndex",
    "ClassificationTokenDataset",
    "MultiLabelIndex",
    "MultiLabelMotionDataset",
    "SingleLabelIndex",
    "SingleLabelMotionDataset",
    "build_cache_metadata",
    "build_classification_datasets",
    "extract_features",
    "load_frozen_encoder",
    "load_or_extract_split",
    "load_classification_label_index",
    "pool_encoder_output",
    "resolve_device",
    "resolve_pretraining_stats",
    "_validate_feature_cache",
]
