"""Prepare reusable frozen text tokens and optional frozen MotionJEPA tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_ANNOTATIONS_PATH, DEFAULT_CACHE_ROOT, DEFAULT_DATASET_ROOT, DEFAULT_TEXT_MODEL
from .features import prepare_caches


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-source", choices=("raw", "jepa"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotations-path", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--checkpoint-key", choices=("target_encoder", "encoder"), default="target_encoder")
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--text-revision")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int)
    return parser


def main(argv=None):
    metadata = prepare_caches(**vars(build_parser().parse_args(argv)))
    print(json.dumps({key: metadata[key] for key in (
        "input_source", "motion_dim", "text_dim", "split_counts", "filtered_counts"
    )}, indent=2))


if __name__ == "__main__":
    main()
