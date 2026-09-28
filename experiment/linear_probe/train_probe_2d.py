"""Group-aware frozen linear probing for 2D MotionJEPA encoders."""

from __future__ import annotations

import json

from .features import SPATIAL_FLATTEN_POOLING
from .train_probe import build_parser, run


def main() -> None:
    parser = build_parser()
    parser.description = (
        "2D MotionJEPA linear probe: temporal mean, spatial flatten, one Linear head"
    )
    parser.set_defaults(pooling=SPATIAL_FLATTEN_POOLING)
    summary = run(parser.parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
