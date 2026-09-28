"""Launch the independent MotionJEPA dataset viewer."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from visualization import MotionJEPADatasetViewer


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a MotionJEPA NPY dataset with viser.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "dataset/bones-seed-processed",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sample_id")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--host", default=os.environ.get("SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "6006")))
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--normalized", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = MotionJEPADatasetViewer(
        args.input,
        split=args.split,
        limit=args.limit,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        host=args.host,
        port=args.port,
        mesh=args.mesh,
        normalized=args.normalized,
    )
    print(f"Found {len(viewer.entries)} NPY motion sample(s) under {args.input}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    main()
