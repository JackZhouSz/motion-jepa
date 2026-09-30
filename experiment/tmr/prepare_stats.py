"""Compute train-only JEPA mean/std from an existing cache, without extraction."""

import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_CACHE_ROOT
from .features import prepare_jepa_statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_jepa_statistics(args.cache_root, recompute=args.recompute), indent=2))


if __name__ == "__main__":
    main()
