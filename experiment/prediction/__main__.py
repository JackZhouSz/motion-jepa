"""Command-line entry points for cache preparation and motion reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

from .settings import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-cache", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        for option in ("jepa-checkpoint", "dataset-root", "stats-path", "cache-root", "output"):
            command.add_argument(f"--{option}", type=str)
        command.add_argument("--device")
        for option in ("seed", "epochs", "batch-size", "cache-batch-size", "num-workers", "warmup-epochs",
                       "limit-train", "limit-val", "limit-test", "export-count"):
            command.add_argument(f"--{option}", type=int)
        for option in ("lr", "final-lr", "weight-decay", "gradient-clip"):
            command.add_argument(f"--{option}", type=float)
        command.add_argument("--bfloat16", dest="use_bfloat16", action=argparse.BooleanOptionalAction, default=None)
        command.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None)
        if name == "prepare-cache":
            command.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
        elif name == "train":
            command.add_argument("--resume", type=Path)
        else:
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--split", choices=("train", "val", "test"), default="test")
    viewer = commands.add_parser("visualize")
    viewer.add_argument("--results", type=Path, required=True)
    viewer.add_argument("--host", default="0.0.0.0")
    viewer.add_argument("--port", type=int, default=8080)
    viewer.add_argument("--mesh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "visualize":
        from .visualize import visualize_results
        visualize_results(args.results, host=args.host, port=args.port, mesh=args.mesh)
        return
    overrides = {key: value for key, value in vars(args).items()
                 if key not in {"command", "config", "splits", "resume", "checkpoint", "split"} and value is not None}
    if args.command == "evaluate":
        from .evaluate import evaluate_checkpoint
        evaluate_checkpoint(args.checkpoint, split=args.split, config_path=args.config, overrides=overrides)
    elif args.command == "train":
        from .train import train
        train(load_config(args.config, overrides), resume=args.resume)
    else:
        from .data import prepare_caches
        prepare_caches(load_config(args.config, overrides), splits=tuple(args.splits))


if __name__ == "__main__":
    main()
