"""Local and torchrun entry point for Motion-JEPA pretraining."""

from __future__ import annotations

import argparse
import logging
import os
import pprint
import sys

import torch
import torch.multiprocessing as mp
import yaml

from train import main as train_main
from utils.distributed import cleanup_distributed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Motion-JEPA")
    parser.add_argument(
        "--config", "--fname",
        dest="config",
        default="configs/mjepa_patch_2d_tiny_coarse7.yaml",
        help="YAML config",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="Local devices, for example --devices cuda:0 cuda:1",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Force a single process on the first configured device",
    )
    parser.add_argument("--master-port", type=int, default=40112)
    return parser


def _load_config(fname: str) -> dict:
    with open(fname, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a mapping: {fname}")
    return config


def _run(fname: str, device: str):
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    config = _load_config(fname)
    if rank == 0:
        logging.getLogger(__name__).info("Loaded %s\n%s", fname, pprint.pformat(config))
    try:
        return train_main(config, device=device)
    finally:
        cleanup_distributed()


def _spawn_worker(
    local_rank: int,
    fname: str,
    devices: list[str],
    master_port: int,
):
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(len(devices))
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    return _run(fname, devices[local_rank])


def _is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def launch(args: argparse.Namespace):
    config = getattr(args, "config", getattr(args, "fname", None))
    if config is None:
        raise ValueError("A training config path is required")
    if _is_torchrun():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        return _run(config, device)
    devices = list(args.devices[:1] if args.debug else args.devices)
    if len(devices) == 1:
        return _spawn_worker(0, config, devices, args.master_port)
    mp.spawn(
        _spawn_worker,
        args=(config, devices, args.master_port),
        nprocs=len(devices),
        join=True,
    )
    return None


if __name__ == "__main__":
    launch(build_parser().parse_args())
