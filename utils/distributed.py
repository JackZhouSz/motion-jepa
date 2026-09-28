"""Distributed runtime helpers for local spawning, torchrun, and SLURM."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    backend: str | None

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _environment_rank() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        return (
            int(os.environ["SLURM_PROCID"]),
            int(os.environ["SLURM_NTASKS"]),
            int(os.environ.get("SLURM_LOCALID", 0)),
        )
    return 0, 1, 0


def init_distributed(
    device: torch.device,
    port: int = 40112,
    timeout_seconds: int | None = None,
) -> DistributedContext:
    if dist.is_available() and dist.is_initialized():
        rank, world_size, local_rank = _environment_rank()
        return DistributedContext(rank, world_size, local_rank, dist.get_backend())
    rank, world_size, local_rank = _environment_rank()
    if world_size <= 1:
        return DistributedContext(0, 1, local_rank, None)
    backend = "nccl" if device.type == "cuda" else "gloo"
    os.environ.setdefault("MASTER_ADDR", socket.gethostname())
    os.environ.setdefault("MASTER_PORT", str(port))
    kwargs = {"backend": backend, "init_method": "env://"}
    if timeout_seconds is not None:
        kwargs["timeout"] = timedelta(seconds=timeout_seconds)
    dist.init_process_group(**kwargs)
    return DistributedContext(rank, world_size, local_rank, backend)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def all_gather_objects(value):
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    output = [None] * dist.get_world_size()
    dist.all_gather_object(output, value)
    return output


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "DistributedContext",
    "all_gather_objects",
    "barrier",
    "cleanup_distributed",
    "init_distributed",
    "reduce_mean",
]
