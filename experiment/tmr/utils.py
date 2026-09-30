"""Small shared helpers for the standalone alignment experiment."""

from __future__ import annotations

import torch

from experiment.linear_probe.features import resolve_device as _resolve_device


def resolve_device(value: str) -> torch.device:
    # torch.cuda.set_device requires an index even though torch.device accepts
    # the common CLI shorthand "cuda".
    return _resolve_device("cuda:0" if value == "cuda" else value)
