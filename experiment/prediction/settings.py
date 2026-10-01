"""Shared decoder experiment configuration, independent of the current directory."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
PATH_FIELDS = ("jepa_checkpoint", "dataset_root", "stats_path", "cache_root", "output")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Merge experiment defaults; relative paths always refer to the repository."""
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(defaults)
    if path is not None:
        requested = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(requested, dict):
            raise ValueError("Prediction config must be a YAML mapping")
        unknown = set(requested) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown prediction config fields: {sorted(unknown)}")
        decoder = {**defaults["decoder"], **requested.get("decoder", {})}
        config.update(requested)
        config["decoder"] = decoder
    for key, value in (overrides or {}).items():
        if value is not None:
            if key not in config:
                raise ValueError(f"Unknown prediction override: {key}")
            config[key] = value
    for key in PATH_FIELDS:
        if config[key] is not None:
            value = Path(config[key]).expanduser()
            config[key] = str((value if value.is_absolute() else PROJECT_ROOT / value).resolve())
    for key in ("epochs", "batch_size", "cache_batch_size"):
        if isinstance(config[key], bool) or int(config[key]) != config[key] or int(config[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "warmup_epochs", "limit_train", "limit_val", "limit_test", "export_count"):
        if isinstance(config[key], bool) or int(config[key]) != config[key] or int(config[key]) < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("lr", "final_lr", "gradient_clip"):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(float(config["weight_decay"])) or float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if config["final_lr"] > config["lr"]:
        raise ValueError("final_lr cannot exceed lr")
    return config
