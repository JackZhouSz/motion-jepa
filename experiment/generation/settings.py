"""Configuration for raw-motion rectified-flow training and inference."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
PATH_FIELDS = ("jepa_checkpoint", "dataset_root", "stats_path", "cache_root", "output",
               "deterministic_checkpoint")


def _read_yaml(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Generation config must be a YAML mapping")
    return value


def _merge(config: dict, requested: dict) -> None:
    unknown = set(requested) - set(config)
    if unknown:
        raise ValueError(f"Unknown generation config fields: {sorted(unknown)}")
    for key, value in requested.items():
        if key == "flow":
            if not isinstance(value, dict):
                raise ValueError("flow must be an architecture mapping")
            unknown_flow = set(value) - set(config["flow"])
            if unknown_flow:
                raise ValueError(f"Unknown flow config fields: {sorted(unknown_flow)}")
            config[key].update(value)
        else:
            config[key] = value


def _validate(config: dict) -> dict:
    for key in PATH_FIELDS:
        if config[key] is not None:
            value = Path(config[key]).expanduser()
            config[key] = str((value if value.is_absolute() else PROJECT_ROOT / value).resolve())
    # PyYAML can represent scientific literals such as `3e-4` as strings.
    for key in ("lr", "final_lr", "gradient_clip", "weight_decay", "guidance_scale",
                "condition_dropout", "ema_decay"):
        if isinstance(config[key], bool):
            raise ValueError(f"{key} must be a number")
        try:
            config[key] = float(config[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a number") from error
    for key in ("epochs", "batch_size", "cache_batch_size", "steps", "num_samples", "diagnostic_count"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "evaluation_seed", "num_workers", "warmup_epochs", "limit_train", "limit_val",
                "limit_test", "export_count"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("lr", "final_lr", "gradient_clip"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("weight_decay", "guidance_scale"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    if config["final_lr"] > config["lr"]:
        raise ValueError("final_lr cannot exceed lr")
    if not 0 <= float(config["condition_dropout"]) <= 1:
        raise ValueError("condition_dropout must be in [0, 1]")
    if not 0 <= float(config["ema_decay"]) < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    for key in ("use_bfloat16", "tensorboard"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    # Keep settings import cheap; architecture validation happens when needed.
    return config


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    config = copy.deepcopy(_read_yaml(DEFAULT_CONFIG))
    if path is not None:
        _merge(config, _read_yaml(path))
    if overrides:
        _merge(config, overrides)
    return _validate(config)


def load_checkpoint_config(saved_config: dict, path: str | Path | None = None,
                           overrides: dict | None = None) -> dict:
    """Defaults < training settings < requested YAML < explicit CLI options."""
    config = copy.deepcopy(_read_yaml(DEFAULT_CONFIG))
    _merge(config, saved_config)
    if path is not None:
        _merge(config, _read_yaml(path))
    if overrides:
        _merge(config, overrides)
    return _validate(config)
