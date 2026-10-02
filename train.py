"""MotionJEPA pretraining loop shared by the 1D and 2D variants."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import shutil
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel

from dataset import make_motion_dataset
from helper import (
    architecture_signature,
    architecture_signature_from_config,
    init_mjepa_model_from_config,
    init_opt,
    normalize_architecture_signature,
)
from mask import (
    MaskCollator1D,
    MaskCollator1DV2,
    MaskCollator2D,
    PatchBodyRegionSegmentMaskCollator2D,
    PatchMaskCollator1D,
    PatchMaskCollator1DV2,
    PatchMaskCollator2D,
    PatchRandomBodySegmentMaskCollator2D,
    PatchRandomSpatialSegmentMaskCollator2D,
)
from model import MODEL_FACTORIES, PREDICTOR_FACTORIES, TokenLayout
from mask.utils import (
    apply_index_masks,
    gather_grid_masks,
    index_mask_validity,
    repeat_mask_blocks,
)
from utils.distributed import (
    all_gather_objects,
    barrier,
    init_distributed,
    reduce_mean,
)
from utils.logging import AverageMeter, CSVLogger, grad_logger
from utils.schedulers import LinearMomentumSchedule


logger = logging.getLogger(__name__)


def _unwrapped(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def _make_tensorboard_writer(log_args: dict, output: Path, purge_step: int):
    if not bool(log_args.get("tensorboard", False)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorBoard logging is enabled; install it with `pip install tensorboard`."
        ) from error
    log_dir = output / str(log_args.get("tensorboard_folder", "tensorboard"))
    return SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)


def _write_tensorboard_interval(
    writer,
    *,
    global_step: int,
    epoch: int,
    loss: float,
    learning_rate: float,
    weight_decay: float,
    time_ms: float,
    memory_mib: float,
    grad_first: float,
    grad_last: float,
    grad_average: float,
) -> None:
    if writer is None:
        return
    scalars = {
        "train/loss": loss,
        "train/learning_rate": learning_rate,
        "train/weight_decay": weight_decay,
        "train/iteration_time_ms": time_ms,
        "train/gpu_memory_mib": memory_mib,
        "train/gradient_first_layer": grad_first,
        "train/gradient_last_layer": grad_last,
        "train/gradient_average": grad_average,
        "train/epoch": float(epoch),
    }
    for name, value in scalars.items():
        writer.add_scalar(name, value, global_step)
    writer.flush()


def _write_tensorboard_linear_probe(
    writer,
    *,
    global_step: int,
    summary: dict,
    best_val_top1: float | None,
) -> None:
    if writer is None:
        return
    scalars = {
        f"linear_probe/test_{name}": float(summary["test"][name])
        for name in ("loss", "top1_accuracy", "macro_accuracy", "top5_accuracy")
        if name in summary["test"]
    }
    if summary.get("validation_used", True):
        scalars["linear_probe/val_top1_accuracy"] = float(
            summary["best_val"]["top1_accuracy"]
        )
        assert best_val_top1 is not None
        scalars["linear_probe/best_val_top1_accuracy"] = float(best_val_top1)
        scalars["linear_probe/probe_best_epoch"] = float(summary["best_epoch"])
    for name, value in scalars.items():
        writer.add_scalar(name, value, global_step)
    writer.flush()


def _write_tensorboard_babel_probes(
    writer,
    *,
    global_step: int,
    summaries: dict[str, dict],
    state: dict[str, dict],
    probe_name: str = "linear_probe",
) -> None:
    if writer is None:
        return
    for name, summary in summaries.items():
        prefix = f"{probe_name}/{name}"
        for metric in ("loss", "mean_average_precision", "top1_hit",
                       "top1_label_row_accuracy", "top5_hit"):
            if metric in summary["best_val"]:
                writer.add_scalar(f"{prefix}/val_{metric}",
                                  float(summary["best_val"][metric]), global_step)
        writer.add_scalar(
            f"{prefix}/best_val_mean_average_precision",
            float(state[name]["best_val_map"]), global_step,
        )
        writer.add_scalar(
            f"{prefix}/probe_best_epoch", float(summary["best_epoch"]), global_step,
        )
        writer.add_scalar(
            f"{prefix}/pretrain_best_epoch", float(state[name]["best_epoch"]), global_step,
        )
    writer.flush()


def _flatten_numeric_metrics(prefix: str, values: dict) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(value, dict):
            flattened.update(_flatten_numeric_metrics(key, value))
        elif isinstance(value, (int, float)):
            flattened[key] = float(value)
    return flattened


def _write_tensorboard_online_metrics(
    writer,
    *,
    global_step: int,
    summary: dict,
) -> None:
    if writer is None:
        return
    if "retrieval" in summary:
        from experiment.motion_online_metrics import tensorboard_metrics

        scalars = tensorboard_metrics(summary)
    else:
        # Legacy 2D evaluator: numeric metadata is not a learning curve.
        metric_names = {
            "rankme", "mean_std", "mean_cosine", "mean_off_diagonal_cosine",
            "effective_rank", "largest_eigenvalue_ratio", "off_diagonal_abs_mean",
            "mse", "smooth_l1", "cosine_similarity", "prediction_gain",
            "baseline_mse", "trajectory_reliance", "body_temporal_std",
            "body_joint_std", "trajectory_temporal_std",
        }
        scalars = {
            key: value for key, value in _flatten_numeric_metrics("", summary).items()
            if key.rsplit("/", 1)[-1] in metric_names
        }
    for name, value in scalars.items():
        writer.add_scalar(f"online_metrics/{name}", value, global_step)
    if summary.get("overhead_percent") is not None:
        writer.add_scalar("online_metrics/timing/overhead_percent",
                          float(summary["overhead_percent"]), global_step)
    writer.flush()


def _write_tensorboard_segmentation_probe(writer, *, global_step, summary, state):
    if writer is None:
        return
    for dataset in ("babel-120", "babel-60"):
        for metric in ("frame_map", "bce", "micro_f1"):
            value = summary["best_val"][dataset][metric]
            if value is not None:
                writer.add_scalar(f"segmentation_probe/{dataset}/val_{metric}",
                                  float(value), global_step)
    prefix = "segmentation_probe/babel-120"
    writer.add_scalar(f"{prefix}/best_val_frame_map", float(state["best_val_map"]), global_step)
    writer.add_scalar(f"{prefix}/probe_best_epoch", float(summary["best_epoch"]), global_step)
    writer.add_scalar(f"{prefix}/pretrain_best_epoch", float(state["best_epoch"]), global_step)
    timing_tags = {
        "feature_extraction_or_cache_load_seconds": "feature_extraction_seconds",
        "head_seconds": "head_training_seconds",
        "evaluation_call_seconds": "total_seconds",
    }
    for key, tag in timing_tags.items():
        if key in summary.get("timings", {}):
            writer.add_scalar(f"segmentation_probe/timing/{tag}",
                              float(summary["timings"][key]), global_step)
    writer.flush()


@torch.no_grad()
def update_ema(online, target, momentum: float) -> None:
    for online_parameter, target_parameter in zip(
        _unwrapped(online).parameters(), target.parameters()
    ):
        target_parameter.mul_(momentum).add_(
            online_parameter.detach(), alpha=1.0 - momentum
        )


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _evaluate_online_probe_preserving_rng(evaluator, encoder) -> dict:
    rng_state = capture_rng_state()
    try:
        return evaluator.evaluate(encoder)
    finally:
        restore_rng_state(rng_state)


def _evaluate_frozen_encoder_preserving_rng(evaluator, encoder, **kwargs) -> dict:
    """Fit heads or run inference without changing the pretraining state."""
    rng_state = capture_rng_state()
    modes = [(module, module.training) for module in encoder.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in encoder.parameters()]
    try:
        encoder.eval()
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        return evaluator.evaluate(encoder, **kwargs)
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        restore_rng_state(rng_state)


def _run_rank_zero(function, *, is_main: bool):
    """Propagate rank-zero failures before other ranks enter later collectives."""
    result = None
    if is_main:
        try:
            result = {"value": function(), "error": None}
        except Exception:
            result = {"value": None, "error": traceback.format_exc()}
    result = all_gather_objects(result)[0]
    if result["error"] is not None:
        raise RuntimeError(f"Rank-zero evaluation failed:\n{result['error']}")
    return result["value"]


def _attentive_probe_args(config: dict) -> dict:
    attentive = config.get("attentive_probe", {})
    if not isinstance(attentive, dict):
        raise ValueError("attentive_probe config must be a mapping")
    linear = config.get("linear_probe", {})
    if not isinstance(linear, dict):
        raise ValueError("linear_probe config must be a mapping")
    return {
        "datasets": linear.get("datasets"),
        "frequency": linear.get("frequency", config.get("logging", {}).get("checkpoint_freq", 50)),
        **attentive,
    }


def _probe_protocols(config: dict, *, legacy: bool = False) -> dict:
    """Identify comparable scores; frequency does not affect a fresh probe fit."""
    shared = {key: config.get("data", {}).get(key) for key in
              ("root_path", "stats_path", "num_frames", "motion_dim", "fps")}
    shared["use_bfloat16"] = config.get("meta", {}).get("use_bfloat16", False)
    common = dict(epochs=50, batch_size=256, feature_batch_size=256, seed=42)
    defaults = {
        "linear_probe": {**common, "lr": 0.3, "momentum": 0.9, "weight_decay": 0.0,
                         "pooling": "valid_token_mean", "standardize": not legacy},
        "attentive_probe": {**common, "lr": 3e-4, "weight_decay": 0.05,
                            "warmup_epochs": 5, "final_lr": 1e-6,
                            "gradient_clip": 1.0, "num_heads": 6},
    }
    protocols = {}
    for kind, values in defaults.items():
        options = _attentive_probe_args(config) if kind == "attentive_probe" else config.get(kind, {})
        protocols[kind] = {**shared, **{key: options.get(key, value) for key, value in values.items()},
                           "datasets": options.get("datasets"),
                           "dataset_root": options.get("dataset_root", "dataset/100style-soma77-processed")}
    return protocols


def _evaluate_online_metrics_preserving_rng(evaluator, encoder, predictor) -> dict:
    rng_state = capture_rng_state()
    encoder_training = encoder.training
    predictor_training = predictor.training
    try:
        encoder.eval()
        predictor.eval()
        return evaluator.evaluate(encoder, predictor)
    finally:
        encoder.train(encoder_training)
        predictor.train(predictor_training)
        restore_rng_state(rng_state)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy an already-complete checkpoint without deserializing it."""
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _upsert_epoch_jsonl(path: Path, summary: dict) -> None:
    """Replace a retried evaluation without duplicating its learning-curve point."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    key = (summary["pretrain_epoch"], summary.get("protocol_hash"))
    records = [record for record in records
               if (record["pretrain_epoch"], record.get("protocol_hash")) != key]
    records.append(summary)
    records.sort(key=lambda record: record["pretrain_epoch"])
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    *,
    encoder,
    predictor,
    target_encoder,
    optimizer,
    scaler,
    lr_scheduler,
    wd_scheduler,
    momentum_scheduler,
    mask_collator,
    next_epoch: int,
    global_step: int,
    loss: float,
    world_size: int,
    rank: int,
    config: dict,
    architecture: dict | None = None,
    linear_probe_latest: dict | None = None,
    best_probe_val_top1: float = float("-inf"),
    best_probe_epoch: int | None = None,
    babel_probe_state: dict[str, dict] | None = None,
    attentive_probe_state: dict[str, dict] | None = None,
    online_metrics_latest: dict | None = None,
    segmentation_probe_state: dict | None = None,
) -> None:
    rng_states = all_gather_objects(capture_rng_state())
    mask_states = all_gather_objects(mask_collator.state_dict())
    if rank != 0:
        return
    payload = {
        "format_version": 1,
        "encoder": _unwrapped(encoder).state_dict(),
        "predictor": _unwrapped(predictor).state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "wd_scheduler": wd_scheduler.state_dict(),
        "momentum_scheduler": momentum_scheduler.state_dict(),
        "mask_states": mask_states,
        "rng_states": rng_states,
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "world_size": int(world_size),
        "loss": float(loss),
        "config": config,
        "linear_probe_latest": linear_probe_latest,
        "best_probe_val_top1": float(best_probe_val_top1),
        "best_probe_epoch": best_probe_epoch,
        "babel_probe_state": babel_probe_state,
        "attentive_probe_state": attentive_probe_state,
        "probe_protocols": _probe_protocols(config),
        "online_metrics_latest": online_metrics_latest,
        "segmentation_probe_state": segmentation_probe_state,
    }
    if architecture is not None:
        payload["architecture"] = architecture
    _atomic_torch_save(payload, path)


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    encoder,
    predictor,
    target_encoder,
    optimizer,
    scaler,
    lr_scheduler,
    wd_scheduler,
    momentum_scheduler,
    mask_collator,
    rank: int,
    world_size: int,
    architecture: dict | None = None,
    linear_probe_state: dict | None = None,
    babel_probe_state: dict[str, dict] | None = None,
    attentive_probe_state: dict[str, dict] | None = None,
    probe_protocols: dict | None = None,
    online_metrics_state: dict | None = None,
    segmentation_probe_state: dict | None = None,
) -> tuple[int, int]:
    # Full training checkpoints contain trusted local Python/NumPy RNG state,
    # optimizer state, and scheduler state in addition to tensor weights.
    # PyTorch 2.6 defaults weights_only=True, which cannot restore that payload.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported MotionJEPA checkpoint format: {path}")
    if int(checkpoint["world_size"]) != world_size:
        raise ValueError(
            f"Exact resume requires world_size={checkpoint['world_size']}, got {world_size}"
        )
    if architecture is not None:
        saved_architecture = checkpoint.get("architecture")
        if saved_architecture is None:
            saved_architecture = architecture_signature_from_config(checkpoint["config"])
        if normalize_architecture_signature(saved_architecture) != normalize_architecture_signature(architecture):
            raise ValueError(
                "Checkpoint architecture differs from the requested run: "
                f"checkpoint={saved_architecture}, requested={architecture}"
            )
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    target_encoder.load_state_dict(checkpoint["target_encoder"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        if checkpoint["scaler"] is None:
            raise ValueError("Checkpoint has no scaler state for float16 resume")
        scaler.load_state_dict(checkpoint["scaler"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    wd_scheduler.load_state_dict(checkpoint["wd_scheduler"])
    momentum_scheduler.load_state_dict(checkpoint["momentum_scheduler"])
    mask_collator.load_state_dict(checkpoint["mask_states"][rank])
    restore_rng_state(checkpoint["rng_states"][rank])
    if linear_probe_state is not None:
        linear_probe_state.update(
            latest=checkpoint.get("linear_probe_latest"),
            best_val_top1=float(checkpoint.get("best_probe_val_top1", float("-inf"))),
            best_epoch=checkpoint.get("best_probe_epoch"),
        )
    if babel_probe_state is not None:
        saved = checkpoint.get("babel_probe_state") or {}
        for name, state in babel_probe_state.items():
            state.update(saved.get(name, {}))
    if attentive_probe_state is not None:
        saved = checkpoint.get("attentive_probe_state") or {}
        for name, state in attentive_probe_state.items():
            state.update(saved.get(name, {}))
    if probe_protocols is not None:
        saved_protocols = checkpoint.get("probe_protocols") or _probe_protocols(checkpoint["config"], legacy=True)
        for kind, states in (("linear_probe", babel_probe_state), ("attentive_probe", attentive_probe_state)):
            if saved_protocols.get(kind) == probe_protocols[kind]:
                continue
            logger.info("Resetting %s best scores: probe protocol changed", kind)
            if states is not None:
                for state in states.values():
                    state.update(latest=None, best_val_map=float("-inf"), best_epoch=None)
            if kind == "linear_probe" and linear_probe_state is not None:
                linear_probe_state.update(latest=None, best_val_top1=float("-inf"), best_epoch=None)
    if online_metrics_state is not None:
        online_metrics_state["latest"] = checkpoint.get("online_metrics_latest")
    if segmentation_probe_state is not None:
        segmentation_probe_state.update(checkpoint.get("segmentation_probe_state") or {})
    return int(checkpoint["next_epoch"]), int(checkpoint["global_step"])


def _build_mask_collator(args: dict, layout: TokenLayout):
    mask = args["mask"]
    version = str(mask.get("version", "v1"))
    if version not in ("v1", "v2"):
        raise ValueError("mask.version must be 'v1' or 'v2'")
    if version == "v2" and layout.kind != "1d":
        raise ValueError("mask.version='v2' currently supports only 1D masks")
    context_selection = str(mask.get("context_selection", "all" if version == "v2" else "prefix"))
    choices = ("all", "prefix", "random") if version == "v2" else ("prefix", "random")
    if context_selection not in choices:
        raise ValueError(f"mask.context_selection must be one of {choices}")
    if layout.kind != "1d" and context_selection != "prefix":
        raise ValueError("mask.context_selection='random' requires a 1D multiblock mask")
    strategy = str(mask.get("strategy", "multiblock"))
    if version == "v2" and strategy != "multiblock":
        raise ValueError("mask.version='v2' requires mask.strategy='multiblock'")
    if strategy == "random_spatial_segment":
        if layout.kind != "2d" or not layout.patchified:
            raise ValueError(
                "mask.strategy='random_spatial_segment' requires patchified 2D tokens"
            )
        patch = args["patch"]
        if int(mask["num_enc_masks"]) != 1:
            raise ValueError(
                "mask.strategy='random_spatial_segment' requires one encoder mask"
            )
        if not bool(mask.get("allow_target_overlap", False)):
            raise ValueError(
                "random_spatial_segment requires allow_target_overlap=true"
            )
        if bool(mask.get("allow_context_target_overlap", False)):
            raise ValueError(
                "random_spatial_segment requires allow_context_target_overlap=false"
            )
        return PatchRandomSpatialSegmentMaskCollator2D(
            raw_num_frames=layout.raw_num_frames,
            raw_num_joints=int(layout.raw_num_joints),
            token_num_joints=int(layout.token_num_joints),
            temporal_patch_size=layout.temporal_patch_size,
            spatial_grouping=str(patch["spatial_grouping"]),
            spatial_pooling=str(patch["spatial_pooling"]),
            pred_frame_mask_ratio=tuple(mask["pred_frame_mask_ratio"]),
            pred_spatial_mask_count=int(mask["pred_spatial_mask_count"]),
            target_union_ratio=tuple(mask["target_union_ratio"]),
            npred=int(mask["num_pred_masks"]),
        )
    if strategy == "random_body_segment":
        if layout.kind != "2d" or not layout.patchified:
            raise ValueError(
                "mask.strategy='random_body_segment' requires patchified 2D tokens"
            )
        patch = args["patch"]
        if str(patch["spatial_grouping"]) != "coarse7":
            raise ValueError(
                "mask.strategy='random_body_segment' requires spatial_grouping='coarse7'"
            )
        if int(mask["num_enc_masks"]) != 1:
            raise ValueError(
                "mask.strategy='random_body_segment' requires one encoder mask"
            )
        if bool(mask["allow_overlap"]):
            raise ValueError(
                "mask.strategy='random_body_segment' requires allow_overlap=false"
            )
        return PatchRandomBodySegmentMaskCollator2D(
            raw_num_frames=layout.raw_num_frames,
            raw_num_joints=int(layout.raw_num_joints),
            token_num_joints=int(layout.token_num_joints),
            temporal_patch_size=layout.temporal_patch_size,
            spatial_grouping=str(patch["spatial_grouping"]),
            spatial_pooling=str(patch["spatial_pooling"]),
            pred_frame_mask_ratio=tuple(mask["pred_frame_mask_ratio"]),
            body_mask_ratio=tuple(mask["body_mask_ratio"]),
            npred=int(mask["num_pred_masks"]),
        )
    if strategy == "body_region_segment":
        if layout.kind != "2d" or not layout.patchified:
            raise ValueError(
                "mask.strategy='body_region_segment' requires patchified 2D tokens"
            )
        patch = args["patch"]
        if str(patch["spatial_grouping"]) != "coarse7":
            raise ValueError(
                "mask.strategy='body_region_segment' requires spatial_grouping='coarse7'"
            )
        if int(mask["num_enc_masks"]) != 1 or int(mask["num_pred_masks"]) != 1:
            raise ValueError(
                "mask.strategy='body_region_segment' requires one encoder and one target mask"
            )
        if bool(mask["allow_overlap"]):
            raise ValueError(
                "mask.strategy='body_region_segment' requires allow_overlap=false"
            )
        return PatchBodyRegionSegmentMaskCollator2D(
            raw_num_frames=layout.raw_num_frames,
            raw_num_joints=int(layout.raw_num_joints),
            token_num_joints=int(layout.token_num_joints),
            temporal_patch_size=layout.temporal_patch_size,
            spatial_grouping=str(patch["spatial_grouping"]),
            spatial_pooling=str(patch["spatial_pooling"]),
            pred_frame_mask_ratio=tuple(mask["pred_frame_mask_ratio"]),
            graph_mask_ratio=tuple(mask["graph_mask_ratio"]),
            num_regions=int(mask.get("num_regions", 1)),
        )
    if strategy != "multiblock":
        raise ValueError(
            f"Unknown mask.strategy {strategy!r}; choose one of: "
            "body_region_segment, random_body_segment, random_spatial_segment, multiblock"
        )
    common = dict(
        enc_frame_mask_ratio=tuple(mask["enc_frame_mask_ratio"]),
        pred_frame_mask_ratio=tuple(mask["pred_frame_mask_ratio"]),
        nenc=int(mask["num_enc_masks"]),
        npred=int(mask["num_pred_masks"]),
        allow_overlap=bool(mask["allow_overlap"]),
    )
    if layout.kind == "1d":
        if version == "v2":
            common["min_context_tokens"] = int(mask.get("min_context_tokens", 1))
            common["min_context_ratio"] = float(mask.get("min_context_ratio", 0.2))
            common["context_selection"] = context_selection
            if layout.patchified:
                return PatchMaskCollator1DV2(
                    raw_num_frames=layout.raw_num_frames,
                    temporal_patch_size=layout.temporal_patch_size,
                    **common,
                )
            return MaskCollator1DV2(num_frames=layout.token_num_frames, **common)
        common["context_selection"] = context_selection
        if layout.patchified:
            return PatchMaskCollator1D(
                raw_num_frames=layout.raw_num_frames,
                temporal_patch_size=layout.temporal_patch_size,
                **common,
            )
        return MaskCollator1D(num_frames=layout.token_num_frames, **common)
    if layout.patchified:
        patch = args["patch"]
        return PatchMaskCollator2D(
            raw_num_frames=layout.raw_num_frames,
            raw_num_joints=int(layout.raw_num_joints),
            token_num_joints=int(layout.token_num_joints),
            temporal_patch_size=layout.temporal_patch_size,
            spatial_grouping=str(patch["spatial_grouping"]),
            spatial_pooling=str(patch["spatial_pooling"]),
            enc_joint_mask_ratio=tuple(mask["enc_joint_mask_ratio"]),
            pred_joint_mask_ratio=tuple(mask["pred_joint_mask_ratio"]),
            **common,
        )
    return MaskCollator2D(
        num_frames=layout.token_num_frames,
        num_joints=int(layout.token_num_joints),
        enc_joint_mask_ratio=tuple(mask["enc_joint_mask_ratio"]),
        pred_joint_mask_ratio=tuple(mask["pred_joint_mask_ratio"]),
        **common,
    )


def _prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks_pred: list[torch.Tensor],
    *,
    num_enc_masks: int,
    kind: str,
) -> torch.Tensor:
    """Keep the legacy loss; exclude v2 padding and weight samples equally."""
    if kind != "1d":
        return F.smooth_l1_loss(prediction, target)
    active = index_mask_validity(masks_pred)
    if bool(active.all()):
        return F.smooth_l1_loss(prediction, target)
    active = repeat_mask_blocks(active, masks_pred[0].shape[0], num_enc_masks)
    if prediction.shape != target.shape or active.shape != prediction.shape[:2]:
        raise ValueError("Prediction, target, and target-mask shapes do not match")
    counts = active.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Each target mask must contain at least one valid token")
    token_loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    token_loss = token_loss.masked_fill(~active, 0.0)
    return (token_loss.sum(dim=1) / counts).mean()


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        resolved = torch.device("cuda", local_rank)
    else:
        resolved = torch.device("cpu")
    if resolved.type == "cuda":
        torch.cuda.set_device(resolved)
    return resolved


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def main(args: dict, resume_preempt: bool = False, device=None):
    device = _resolve_device(device)
    probe_configuration = args.get("linear_probe", {})
    attentive_args = _attentive_probe_args(args)
    attentive_enabled = bool(attentive_args.get("enabled", False))
    segmentation_args = args.get("segmentation_probe", {})
    if not isinstance(segmentation_args, dict):
        raise ValueError("segmentation_probe config must be a mapping")
    segmentation_enabled = bool(segmentation_args.get("enabled", False))
    babel_probe_requested = (
        isinstance(probe_configuration, dict)
        and bool(probe_configuration.get("enabled", False))
        and "datasets" in probe_configuration
    )
    distributed = init_distributed(
        device, timeout_seconds=4 * 60 * 60
        if (babel_probe_requested or attentive_enabled or segmentation_enabled) else None
    )
    rank, world_size = distributed.rank, distributed.world_size
    if rank != 0:
        logger.setLevel(logging.ERROR)

    seed = int(args.get("meta", {}).get("seed", 0))
    # Model and EMA-target initialization must be identical before DDP broadcasts.
    _seed_all(seed, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_args = args["data"]
    meta_args = args["meta"]
    opt_args = args["optimization"]
    log_args = args["logging"]
    probe_args = args.get("linear_probe", {})
    if not isinstance(probe_args, dict):
        raise ValueError("linear_probe config must be a mapping")
    probe_enabled = bool(probe_args.get("enabled", False))
    babel_probe_enabled = probe_enabled and "datasets" in probe_args
    metric_args = args.get("online_metrics", {})
    if not isinstance(metric_args, dict):
        raise ValueError("online_metrics config must be a mapping")
    metrics_enabled = bool(metric_args.get("enabled", False))
    probe_frequency = int(
        probe_args.get("frequency", log_args.get("checkpoint_freq", 50))
    )
    if probe_enabled and probe_frequency <= 0:
        raise ValueError("linear_probe.frequency must be positive")
    metrics_frequency = int(metric_args.get("frequency", probe_frequency))
    if metrics_enabled and metrics_frequency <= 0:
        raise ValueError("online_metrics.frequency must be positive")
    metrics_kind = str(metric_args.get("kind", "legacy"))
    if metrics_enabled and metrics_kind not in {"legacy", "motion"}:
        raise ValueError("online_metrics.kind must be legacy or motion")
    segmentation_frequency = int(segmentation_args.get("frequency", 30))
    if segmentation_enabled and segmentation_frequency <= 0:
        raise ValueError("segmentation_probe.frequency must be positive")
    attentive_frequency = int(attentive_args["frequency"])
    if attentive_enabled and attentive_frequency <= 0:
        raise ValueError("attentive_probe.frequency must be positive")
    if attentive_enabled and not isinstance(attentive_args.get("datasets"), dict):
        raise ValueError("attentive_probe requires BABEL datasets (or linear_probe.datasets)")
    model_name = str(meta_args["model_name"])
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown MotionJEPA model_name: {model_name!r}")
    predictor_name = str(meta_args["predictor_name"])
    if predictor_name not in PREDICTOR_FACTORIES:
        raise ValueError(f"Unknown MotionJEPA predictor_name: {predictor_name!r}")

    output = Path(log_args["folder"])
    resume_requested = bool(meta_args.get("load_checkpoint", False) or resume_preempt)

    def initialize_output():
        # Check and create on the same rank: another rank may arrive after the
        # directory has already been created by rank zero. Broadcast failures
        # before anyone enters model/DDP initialization.
        if output.exists() and not resume_requested:
            raise FileExistsError(f"Output folder already exists: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "params-motion-jepa.yaml").write_text(
            yaml.safe_dump(args, sort_keys=False), encoding="utf-8"
        )

    _run_rank_zero(initialize_output, is_main=distributed.is_main)

    encoder, predictor = init_mjepa_model_from_config(args, device)
    layout = encoder.token_layout
    architecture = architecture_signature(
        encoder,
        predictor,
        model_name=model_name,
        predictor_name=predictor_name,
        motion_dim=int(data_args["motion_dim"]),
    )
    mask_collator = _build_mask_collator(args, layout)
    _, loader, sampler = make_motion_dataset(
        root_path=data_args["root_path"],
        meta_files=data_args["meta_files"],
        batch_size=int(data_args["batch_size"]),
        num_frames=int(data_args["num_frames"]),
        fps=int(data_args["fps"]),
        motion_dim=int(data_args["motion_dim"]),
        normalize=bool(data_args.get("normalize", False)),
        stats_path=data_args.get("stats_path"),
        rank=rank,
        world_size=world_size,
        collator=mask_collator,
        drop_last=bool(data_args.get("drop_last", True)),
        num_workers=int(data_args.get("num_workers", 8)),
        pin_mem=bool(data_args.get("pin_mem", True)),
        persistent_workers=bool(data_args.get("persistent_workers", True)),
    )
    if len(loader) == 0:
        raise ValueError("Data loader has no batches; reduce batch_size or disable drop_last")

    target_encoder = copy.deepcopy(encoder).to(device)
    target_encoder.requires_grad_(False)
    target_encoder.eval()

    optimizer, scaler, lr_scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        iterations_per_epoch=len(loader),
        start_lr=float(opt_args["start_lr"]),
        ref_lr=float(opt_args["lr"]),
        final_lr=float(opt_args["final_lr"]),
        warmup=float(opt_args["warmup"]),
        num_epochs=int(opt_args["epochs"]),
        wd=float(opt_args["weight_decay"]),
        final_wd=float(opt_args["final_weight_decay"]),
        use_float16=bool(meta_args.get("use_float16", False)),
        ipe_scale=float(opt_args.get("ipe_scale", 1.0)),
    )
    total_steps = int(
        len(loader) * int(opt_args["epochs"]) * float(opt_args.get("ipe_scale", 1.0))
    )
    momentum_scheduler = LinearMomentumSchedule(
        opt_args["ema"][0], opt_args["ema"][1], total_steps
    )

    start_epoch = 0
    global_step = 0
    linear_probe_state = {
        "latest": None,
        "best_val_top1": float("-inf"),
        "best_epoch": None,
    }
    babel_probe_state = (
        {
            name: {"latest": None, "best_val_map": float("-inf"), "best_epoch": None}
            for name in ("babel-60", "babel-120")
        }
        if babel_probe_enabled else None
    )
    online_metrics_state = {"latest": None}
    segmentation_probe_state = {
        "latest": None, "best_val_map": float("-inf"), "best_epoch": None,
        "protocol_hash": None,
    }
    attentive_probe_state = (
        {name: {"latest": None, "best_val_map": float("-inf"), "best_epoch": None}
         for name in ("babel-60", "babel-120")} if attentive_enabled else None
    )
    latest_path = output / f"{log_args['write_tag']}-latest.pth.tar"
    best_accuracy_path = output / f"{log_args['write_tag']}-best-accuracy.pth.tar"
    should_load = resume_requested
    if should_load:
        read_name = meta_args.get("read_checkpoint")
        load_path = output / read_name if read_name else latest_path
        if not load_path.is_file():
            raise FileNotFoundError(f"Checkpoint requested but not found: {load_path}")
        start_epoch, global_step = _load_checkpoint(
            load_path,
            device=device,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            momentum_scheduler=momentum_scheduler,
            mask_collator=mask_collator,
            rank=rank,
            world_size=world_size,
            architecture=architecture,
            linear_probe_state=linear_probe_state,
            babel_probe_state=babel_probe_state,
            attentive_probe_state=attentive_probe_state,
            probe_protocols=_probe_protocols(args),
            online_metrics_state=online_metrics_state,
            segmentation_probe_state=segmentation_probe_state,
        )
        logger.info("Resumed %s at epoch=%d global_step=%d", load_path, start_epoch, global_step)

    if distributed.distributed:
        ddp_kwargs = {"broadcast_buffers": False}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        encoder = DistributedDataParallel(encoder, **ddp_kwargs)
        predictor = DistributedDataParallel(predictor, **ddp_kwargs)
    if not should_load:
        # Runtime stochasticity may differ by rank after identical model creation.
        _seed_all(seed + rank, device)

    encoder_params = sum(p.numel() for p in _unwrapped(encoder).parameters())
    predictor_params = sum(p.numel() for p in _unwrapped(predictor).parameters())
    logger.info(
        "Initialized %s on %s rank=%d/%d (encoder %.2fM, predictor %.2fM)",
        model_name,
        device,
        rank,
        world_size,
        encoder_params / 1.0e6,
        predictor_params / 1.0e6,
    )

    csv_logger = None
    tensorboard_writer = None
    if distributed.is_main:
        csv_logger = CSVLogger(
            str(output / f"{log_args['write_tag']}.csv"),
            ("%d", "epoch"),
            ("%d", "iteration"),
            ("%d", "global_step"),
            ("%.7f", "loss"),
            ("%.7e", "learning_rate"),
            ("%.7e", "weight_decay"),
            ("%.3f", "time_ms"),
        )
        # A committed checkpoint already includes its evaluation at global_step.
        # Purge only later, uncommitted events; compatible resume skips that
        # completed evaluation and therefore will not write its scalar again.
        tensorboard_writer = _make_tensorboard_writer(
            log_args, output, global_step + 1 if should_load else global_step
        )

    online_attentive_probe = None
    if attentive_enabled and distributed.is_main:
        from experiment.linear_probe.online_attentive import OnlineAttentiveBabelProbes

        online_attentive_probe = OnlineAttentiveBabelProbes(args, attentive_args, device=device)
        logger.info("Enabled online attentive BABEL probes (epochs=%d, lr=%.3g, frequency=%d)",
                    online_attentive_probe.epochs, online_attentive_probe.learning_rate, attentive_frequency)
    online_linear_probe = None
    if probe_enabled and distributed.is_main:
        if babel_probe_enabled:
            from experiment.linear_probe.online import OnlineBabelProbes

            online_linear_probe = OnlineBabelProbes(args, probe_args, device=device)
            logger.info(
                "Enabled online BABEL-60 and BABEL-120 probes "
                "(epochs=%d, lr=%.3g, frequency=%d)",
                online_linear_probe.epochs,
                online_linear_probe.learning_rate,
                probe_frequency,
            )
        else:
            from experiment.linear_probe.online import OnlineLinearProbe

            online_linear_probe = OnlineLinearProbe(args, probe_args, device=device)
            logger.info(
                "Enabled online linear probe on %s (epochs=%d, lr=%.3g, frequency=%d)",
                online_linear_probe.dataset_root,
                online_linear_probe.epochs,
                online_linear_probe.learning_rate,
                probe_frequency,
            )

    online_segmentation_probe = None
    if segmentation_enabled:
        def initialize_segmentation():
            nonlocal online_segmentation_probe
            from experiment.segmentation_probe import OnlineSegmentationProbe

            rng_state = capture_rng_state()
            try:
                online_segmentation_probe = OnlineSegmentationProbe(
                    args, segmentation_args, device=device
                )
            finally:
                restore_rng_state(rng_state)
            return online_segmentation_probe.protocol_hash

        protocol_hash = _run_rank_zero(initialize_segmentation, is_main=distributed.is_main)
        if segmentation_probe_state["protocol_hash"] != protocol_hash:
            segmentation_probe_state.update(latest=None, best_val_map=float("-inf"),
                                            best_epoch=None, protocol_hash=protocol_hash)
        logger.info("Enabled BABEL segmentation linear probe (frequency=%d)", segmentation_frequency)

    online_metrics = None
    if metrics_enabled and metrics_kind == "motion":
        def initialize_motion_metrics():
            nonlocal online_metrics
            from experiment.motion_online_metrics import MotionOnlineMetrics

            rng_state = capture_rng_state()
            try:
                online_metrics = MotionOnlineMetrics(args, metric_args, device=device)
            finally:
                restore_rng_state(rng_state)
            return online_metrics.protocol_hash

        protocol_hash = _run_rank_zero(initialize_motion_metrics, is_main=distributed.is_main)
        saved = online_metrics_state["latest"] or {}
        if saved.get("protocol_hash") != protocol_hash:
            online_metrics_state["latest"] = None
        logger.info("Enabled online motion metrics (frequency=%d)", metrics_frequency)
    elif metrics_enabled and distributed.is_main:
        from experiment.online_metrics import OnlineRepresentationMetrics

        online_metrics = OnlineRepresentationMetrics(
            args,
            metric_args,
            device=device,
            collator=_build_mask_collator(args, layout),
        )
        logger.info(
            "Enabled online representation metrics on %d fixed validation samples "
            "(frequency=%d)",
            len(online_metrics.indices),
            metrics_frequency,
        )

    use_bfloat16 = bool(meta_args.get("use_bfloat16", False))
    use_float16 = bool(meta_args.get("use_float16", False))
    if use_bfloat16 and use_float16:
        raise ValueError("Configure only one of use_bfloat16 and use_float16")
    if use_float16 and device.type != "cuda":
        raise ValueError("Float16 training requires CUDA; use bfloat16 or float32 on CPU")
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16 if use_float16 else None
    epochs = int(opt_args["epochs"])
    checkpoint_frequency = int(log_args.get("checkpoint_freq", 50))
    log_frequency = int(log_args.get("log_freq", 10))
    if checkpoint_frequency <= 0:
        raise ValueError("logging.checkpoint_freq must be positive")
    if log_frequency <= 0:
        raise ValueError("logging.log_freq must be positive")
    warmup = float(opt_args["warmup"])
    clip_grad = float(opt_args["clip_grad"]) if opt_args.get("clip_grad") is not None else None

    def record_babel_probes(pretrain_epoch: int, summaries: dict, state: dict, kind: str) -> set[str]:
        improved = set()
        for name, summary in summaries.items():
            score = float(summary["best_val"]["mean_average_precision"])
            current = state[name]
            current["latest"] = {"pretrain_epoch": pretrain_epoch, "global_step": global_step, **summary}
            if score > float(current["best_val_map"]):
                current.update(best_val_map=score, best_epoch=pretrain_epoch)
                improved.add(name)
            logger.info("epoch=%d %s %s val_mAP=%.4f best_val_mAP=%.4f best_epoch=%s",
                        pretrain_epoch, kind, name, score, current["best_val_map"], current["best_epoch"])
        _write_tensorboard_babel_probes(tensorboard_writer, global_step=global_step,
            summaries=summaries, state=state, probe_name=kind)
        return improved

    def run_online_attentive_probe(pretrain_epoch: int) -> set[str]:
        if not distributed.is_main:
            return set()
        if online_attentive_probe is None or attentive_probe_state is None:
            raise RuntimeError("Online attentive probe was not initialized on rank 0")
        summaries = _evaluate_online_probe_preserving_rng(online_attentive_probe, target_encoder)
        return record_babel_probes(pretrain_epoch, summaries, attentive_probe_state, "attentive_probe")

    def run_online_linear_probe(pretrain_epoch: int) -> tuple[bool, set[str]]:
        if not distributed.is_main:
            return False, set()
        if online_linear_probe is None:
            raise RuntimeError("Online linear probe was not initialized on rank 0")
        probe_summary = _evaluate_online_probe_preserving_rng(
            online_linear_probe, target_encoder
        )
        if babel_probe_enabled:
            assert babel_probe_state is not None
            return False, record_babel_probes(pretrain_epoch, probe_summary, babel_probe_state, "linear_probe")
        test_top1 = float(probe_summary["test"]["top1_accuracy"])
        linear_probe_state["latest"] = {
            "pretrain_epoch": pretrain_epoch,
            "global_step": global_step,
            **probe_summary,
        }
        validation_used = bool(probe_summary.get("validation_used", True))
        val_top1 = (
            float(probe_summary["best_val"]["top1_accuracy"])
            if validation_used
            else None
        )
        improved = validation_used and val_top1 > float(linear_probe_state["best_val_top1"])
        if improved:
            linear_probe_state["best_val_top1"] = val_top1
            linear_probe_state["best_epoch"] = pretrain_epoch
        _write_tensorboard_linear_probe(
            tensorboard_writer,
            global_step=global_step,
            summary=probe_summary,
            best_val_top1=(
                float(linear_probe_state["best_val_top1"])
                if validation_used
                else None
            ),
        )
        if validation_used:
            logger.info(
                "epoch=%d linear_probe val_top1=%.4f test_top1=%.4f "
                "best_val_top1=%.4f best_epoch=%s",
                pretrain_epoch,
                val_top1,
                test_top1,
                float(linear_probe_state["best_val_top1"]),
                linear_probe_state["best_epoch"],
            )
        else:
            logger.info(
                "epoch=%d linear_probe test_top1=%.4f selection=fixed_last_epoch "
                "(diagnostic only)",
                pretrain_epoch,
                test_top1,
            )
        return improved, set()

    def run_online_segmentation_probe(pretrain_epoch: int) -> bool:
        summary = _run_rank_zero(
            lambda: _evaluate_frozen_encoder_preserving_rng(
                online_segmentation_probe, target_encoder, pretrain_epoch=pretrain_epoch
            ), is_main=distributed.is_main,
        )
        summary = {**summary, "pretrain_epoch": int(pretrain_epoch),
                   "global_step": int(global_step)}
        score = float(summary["best_val"]["babel-120"]["frame_map"])
        if not np.isfinite(score):
            raise FloatingPointError("Segmentation probe returned non-finite validation mAP")
        improved = score > segmentation_probe_state["best_val_map"]
        segmentation_probe_state["latest"] = summary
        if improved:
            segmentation_probe_state.update(best_val_map=score, best_epoch=pretrain_epoch)
        if distributed.is_main:
            _write_tensorboard_segmentation_probe(
                tensorboard_writer, global_step=global_step, summary=summary,
                state=segmentation_probe_state,
            )
            _upsert_epoch_jsonl(output / "segmentation-probe.jsonl", summary)
            logger.info("epoch=%d segmentation frame_mAP120=%.5f frame_mAP60=%s best_epoch=%s",
                        pretrain_epoch, score, summary["best_val"]["babel-60"]["frame_map"],
                        segmentation_probe_state["best_epoch"])
        return improved

    def run_online_representation_metrics(pretrain_epoch: int) -> None:
        if metrics_kind == "motion":
            summary = _run_rank_zero(
                lambda: _evaluate_frozen_encoder_preserving_rng(online_metrics, target_encoder),
                is_main=distributed.is_main,
            )
        else:
            if not distributed.is_main:
                return
            if online_metrics is None:
                raise RuntimeError("Online representation metrics were not initialized")
            summary = _evaluate_online_metrics_preserving_rng(
                online_metrics, target_encoder, _unwrapped(predictor)
            )
        summary = {
            "pretrain_epoch": int(pretrain_epoch),
            "global_step": int(global_step),
            **summary,
        }
        if metrics_kind == "motion" and distributed.is_main:
            previous = online_metrics_state["latest"] or {}
            previous_epoch = int(previous.get("pretrain_epoch", 0))
            epoch_path = output / "training-epochs.jsonl"
            epoch_records = ([json.loads(line) for line in epoch_path.read_text().splitlines()]
                             if epoch_path.exists() else [])
            training_seconds = sum(record["training_wall_seconds"] for record in epoch_records
                                   if previous_epoch < record["pretrain_epoch"] <= pretrain_epoch)
            summary["training_interval_seconds"] = training_seconds
            summary["overhead_percent"] = (
                100 * float(summary["elapsed_seconds"]) / training_seconds
                if training_seconds > 0 else None
            )
        online_metrics_state["latest"] = summary
        if not distributed.is_main:
            return
        _write_tensorboard_online_metrics(
            tensorboard_writer, global_step=global_step, summary=summary
        )
        _upsert_epoch_jsonl(output / "online-metrics.jsonl", summary)
        if metrics_kind == "motion":
            logger.info("epoch=%d online_motion %s", pretrain_epoch,
                        json.dumps(summary["retrieval"], sort_keys=True))
            return
        heldout = summary["heldout_jepa"]
        logger.info(
            "epoch=%d online_metrics rankme=%.3f body_std=%.4g "
            "prediction_gain=%.4f trajectory_reliance=%.4f",
            pretrain_epoch,
            summary["representation"]["body"]["rankme"],
            summary["representation"]["body"]["mean_std"],
            heldout["prediction_gain"],
            heldout["trajectory_ablation"]["trajectory_reliance"],
        )

    # Establish a baseline for new or changed probe protocols. Compatible
    # resumed results are retained without repeating the expensive evaluation.
    needs_initial_probe = (
        probe_enabled and (
            any(state["latest"] is None for state in babel_probe_state.values())
            if babel_probe_state is not None
            else linear_probe_state["latest"] is None
        )
    )
    needs_initial_metrics = (
        metrics_enabled and online_metrics_state["latest"] is None
    )
    needs_initial_attentive = attentive_enabled and any(
        state["latest"] is None for state in attentive_probe_state.values()
    )
    needs_initial_segmentation = segmentation_enabled and segmentation_probe_state["latest"] is None
    if needs_initial_probe or needs_initial_metrics or needs_initial_attentive or needs_initial_segmentation:
        if not should_load:
            # A baseline head can take many epochs. Persist the untouched
            # pretraining state first so a failure can resume that head instead
            # of leaving an existing output directory with no latest checkpoint.
            # Every rank participates in the RNG/mask-state collectives.
            _save_checkpoint(
                latest_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                optimizer=optimizer,
                scaler=scaler,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
                momentum_scheduler=momentum_scheduler,
                mask_collator=mask_collator,
                next_epoch=start_epoch,
                global_step=global_step,
                loss=float("nan"),
                world_size=world_size,
                rank=rank,
                config=args,
                architecture=architecture,
                linear_probe_latest=linear_probe_state["latest"],
                best_probe_val_top1=float(linear_probe_state["best_val_top1"]),
                best_probe_epoch=linear_probe_state["best_epoch"],
                babel_probe_state=babel_probe_state,
                attentive_probe_state=attentive_probe_state,
                online_metrics_latest=online_metrics_state["latest"],
                segmentation_probe_state=segmentation_probe_state,
            )
        initial_probe_improved, initial_babel_improved = (
            run_online_linear_probe(pretrain_epoch=start_epoch)
            if needs_initial_probe else (False, set())
        )
        initial_attentive_improved = (
            run_online_attentive_probe(pretrain_epoch=start_epoch) if needs_initial_attentive else set()
        )
        initial_segmentation_improved = (
            run_online_segmentation_probe(pretrain_epoch=start_epoch)
            if needs_initial_segmentation else False
        )
        if needs_initial_metrics:
            run_online_representation_metrics(pretrain_epoch=start_epoch)
        _save_checkpoint(
            latest_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            momentum_scheduler=momentum_scheduler,
            mask_collator=mask_collator,
            next_epoch=start_epoch,
            global_step=global_step,
            loss=float("nan"),
            world_size=world_size,
            rank=rank,
            config=args,
            architecture=architecture,
            linear_probe_latest=linear_probe_state["latest"],
            best_probe_val_top1=float(linear_probe_state["best_val_top1"]),
            best_probe_epoch=linear_probe_state["best_epoch"],
            babel_probe_state=babel_probe_state,
            attentive_probe_state=attentive_probe_state,
            online_metrics_latest=online_metrics_state["latest"],
            segmentation_probe_state=segmentation_probe_state,
        )
        if distributed.is_main and initial_probe_improved:
            _atomic_copy(latest_path, best_accuracy_path)
        if distributed.is_main:
            for name in initial_babel_improved:
                _atomic_copy(
                    latest_path,
                    output / f"{log_args['write_tag']}-best-{name}-map.pth.tar",
                )
        if distributed.is_main:
            for name in initial_attentive_improved:
                _atomic_copy(latest_path, output / f"{log_args['write_tag']}-best-attentive-{name}-map.pth.tar")
            if initial_segmentation_improved:
                _atomic_copy(latest_path, output / f"{log_args['write_tag']}-best-segmentation-map.pth.tar")
            if start_epoch == 0:
                _atomic_copy(latest_path, output / "initial-checkpoint.pth.tar")
        barrier()

    # These meters span epoch boundaries and reset only after a log event, so
    # every TensorBoard point summarizes exactly the steps since the prior one.
    interval_loss_meter = AverageMeter()
    interval_time_meter = AverageMeter()
    interval_lr_meter = AverageMeter()
    interval_wd_meter = AverageMeter()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        encoder.train()
        predictor.train()
        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        for iteration, (batch, masks_enc, masks_pred) in enumerate(loader):
            started = time.perf_counter()
            motion = batch[0].to(device=device, dtype=torch.float32, non_blocking=True)
            fps = batch[1].to(device=device, dtype=torch.float32, non_blocking=True)
            valid_length = batch[2].to(device=device, non_blocking=True)
            valid_frames = (
                torch.arange(motion.shape[1], device=device).unsqueeze(0)
                < valid_length.unsqueeze(1)
            )
            masks_enc = [mask.to(device, non_blocking=True) for mask in masks_enc]
            masks_pred = [mask.to(device, non_blocking=True) for mask in masks_pred]
            learning_rate = lr_scheduler.step()
            weight_decay = wd_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with amp_context:
                with torch.no_grad():
                    target = target_encoder(motion, fps, valid_frames=valid_frames)
                    target = F.layer_norm(target, (target.shape[-1],))
                    if layout.kind == "1d":
                        target = apply_index_masks(target, masks_pred)
                    else:
                        target = gather_grid_masks(target, masks_pred)
                    target = repeat_mask_blocks(target, len(motion), len(masks_enc))
                context = encoder(motion, fps, masks_enc, valid_frames=valid_frames)
                prediction = predictor(context, fps, masks_enc, masks_pred)
                loss = _prediction_loss(
                    prediction, target, masks_pred,
                    num_enc_masks=len(masks_enc), kind=layout.kind,
                )

            # full precision
            if scaler is None:
                loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                optimizer.step()

            # mixed precision
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if (epoch > warmup) and (clip_grad is not None):
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()

            momentum = momentum_scheduler.step()
            update_ema(encoder, target_encoder, momentum)
            global_step += 1
            reported_loss = float(reduce_mean(loss).cpu())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            loss_meter.update(reported_loss)
            time_meter.update(elapsed_ms)
            interval_loss_meter.update(reported_loss)
            interval_time_meter.update(elapsed_ms)
            interval_lr_meter.update(learning_rate)
            interval_wd_meter.update(weight_decay)
            if distributed.is_main:
                csv_logger.log(
                    epoch + 1,
                    iteration,
                    global_step,
                    reported_loss,
                    learning_rate,
                    weight_decay,
                    elapsed_ms,
                )
                if global_step % log_frequency == 0:
                    stats = grad_logger(_unwrapped(encoder).named_parameters())
                    memory = (
                        torch.cuda.max_memory_allocated(device) / 1024.0**2
                        if device.type == "cuda"
                        else 0.0
                    )
                    logger.info(
                        "epoch=%d iteration=%d loss=%.5f lr=%.3e wd=%.3e "
                        "time=%.1fms memory=%.0fMiB grad=[%.2e, %.2e]",
                        epoch + 1,
                        iteration,
                        interval_loss_meter.avg,
                        interval_lr_meter.avg,
                        interval_wd_meter.avg,
                        interval_time_meter.avg,
                        memory,
                        stats.first_layer,
                        stats.last_layer,
                    )
                    _write_tensorboard_interval(
                        tensorboard_writer,
                        global_step=global_step,
                        epoch=epoch + 1,
                        loss=interval_loss_meter.avg,
                        learning_rate=interval_lr_meter.avg,
                        weight_decay=interval_wd_meter.avg,
                        time_ms=interval_time_meter.avg,
                        memory_mib=memory,
                        grad_first=stats.first_layer,
                        grad_last=stats.last_layer,
                        grad_average=stats.avg,
                    )
                    interval_loss_meter.reset()
                    interval_time_meter.reset()
                    interval_lr_meter.reset()
                    interval_wd_meter.reset()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
            if not np.isfinite(reported_loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {reported_loss}")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if distributed.is_main:
            _upsert_epoch_jsonl(output / "training-epochs.jsonl", {
                "pretrain_epoch": epoch + 1, "global_step": global_step,
                "training_wall_seconds": time.perf_counter() - epoch_started,
                "step_compute_seconds": time_meter.sum / 1000,
                "loss": loss_meter.avg,
            })
        should_probe = probe_enabled and (
            (epoch + 1) % probe_frequency == 0 or (epoch + 1) == epochs
        )
        probe_improved = False
        babel_improved: set[str] = set()
        if should_probe and distributed.is_main:
            probe_improved, babel_improved = run_online_linear_probe(pretrain_epoch=epoch + 1)
        attentive_improved: set[str] = set()
        if attentive_enabled and distributed.is_main and (
            (epoch + 1) % attentive_frequency == 0 or (epoch + 1) == epochs
        ):
            attentive_improved = run_online_attentive_probe(pretrain_epoch=epoch + 1)
        segmentation_improved = False
        if segmentation_enabled and (
            (epoch + 1) % segmentation_frequency == 0 or (epoch + 1) == epochs
        ):
            segmentation_improved = run_online_segmentation_probe(pretrain_epoch=epoch + 1)
        should_measure = metrics_enabled and (
            (epoch + 1) % metrics_frequency == 0 or (epoch + 1) == epochs
        )
        if should_measure:
            run_online_representation_metrics(pretrain_epoch=epoch + 1)

        _save_checkpoint(
            latest_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            momentum_scheduler=momentum_scheduler,
            mask_collator=mask_collator,
            next_epoch=epoch + 1,
            global_step=global_step,
            loss=loss_meter.avg,
            world_size=world_size,
            rank=rank,
            config=args,
            architecture=architecture,
            linear_probe_latest=linear_probe_state["latest"],
            best_probe_val_top1=float(linear_probe_state["best_val_top1"]),
            best_probe_epoch=linear_probe_state["best_epoch"],
            babel_probe_state=babel_probe_state,
            attentive_probe_state=attentive_probe_state,
            online_metrics_latest=online_metrics_state["latest"],
            segmentation_probe_state=segmentation_probe_state,
        )
        if distributed.is_main and ((epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == epochs):
            checkpoint_path = output / f"{log_args['write_tag']}-ep{epoch + 1}.pth.tar"
            # The latest checkpoint is already complete and atomically written.
            # Copy its bytes instead of loading arbitrary pickle content merely
            # to create the named epoch snapshot.
            _atomic_copy(latest_path, checkpoint_path)
        if distributed.is_main and probe_improved:
            _atomic_copy(latest_path, best_accuracy_path)
        if distributed.is_main:
            for name in babel_improved:
                _atomic_copy(
                    latest_path,
                    output / f"{log_args['write_tag']}-best-{name}-map.pth.tar",
                )
        if distributed.is_main:
            for name in attentive_improved:
                _atomic_copy(latest_path, output / f"{log_args['write_tag']}-best-attentive-{name}-map.pth.tar")
            if segmentation_improved:
                _atomic_copy(latest_path, output / f"{log_args['write_tag']}-best-segmentation-map.pth.tar")
        barrier()
        logger.info("epoch=%d average_loss=%.6f", epoch + 1, loss_meter.avg)

    if tensorboard_writer is not None:
        tensorboard_writer.close()
    return {
        "next_epoch": epochs,
        "global_step": global_step,
        "checkpoint": str(latest_path),
    }


__all__ = [
    "capture_rng_state",
    "main",
    "restore_rng_state",
    "update_ema",
]
