"""Model and optimizer construction for MotionJEPA training."""

from __future__ import annotations

import logging

import torch

from model import (
    MODEL_FACTORIES,
    MODEL_KINDS,
    PATCH_MODEL_NAMES,
    PATCH_PREDICTOR_NAMES,
    PREDICTOR_FACTORIES,
    PREDICTOR_KINDS,
    get_spatial_grouping,
    spatial_patch_signature,
)
from model.motion_patch_transformer_2d import (
    BODY_TOKEN_OFFSET,
    TRAJECTORY_FIELDS,
    TRAJECTORY_TOKEN_INDEX,
)
from model.pos_embs import normalize_position_encoding
from utils.schedulers import CosineWDSchedule, WarmupCosineSchedule


logger = logging.getLogger(__name__)


def position_encoding_from_config(config: dict) -> dict:
    """Resolve position settings, retaining absolute positions for legacy configs."""
    settings = config.get("position_encoding", {})
    if not isinstance(settings, dict):
        raise ValueError("position_encoding must be a mapping")
    unknown = set(settings) - {"temporal", "rope_theta", "rope_time_scale"}
    if unknown:
        raise ValueError(f"Unknown position_encoding settings: {sorted(unknown)}")
    return normalize_position_encoding(
        settings.get("temporal", "absolute"),
        settings.get("rope_theta", 100.0),
        settings.get("rope_time_scale", 1.0),
    )


def position_encoding_from_model(model) -> dict:
    return normalize_position_encoding(
        getattr(model, "position_encoding", "absolute"),
        getattr(model, "rope_theta", 100.0),
        getattr(model, "rope_time_scale", 1.0),
    )


def normalize_architecture_signature(signature: dict) -> dict:
    """Interpret signatures saved before configurable positions as absolute."""
    return {**signature, "position_encoding": position_encoding_from_config(signature)}


def _position_kwargs(settings: dict, kind: str) -> dict:
    if kind == "2d":
        if settings["temporal"] == "rope":
            raise ValueError("RoPE currently supports only 1D encoders and predictors")
        return {}
    return {
        "position_encoding": settings["temporal"],
        **{key: value for key, value in settings.items() if key != "temporal"},
    }


def init_mjepa_model(
    device: torch.device,
    num_frames: int,
    motion_dim: int,
    num_joints: int,
    model_name: str,
    predictor_name: str,
    temporal_patch_size: int = 1,
    spatial_grouping: str = "fine11",
    spatial_pooling: str = "graph_mean",
    position_encoding: str = "absolute",
    rope_theta: float = 100.0,
    rope_time_scale: float = 1.0,
):
    positions = normalize_position_encoding(position_encoding, rope_theta, rope_time_scale)
    try:
        factory = MODEL_FACTORIES[model_name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(f"Unknown model_name {model_name!r}; choose one of: {choices}") from error
    is_patch_model = model_name in PATCH_MODEL_NAMES
    encoder_kwargs = {"in_chans": motion_dim, "num_frames": num_frames}
    encoder_kwargs.update(_position_kwargs(positions, MODEL_KINDS[model_name]))
    if is_patch_model:
        encoder_kwargs["temporal_patch_size"] = int(temporal_patch_size)
    if MODEL_KINDS[model_name] == "2d":
        encoder_kwargs["num_joints"] = num_joints
        if is_patch_model:
            encoder_kwargs.update(
                spatial_grouping=str(spatial_grouping),
                spatial_pooling=str(spatial_pooling),
            )
    encoder = factory(**encoder_kwargs)

    try:
        pred_factory = PREDICTOR_FACTORIES[predictor_name]
    except KeyError as error:
        choices = ", ".join(sorted(PREDICTOR_FACTORIES))
        raise ValueError(f"Unknown predictor_name {predictor_name!r}; choose one of: {choices}") from error
    is_patch_predictor = predictor_name in PATCH_PREDICTOR_NAMES
    predictor_kwargs = {"num_frames": num_frames, "embed_dim": encoder.embed_dim}
    predictor_kwargs.update(_position_kwargs(positions, PREDICTOR_KINDS[predictor_name]))
    if is_patch_predictor:
        predictor_kwargs["temporal_patch_size"] = int(temporal_patch_size)
    if PREDICTOR_KINDS[predictor_name] == "2d":
        predictor_kwargs["num_joints"] = num_joints
        if is_patch_predictor:
            predictor_kwargs.update(
                spatial_grouping=str(spatial_grouping),
                spatial_pooling=str(spatial_pooling),
            )
    predictor = pred_factory(**predictor_kwargs)
    if encoder.token_layout != predictor.token_layout:
        raise ValueError(
            "Encoder and predictor token layouts differ: "
            f"encoder={encoder.token_layout}, predictor={predictor.token_layout}"
        )
    return encoder.to(device), predictor.to(device)


def patch_size_from_config(config: dict) -> int:
    patch = config.get("patch")
    return int(patch.get("temporal_patch_size", 1)) if isinstance(patch, dict) else 1


def spatial_patch_from_config(config: dict) -> tuple[str, str]:
    patch = config.get("patch")
    if not isinstance(patch, dict):
        return "fine11", "graph_mean"
    return (
        str(patch.get("spatial_grouping", "fine11")),
        str(patch.get("spatial_pooling", "graph_mean")),
    )


def init_mjepa_model_from_config(config: dict, device: torch.device):
    data = config["data"]
    meta = config["meta"]
    grouping, pooling = spatial_patch_from_config(config)
    positions = position_encoding_from_config(config)
    encoder, predictor = init_mjepa_model(
        device=device,
        num_frames=int(data["num_frames"]),
        motion_dim=int(data["motion_dim"]),
        num_joints=int(data["num_joints"]),
        model_name=str(meta["model_name"]),
        predictor_name=str(meta["predictor_name"]),
        temporal_patch_size=patch_size_from_config(config),
        spatial_grouping=grouping,
        spatial_pooling=pooling,
        position_encoding=positions["temporal"],
        rope_theta=positions.get("rope_theta", 100.0),
        rope_time_scale=positions.get("rope_time_scale", 1.0),
    )
    if hasattr(predictor, "_packed_spatial_disjoint"):
        mask = config.get("mask")
        predictor._packed_spatial_disjoint = (
            isinstance(mask, dict)
            and not bool(
                mask.get(
                    "allow_context_target_overlap",
                    mask.get("allow_overlap", False),
                )
            )
        )
    return encoder, predictor


def init_mjepa_encoder_from_config(config: dict, device: torch.device):
    data = config["data"]
    meta = config["meta"]
    model_name = str(meta["model_name"])
    try:
        factory = MODEL_FACTORIES[model_name]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_FACTORIES))
        raise ValueError(
            f"Unknown model_name {model_name!r}; choose one of: {choices}"
        ) from error
    kwargs = {
        "in_chans": int(data["motion_dim"]),
        "num_frames": int(data["num_frames"]),
    }
    kwargs.update(_position_kwargs(position_encoding_from_config(config), MODEL_KINDS[model_name]))
    if model_name in PATCH_MODEL_NAMES:
        kwargs["temporal_patch_size"] = patch_size_from_config(config)
    if MODEL_KINDS[model_name] == "2d":
        kwargs["num_joints"] = int(data["num_joints"])
        if model_name in PATCH_MODEL_NAMES:
            grouping, pooling = spatial_patch_from_config(config)
            kwargs.update(spatial_grouping=grouping, spatial_pooling=pooling)
    return factory(**kwargs).to(device)


def architecture_signature(
    encoder,
    predictor,
    *,
    model_name: str,
    predictor_name: str,
    motion_dim: int,
) -> dict:
    positions = position_encoding_from_model(encoder)
    if positions != position_encoding_from_model(predictor):
        raise ValueError("Encoder and predictor positional configurations differ")
    signature = {
        "model_name": str(model_name),
        "predictor_name": str(predictor_name),
        "motion_dim": int(motion_dim),
        "encoder_layout": encoder.token_layout.signature(),
        "predictor_layout": predictor.token_layout.signature(),
        "position_encoding": positions,
    }
    encoder_spatial = getattr(encoder, "spatial_patch_signature", None)
    predictor_spatial = getattr(predictor, "spatial_patch_signature", None)
    if encoder_spatial is not None or predictor_spatial is not None:
        if encoder_spatial is None or predictor_spatial is None:
            raise ValueError("Encoder and predictor spatial patch configurations differ")
        encoder_signature = encoder_spatial()
        predictor_signature = predictor_spatial()
        if encoder_signature != predictor_signature:
            raise ValueError(
                "Encoder and predictor spatial patch configurations differ: "
                f"encoder={encoder_signature}, predictor={predictor_signature}"
            )
        signature["spatial_patch"] = encoder_signature
    return signature


def architecture_signature_from_config(config: dict) -> dict:
    data = config["data"]
    meta = config["meta"]
    model_name = str(meta["model_name"])
    predictor_name = str(meta["predictor_name"])
    patchified = model_name in PATCH_MODEL_NAMES
    predictor_patchified = predictor_name in PATCH_PREDICTOR_NAMES
    patch_size = patch_size_from_config(config) if patchified else 1
    predictor_patch_size = patch_size_from_config(config) if predictor_patchified else 1
    raw_frames = int(data["num_frames"])
    kind = MODEL_KINDS[model_name]
    predictor_kind = PREDICTOR_KINDS[predictor_name]
    positions = position_encoding_from_config(config)
    _position_kwargs(positions, kind)
    _position_kwargs(positions, predictor_kind)
    grouping, pooling = spatial_patch_from_config(config)
    has_spatial_patch = (
        (kind == "2d" and patchified)
        or (predictor_kind == "2d" and predictor_patchified)
    )
    if has_spatial_patch and pooling != "graph_mean":
        raise ValueError("Patchified 2D models support only graph_mean pooling")
    body_groups = get_spatial_grouping(grouping) if has_spatial_patch else ()
    token_joints = 1 + len(body_groups) if has_spatial_patch else None

    def layout_signature(layout_kind: str, is_patch: bool, size: int) -> dict:
        joints = int(data["num_joints"]) if layout_kind == "2d" else None
        pooled_joints = (
            int(token_joints) if layout_kind == "2d" and is_patch else joints
        )
        result = {
            "kind": layout_kind,
            "patchified": is_patch,
            "raw_num_frames": raw_frames,
            "token_num_frames": raw_frames // size,
            "temporal_patch_size": size,
            "raw_num_joints": joints,
            "token_num_joints": pooled_joints,
        }
        if layout_kind == "2d" and is_patch:
            result.update(
                trajectory_token_index=TRAJECTORY_TOKEN_INDEX,
                body_token_offset=BODY_TOKEN_OFFSET,
                spatial_token_names=[
                    "trajectory", *[name for name, _ in body_groups]
                ],
                trajectory_fields=list(TRAJECTORY_FIELDS),
            )
        else:
            result.update(
                trajectory_token_index=None,
                body_token_offset=None,
                spatial_token_names=None,
                trajectory_fields=None,
            )
        return result

    signature = {
        "model_name": model_name,
        "predictor_name": predictor_name,
        "motion_dim": int(data["motion_dim"]),
        "position_encoding": positions,
        "encoder_layout": layout_signature(kind, patchified, patch_size),
        "predictor_layout": layout_signature(
            predictor_kind, predictor_patchified, predictor_patch_size
        ),
    }
    if kind == "2d" and patchified:
        signature["spatial_patch"] = spatial_patch_signature(grouping)
    return signature


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1.0e-6,
    final_wd=1.0e-6,
    final_lr=0.0,
    use_float16=False,
    ipe_scale=1.0,
):
    decay, no_decay = [], []
    for module in (encoder, predictor):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if name.endswith("bias") or parameter.ndim == 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0, "WD_exclude": True},
        ]
    )
    total_steps = int(ipe_scale * num_epochs * iterations_per_epoch)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_float16 else None
    logger.info("Using AdamW with %d decay and %d no-decay tensors", len(decay), len(no_decay))
    return optimizer, scaler, scheduler, wd_scheduler


__all__ = [
    "architecture_signature",
    "architecture_signature_from_config",
    "init_mjepa_encoder_from_config",
    "init_mjepa_model",
    "init_mjepa_model_from_config",
    "init_opt",
    "normalize_architecture_signature",
    "patch_size_from_config",
    "position_encoding_from_config",
    "position_encoding_from_model",
    "spatial_patch_from_config",
]
