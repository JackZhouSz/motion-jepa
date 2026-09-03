"""Motion-JEPA pretraining loop shared by the 1D and 2D variants."""

from __future__ import annotations

import copy
import logging
import os
import random
import shutil
import time
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
)
from mask import (
    MaskCollator1D,
    MaskCollator2D,
    PatchBodyRegionSegmentMaskCollator2D,
    PatchMaskCollator1D,
    PatchMaskCollator2D,
    PatchRandomBodySegmentMaskCollator2D,
)
from model import MODEL_FACTORIES, PREDICTOR_FACTORIES, TokenLayout
from mask.utils import (
    apply_index_masks,
    gather_grid_masks,
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
        f"linear_probe/test_{name}": float(value)
        for name, value in summary["test"].items()
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


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy an already-complete checkpoint without deserializing it."""
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


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
) -> tuple[int, int]:
    # Full training checkpoints contain trusted local Python/NumPy RNG state,
    # optimizer state, and scheduler state in addition to tensor weights.
    # PyTorch 2.6 defaults weights_only=True, which cannot restore that payload.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported Motion-JEPA checkpoint format: {path}")
    if int(checkpoint["world_size"]) != world_size:
        raise ValueError(
            f"Exact resume requires world_size={checkpoint['world_size']}, got {world_size}"
        )
    if architecture is not None:
        saved_architecture = checkpoint.get("architecture")
        if saved_architecture is None:
            saved_architecture = architecture_signature_from_config(checkpoint["config"])
        if saved_architecture != architecture:
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
    return int(checkpoint["next_epoch"]), int(checkpoint["global_step"])


def _build_mask_collator(args: dict, layout: TokenLayout):
    mask = args["mask"]
    strategy = str(mask.get("strategy", "multiblock"))
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
            "body_region_segment, random_body_segment, multiblock"
        )
    common = dict(
        enc_frame_mask_ratio=tuple(mask["enc_frame_mask_ratio"]),
        pred_frame_mask_ratio=tuple(mask["pred_frame_mask_ratio"]),
        nenc=int(mask["num_enc_masks"]),
        npred=int(mask["num_pred_masks"]),
        allow_overlap=bool(mask["allow_overlap"]),
    )
    if layout.kind == "1d":
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
    distributed = init_distributed(device)
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
    probe_frequency = int(
        probe_args.get("frequency", log_args.get("checkpoint_freq", 50))
    )
    if probe_enabled and probe_frequency <= 0:
        raise ValueError("linear_probe.frequency must be positive")
    model_name = str(meta_args["model_name"])
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown Motion-JEPA model_name: {model_name!r}")
    predictor_name = str(meta_args["predictor_name"])
    if predictor_name not in PREDICTOR_FACTORIES:
        raise ValueError(f"Unknown Motion-JEPA predictor_name: {predictor_name!r}")

    output = Path(log_args["folder"])
    resume_requested = bool(meta_args.get("load_checkpoint", False) or resume_preempt)
    if output.exists() and not resume_requested:
        raise FileExistsError(f"Output folder already exists: {output}")
    if distributed.is_main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "params-motion-jepa.yaml").write_text(
            yaml.safe_dump(args, sort_keys=False), encoding="utf-8"
        )
    barrier()

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
        tensorboard_writer = _make_tensorboard_writer(log_args, output, global_step)

    online_linear_probe = None
    if probe_enabled and distributed.is_main:
        from experiment.linear_probe.online import OnlineLinearProbe

        online_linear_probe = OnlineLinearProbe(args, probe_args, device=device)
        logger.info(
            "Enabled online linear probe on %s (epochs=%d, lr=%.3g, frequency=%d)",
            online_linear_probe.dataset_root,
            online_linear_probe.epochs,
            online_linear_probe.learning_rate,
            probe_frequency,
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

    def run_online_linear_probe(pretrain_epoch: int) -> bool:
        if not distributed.is_main:
            return False
        if online_linear_probe is None:
            raise RuntimeError("Online linear probe was not initialized on rank 0")
        probe_summary = _evaluate_online_probe_preserving_rng(
            online_linear_probe, target_encoder
        )
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
        return improved

    # Establish a frozen-encoder baseline before the first optimization step.
    # A resumed epoch-0 checkpoint already contains this result, so do not
    # repeat the relatively expensive probe in that case.
    if (
        probe_enabled
        and start_epoch == 0
        and linear_probe_state["latest"] is None
    ):
        initial_probe_improved = run_online_linear_probe(pretrain_epoch=0)
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
            next_epoch=0,
            global_step=global_step,
            loss=float("nan"),
            world_size=world_size,
            rank=rank,
            config=args,
            architecture=architecture,
            linear_probe_latest=linear_probe_state["latest"],
            best_probe_val_top1=float(linear_probe_state["best_val_top1"]),
            best_probe_epoch=linear_probe_state["best_epoch"],
        )
        if distributed.is_main and initial_probe_improved:
            _atomic_copy(latest_path, best_accuracy_path)
        barrier()

    # These meters span epoch boundaries and reset only after a log event, so
    # every TensorBoard point summarizes exactly the steps since the prior one.
    interval_loss_meter = AverageMeter()
    interval_time_meter = AverageMeter()
    interval_lr_meter = AverageMeter()
    interval_wd_meter = AverageMeter()
    for epoch in range(start_epoch, epochs):
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
                loss = F.smooth_l1_loss(prediction, target)

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

        should_probe = probe_enabled and (
            (epoch + 1) % probe_frequency == 0 or (epoch + 1) == epochs
        )
        probe_improved = False
        if should_probe and distributed.is_main:
            probe_improved = run_online_linear_probe(pretrain_epoch=epoch + 1)

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
        )
        if distributed.is_main and (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = output / f"{log_args['write_tag']}-ep{epoch + 1}.pth.tar"
            # The latest checkpoint is already complete and atomically written.
            # Copy its bytes instead of loading arbitrary pickle content merely
            # to create the named epoch snapshot.
            _atomic_copy(latest_path, checkpoint_path)
        if distributed.is_main and probe_improved:
            _atomic_copy(latest_path, best_accuracy_path)
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
