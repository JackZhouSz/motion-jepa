"""Train a conditional raw-motion flow, keeping JEPA features frozen."""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from experiment.linear_probe.features import (
    _atomic_json_save, _atomic_torch_save, _seed_all, _torch_load_checkpoint, resolve_device,
)
from experiment.prediction.data import FEATURE_TRANSFORM, PredictionDataset, prepare_caches
from experiment.prediction.metrics import masked_mse
from experiment.prediction.train import (
    autocast, capture_rng, learning_rate, make_loader, move_batch, restore_rng, save_history,
)
from .model import FlowConfig, MotionFlow
from .sampling import seeded_noise, seeded_times

FORMAT_VERSION = 1
RESUME_FIELDS = (
    "seed", "epochs", "batch_size", "lr", "final_lr", "weight_decay", "warmup_epochs",
    "gradient_clip", "use_bfloat16", "flow", "condition_dropout", "ema_decay",
    "evaluation_seed",
)


def make_generator(dataset: PredictionDataset, config: dict) -> MotionFlow:
    return MotionFlow(
        feature_dim=int(dataset.model_info["feature_dim"]), token_layout=dataset.token_layout,
        motion_dim=int(dataset.model_info["motion_dim"]), config=FlowConfig(**config["flow"]),
    )


def interpolate_motion(motion: torch.Tensor, noise: torch.Tensor, times: torch.Tensor,
                       valid_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the noise-to-data path and target velocity in FP32."""
    if motion.ndim != 3 or motion.shape != noise.shape:
        raise ValueError("Motion and noise must have matching shape [B,T,C]")
    if times.shape != (len(motion),) or valid_frames.shape != motion.shape[:2]:
        raise ValueError("Flow times or valid frame mask has the wrong shape")
    motion = motion.float().masked_fill(~valid_frames[..., None], 0)
    noise = noise.float().masked_fill(~valid_frames[..., None], 0)
    if not torch.isfinite(motion).all() or not torch.isfinite(noise).all():
        raise ValueError("Valid flow motion and noise must be finite")
    if not torch.isfinite(times).all() or ((times < 0) | (times > 1)).any():
        raise ValueError("Flow times must be finite values in [0, 1]")
    t = times.float()[:, None, None]
    return (1 - t) * noise + t * motion, motion - noise


@torch.no_grad()
def update_ema(model: MotionFlow, ema: MotionFlow, global_step: int,
               maximum_decay: float) -> float:
    decay = min(float(maximum_decay), (1 + global_step) / (10 + global_step))
    for average, value in zip(ema.parameters(), model.parameters(), strict=True):
        average.lerp_(value.detach(), 1 - decay)
    for average, value in zip(ema.buffers(), model.buffers(), strict=True):
        average.copy_(value)
    return decay


@torch.inference_mode()
def validation_flow_mse(model: MotionFlow, loader, device: torch.device,
                        use_bfloat16: bool, evaluation_seed: int) -> float:
    """Fixed per-ID noise/time makes losses comparable between epochs."""
    model.eval()
    squared_error, elements = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        noise = seeded_noise(batch["sample_id"], evaluation_seed, 0,
                             tuple(batch["motion"].shape[1:]), device, stream="validation_noise")
        times = seeded_times(batch["sample_id"], evaluation_seed, device)
        state, velocity = interpolate_motion(batch["motion"], noise, times, batch["valid_frames"])
        with autocast(device, use_bfloat16):
            prediction = model(state, times, batch["tokens"], batch["fps"], batch["valid_frames"])
        errors = (prediction.float() - velocity)[batch["valid_frames"]]
        if not torch.isfinite(errors).all():
            raise FloatingPointError("Non-finite validation flow prediction")
        squared_error += float(errors.double().square().sum())
        elements += errors.numel()
    if elements == 0:
        raise ValueError("Validation has no valid frames")
    return squared_error / elements


def train(config: dict, *, resume: str | Path | None = None,
          stop_after_epoch: int | None = None) -> dict:
    """Select EMA by full validation flow loss and evaluate best after training."""
    device = resolve_device(str(config["device"]))
    if config["use_bfloat16"] and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested CUDA BF16 is not supported by this device")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any((output / name).exists() for name in ("latest.pth.tar", "best.pth.tar")):
        raise FileExistsError(f"Existing generation run: {output}; use --resume or another --output")
    if resume is not None and any((output / name).exists() for name in ("latest.pth.tar", "best.pth.tar")):
        if Path(resume).expanduser().resolve() != (output / "latest.pth.tar").resolve():
            raise FileExistsError(f"Resume destination contains another checkpoint: {output}; use a new --output")
    prepare_caches(config, splits=("train", "val"))
    datasets = {split: PredictionDataset(config, split) for split in ("train", "val")}
    provenance = {split: dataset.provenance for split, dataset in datasets.items()}
    _seed_all(int(config["seed"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    val_generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
    model = make_generator(datasets["train"], config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: learning_rate(epoch, config) / float(config["lr"]),
    )
    start_epoch, global_step, best_loss, history = 0, 0, float("inf"), []
    if resume is not None:
        resume = Path(resume).expanduser().resolve()
        saved = _torch_load_checkpoint(resume)
        if saved.get("format_version") != FORMAT_VERSION or saved.get("kind") != "motion_flow":
            raise ValueError("Unsupported generation training checkpoint")
        if saved["provenance"] != provenance:
            raise ValueError("Resume dataset/checkpoint/statistics provenance differs")
        for key in RESUME_FIELDS:
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume requires unchanged {key}")
        if (saved["model_info"] != datasets["train"].model_info
                or saved["token_layout"] != datasets["train"].token_layout.signature()
                or saved["flow_config"] != asdict(model.config)):
            raise ValueError("Resume flow architecture or feature layout differs")
        if (not torch.equal(saved["mean"], datasets["train"].mean)
                or not torch.equal(saved["std"], datasets["train"].std)):
            raise ValueError("Resume normalization statistics differ")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, global_step = int(saved["next_epoch"]), int(saved["global_step"])
        if saved["ema_updates"] != global_step:
            raise ValueError("Resume EMA update count differs from optimizer step count")
        best_loss, history = float(saved["best_val_flow_mse"]), saved["history"]
        restore_rng(saved["rng"], generator, device)
        prior_best = resume.parent / "best.pth.tar"
        if not (output / "best.pth.tar").exists():
            if not prior_best.is_file():
                raise FileNotFoundError("Relocated resume requires the previously selected best.pth.tar")
            shutil.copy2(prior_best, output / "best.pth.tar")
    _atomic_json_save(config, output / "config.json")
    train_loader = make_loader(datasets["train"], config, device, shuffle=True, generator=generator)
    val_loader = make_loader(datasets["val"], config, device, generator=val_generator)
    writer = None
    if config["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"), purge_step=global_step + 1 if resume else None)
    try:
        print(json.dumps({"event": "training_start", "kind": "motion_flow", "device": str(device),
                          "train_samples": len(datasets["train"]), "val_samples": len(datasets["val"]),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "start_epoch": start_epoch, "epochs": config["epochs"], "output": str(output)}), flush=True)
        for epoch in range(start_epoch, int(config["epochs"])):
            started = time.perf_counter()
            model.train()
            squared_error, elements = 0.0, 0
            progress = tqdm(train_loader, desc=f"Flow epoch {epoch + 1}/{config['epochs']}", mininterval=10)
            for batch in progress:
                batch = move_batch(batch, device)
                motion, valid = batch["motion"].float(), batch["valid_frames"]
                noise = torch.randn_like(motion)
                times = torch.rand(len(motion), device=device, dtype=torch.float32)
                dropped = torch.rand(len(motion), device=device) < float(config["condition_dropout"])
                state, velocity = interpolate_motion(motion, noise, times, valid)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, bool(config["use_bfloat16"])):
                    prediction = model(state, times, batch["tokens"], batch["fps"], valid,
                                       condition_drop=dropped)
                    loss = masked_mse(prediction.float(), velocity, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training flow loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]), error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                ema_decay = update_ema(model, ema, global_step, float(config["ema_decay"]))
                count = int(valid.sum()) * motion.shape[-1]
                squared_error += float(loss.detach()) * count
                elements += count
                if writer is not None and global_step % 20 == 0:
                    writer.add_scalar("train/step_flow_mse", float(loss.detach()), global_step)
                progress.set_postfix(flow_mse=f"{squared_error / elements:.5f}", refresh=False)
            lr = float(optimizer.param_groups[0]["lr"])
            val_loss = validation_flow_mse(ema, val_loader, device, bool(config["use_bfloat16"]),
                                            int(config["evaluation_seed"]))
            row = {"epoch": epoch + 1, "global_step": global_step, "train_flow_mse": squared_error / elements,
                   "val_flow_mse": val_loss, "lr": lr, "ema_decay": ema_decay,
                   "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            improved = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            scheduler.step()
            payload = {
                "format_version": FORMAT_VERSION, "kind": "motion_flow", "config": config,
                "flow_config": asdict(model.config), "model_info": datasets["train"].model_info,
                "token_layout": datasets["train"].token_layout.signature(), "provenance": provenance,
                "feature_transform": FEATURE_TRANSFORM, "mean": datasets["train"].mean,
                "std": datasets["train"].std, "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "next_epoch": epoch + 1, "global_step": global_step, "ema_updates": global_step,
                "best_val_flow_mse": best_loss, "history": history, "rng": capture_rng(generator, device),
            }
            _atomic_torch_save(payload, output / "latest.pth.tar")
            if improved:
                _atomic_torch_save(payload, output / "best.pth.tar")
            save_history(history, output)
            if writer is not None:
                for key in ("train_flow_mse", "val_flow_mse", "lr", "ema_decay"):
                    writer.add_scalar(f"epoch/{key}", row[key], global_step)
                writer.flush()
            print(json.dumps({"event": "epoch_complete", **row, "best_val_flow_mse": best_loss}), flush=True)
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                return {"output": str(output), "history": history, "complete": False}
    finally:
        if writer is not None:
            writer.close()
    from .evaluate import evaluate_checkpoint
    metrics = evaluate_checkpoint(output / "best.pth.tar", split="test", overrides=config)
    return {"output": str(output), "history": history, "complete": True, "test": metrics}
