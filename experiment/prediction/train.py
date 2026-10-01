"""Train only a deterministic decoder from completed frozen-feature caches."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment.linear_probe.features import (
    _atomic_json_save, _atomic_torch_save, _seed_all, _torch_load_checkpoint, resolve_device,
)
from .data import PredictionDataset, prepare_caches
from .metrics import masked_mse
from .model import DecoderConfig, MotionDecoder

FORMAT_VERSION = 1
RESUME_FIELDS = ("seed", "epochs", "batch_size", "lr", "final_lr", "weight_decay",
                 "warmup_epochs", "gradient_clip", "use_bfloat16", "decoder")


def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_decoder(dataset: PredictionDataset, config: dict) -> MotionDecoder:
    return MotionDecoder(
        feature_dim=int(dataset.model_info["feature_dim"]),
        token_layout=dataset.token_layout,
        motion_dim=int(dataset.model_info["motion_dim"]),
        config=DecoderConfig(**config["decoder"]),
    )


def make_loader(dataset, config: dict, device: torch.device, *, shuffle: bool = False,
                generator: torch.Generator | None = None) -> DataLoader:
    return DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=shuffle,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
        drop_last=False, generator=generator,
        # Recreating workers each epoch makes loader RNG restoration reproducible.
        persistent_workers=False,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def capture_rng(generator: torch.Generator, device: torch.device) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state: dict, generator: torch.Generator, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def learning_rate(epoch: int, config: dict) -> float:
    warmup = min(int(config["warmup_epochs"]), int(config["epochs"]))
    if epoch < warmup:
        return float(config["lr"]) * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, int(config["epochs"]) - warmup - 1)
    return float(config["final_lr"]) + (float(config["lr"]) - float(config["final_lr"])) * (1 + math.cos(math.pi * progress)) / 2


@torch.inference_mode()
def validation_mse(model: MotionDecoder, loader: DataLoader, device: torch.device,
                   use_bfloat16: bool) -> float:
    model.eval()
    squared_error, elements = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast(device, use_bfloat16):
            prediction = model(batch["tokens"], batch["fps"], batch["valid_frames"])
        error = (prediction.float() - batch["motion"].float())[batch["valid_frames"]]
        if not torch.isfinite(error).all():
            raise FloatingPointError("Non-finite validation reconstruction")
        squared_error += float(error.double().square().sum())
        elements += error.numel()
    if not elements:
        raise ValueError("Validation split has no valid motion frames")
    return squared_error / elements


def save_history(history: list[dict], output: Path) -> None:
    temporary = output / "metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output / "metrics.csv")


def train(config: dict, *, resume: Path | None = None,
          stop_after_epoch: int | None = None) -> dict:
    """Train, select by validation MSE, then evaluate best on the test split.

    stop_after_epoch is an epoch-boundary interruption hook for resume checks.
    """
    device = resolve_device(str(config["device"]))
    if config["use_bfloat16"] and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 training is not supported by this CUDA device")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "latest.pth.tar").exists():
        raise FileExistsError(f"Existing training run: {output}; use --resume or another --output")
    if resume is not None and any((output / name).exists() for name in ("latest.pth.tar", "best.pth.tar")):
        if Path(resume).expanduser().resolve() != (output / "latest.pth.tar").resolve():
            raise FileExistsError(f"Resume destination already contains another checkpoint: {output}; use a new --output")
    prepare_caches(config, splits=("train", "val"))
    datasets = {split: PredictionDataset(config, split) for split in ("train", "val")}
    provenance = {split: dataset.provenance for split, dataset in datasets.items()}
    _seed_all(int(config["seed"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    val_generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
    model = make_decoder(datasets["train"], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    # Epoch-indexed schedule retains its configuration in every checkpoint.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: learning_rate(epoch, config) / float(config["lr"]),
    )
    start_epoch, global_step, best_mse, history = 0, 0, float("inf"), []
    if resume is not None:
        saved = _torch_load_checkpoint(Path(resume).expanduser().resolve())
        if saved.get("format_version") != FORMAT_VERSION or saved.get("kind") != "motion_decoder":
            raise ValueError("Unsupported prediction training checkpoint")
        if saved["provenance"] != provenance:
            raise ValueError("Resume dataset/checkpoint/statistics provenance differs")
        for key in RESUME_FIELDS:
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume requires unchanged {key}")
        if (saved["model_info"] != datasets["train"].model_info
                or saved["token_layout"] != datasets["train"].token_layout.signature()):
            raise ValueError("Resume decoder feature layout differs")
        model.load_state_dict(saved["decoder"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, global_step = int(saved["next_epoch"]), int(saved["global_step"])
        best_mse, history = float(saved["best_val_mse"]), saved["history"]
        restore_rng(saved["rng"], generator, device)
        # A relocated run must keep its previously selected best model too.
        prior_best = Path(resume).resolve().parent / "best.pth.tar"
        if not (output / "best.pth.tar").exists() and prior_best.exists():
            import shutil
            shutil.copy2(prior_best, output / "best.pth.tar")
    _atomic_json_save(config, output / "config.json")
    train_loader = make_loader(datasets["train"], config, device, shuffle=True, generator=generator)
    val_loader = make_loader(datasets["val"], config, device, generator=val_generator)
    writer = None
    if config["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"), purge_step=global_step + 1 if resume else None)
    try:
        print(json.dumps({"event": "training_start", "device": str(device), "train_samples": len(datasets["train"]),
                          "val_samples": len(datasets["val"]), "decoder_parameters": sum(p.numel() for p in model.parameters()),
                          "start_epoch": start_epoch, "epochs": config["epochs"], "output": str(output)}), flush=True)
        for epoch in range(start_epoch, int(config["epochs"])):
            started = time.perf_counter()
            model.train()
            squared_error, elements = 0.0, 0
            progress = tqdm(train_loader, desc=f"Decoder epoch {epoch + 1}/{config['epochs']}", mininterval=10)
            for batch in progress:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, bool(config["use_bfloat16"])):
                    prediction = model(batch["tokens"], batch["fps"], batch["valid_frames"])
                    loss = masked_mse(prediction.float(), batch["motion"].float(), batch["valid_frames"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite decoder training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]), error_if_nonfinite=True)
                optimizer.step()
                count = int(batch["valid_frames"].sum()) * int(batch["motion"].shape[-1])
                squared_error += float(loss.detach()) * count
                elements += count
                global_step += 1
                if writer is not None and global_step % 20 == 0:
                    writer.add_scalar("train/step_mse", float(loss.detach()), global_step)
                progress.set_postfix(mse=f"{squared_error / elements:.5f}", refresh=False)
            lr = float(optimizer.param_groups[0]["lr"])
            val_mse = validation_mse(model, val_loader, device, bool(config["use_bfloat16"]))
            row = {"epoch": epoch + 1, "global_step": global_step, "train_mse": squared_error / elements,
                   "val_mse": val_mse, "lr": lr, "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            improved = val_mse < best_mse
            best_mse = min(best_mse, val_mse)
            scheduler.step()
            payload = {
                "format_version": FORMAT_VERSION, "kind": "motion_decoder", "config": config,
                "decoder_config": asdict(model.config), "model_info": datasets["train"].model_info,
                "token_layout": datasets["train"].token_layout.signature(), "provenance": provenance,
                "feature_transform": "jepa_target_layer_norm", "mean": datasets["train"].mean,
                "std": datasets["train"].std, "decoder": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "next_epoch": epoch + 1, "global_step": global_step, "best_val_mse": best_mse,
                "history": history, "rng": capture_rng(generator, device),
            }
            _atomic_torch_save(payload, output / "latest.pth.tar")
            if improved:
                _atomic_torch_save(payload, output / "best.pth.tar")
            save_history(history, output)
            if writer is not None:
                for key in ("train_mse", "val_mse", "lr"):
                    writer.add_scalar(f"epoch/{key}", row[key], global_step)
                writer.flush()
            print(json.dumps({"event": "epoch_complete", **row, "best_val_mse": best_mse}), flush=True)
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                return {"output": str(output), "history": history, "complete": False}
    finally:
        if writer is not None:
            writer.close()
    from .evaluate import evaluate_checkpoint
    metrics = evaluate_checkpoint(output / "best.pth.tar", split="test", overrides=config)
    return {"output": str(output), "history": history, "complete": True, "test": metrics}
