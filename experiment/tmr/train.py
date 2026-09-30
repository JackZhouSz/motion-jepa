"""Train independent CLS alignment heads using prepared frozen token features."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from experiment.linear_probe.features import (
    PROJECT_ROOT,
    _atomic_json_save,
    _atomic_torch_save,
    _seed_all,
    _torch_load_checkpoint,
)
from .dataset import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_TEXT_MODEL,
    collate_pairs,
)
from .features import load_prepared_datasets
from .losses import symmetric_multi_positive_info_nce
from .model import AlignmentConfig, TextMotionAlignment
from .retrieval import evaluate_retrieval
from .utils import resolve_device


DATA_FIELDS = (
    "dataset_root", "annotations_path", "cache_root", "input_source",
    "jepa_checkpoint", "checkpoint_key", "stats_path", "text_model",
    "text_revision", "max_text_length",
)
OPTIMIZATION_FIELDS = (
    "seed", "epochs", "warmup_epochs", "batch_size", "lr", "final_lr",
    "weight_decay", "gradient_clip", "temperature", "use_bfloat16",
)


def data_configuration(args: argparse.Namespace) -> dict[str, Any]:
    values = {name: getattr(args, name) for name in DATA_FIELDS}
    for name in ("dataset_root", "annotations_path", "cache_root", "jepa_checkpoint", "stats_path"):
        if values[name] is not None:
            values[name] = str(Path(values[name]).expanduser().resolve())
    values["text_model"] = str(values["text_model"])
    return values


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate_split(
    model: TextMotionAlignment,
    dataset,
    device: torch.device,
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    use_bfloat16: bool = True,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Encode all motions and each distinct caption once, then rank the full gallery."""
    if not len(dataset):
        raise ValueError("Retrieval evaluation requires a nonempty split")
    dataset = copy.copy(dataset)
    dataset.sample_captions = False
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_pairs, pin_memory=device.type == "cuda",
    )
    motion_parts, text_parts = [], []
    motion_ids = dataset.caption_candidates
    text_ids = list(dict.fromkeys(caption for candidates in motion_ids for caption in candidates))
    for batch in loader:
        with _autocast(device, use_bfloat16):
            motion = model.encode_motion(
                batch["motion_tokens"].to(device, non_blocking=True),
                batch["motion_mask"].to(device, non_blocking=True),
            )
        motion_parts.append(motion.float().cpu())
    for start in range(0, len(text_ids), batch_size):
        tokens = [dataset.text_bank[caption_id] for caption_id in text_ids[start:start + batch_size]]
        padded = pad_sequence(tokens, batch_first=True).to(device)
        lengths = torch.tensor([len(value) for value in tokens], device=device)
        mask = torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]
        with _autocast(device, use_bfloat16):
            text = model.encode_text(padded, mask)
        text_parts.append(text.float().cpu())
    return evaluate_retrieval(
        torch.cat(motion_parts).to(device), torch.cat(text_parts).to(device),
        motion_ids, text_ids, chunk_size=chunk_size,
    )


def _capture_rng(loader_generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "loader_generator": loader_generator.get_state(),
    }


def _restore_rng(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader_generator"])


def _lr_factor(epoch: int, *, epochs: int, warmup_epochs: int, final_factor: float) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    remaining = epochs - warmup_epochs
    progress = min(1.0, (epoch - warmup_epochs) / max(1, remaining - 1))
    return final_factor + (1 - final_factor) * (1 + math.cos(math.pi * progress)) / 2


def _write_metrics(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _tensorboard_writer(args, output: Path, global_step: int):
    if not args.tensorboard:
        return nullcontext(None)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requires tensorboard; install it in motion-jepa with "
            "python -m pip install -r experiment/tmr/requirements.txt, or use --no-tensorboard"
        ) from error
    directory = Path(args.tensorboard_dir or output / "tensorboard").expanduser().resolve()
    # The saved checkpoint includes evaluation at global_step. Discard only later
    # events from uncommitted optimizer steps when resuming an interrupted run.
    return SummaryWriter(log_dir=str(directory), purge_step=global_step + 1 if args.resume else None)


def load_alignment_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = _torch_load_checkpoint(path)
    if checkpoint.get("format_version") != 1 or checkpoint.get("experiment") != "tmr_alignment":
        raise ValueError(f"Not a supported TMR alignment checkpoint: {path}")
    for key in ("model", "config", "provenance"):
        if key not in checkpoint:
            raise ValueError(f"TMR checkpoint has no {key!r}: {path}")
    return checkpoint


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("epochs must be positive and warmup_epochs in [0, epochs)")
    if args.batch_size < 2 or args.eval_batch_size < 1 or args.num_workers < 0 or args.chunk_size < 1:
        raise ValueError("Training batch size must be >=2; evaluation sizes positive; workers >=0")
    for name in ("lr", "final_lr", "temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.final_lr > args.lr:
        raise ValueError("final_lr cannot exceed lr")
    for name in ("weight_decay", "gradient_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.input_source == "jepa" and args.jepa_checkpoint is None:
        raise ValueError("--jepa-checkpoint is required for --input-source jepa")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output = Path(args.output_root or PROJECT_ROOT / "output/tmr" / args.input_source / f"seed-{args.seed}")
    output = output.expanduser().resolve()
    latest_path, best_path = output / "latest.pth.tar", output / "best.pth.tar"
    existing = [name for name in ("config.json", "metrics.csv", "latest.pth.tar", "best.pth.tar", "summary.json") if (output / name).exists()]
    if existing and not args.resume:
        raise FileExistsError(f"TMR output already exists under {output}; use --resume or another --output-root")
    if args.resume and not latest_path.is_file():
        raise FileNotFoundError(f"--resume requires {latest_path}")
    device = resolve_device(args.device)
    _seed_all(args.seed)
    data_config = data_configuration(args)
    datasets, provenance = load_prepared_datasets(**data_config)
    if len(datasets["train"]) < 2 or not len(datasets["val"]):
        raise ValueError("TMR needs at least two training pairs and a nonempty validation split")
    model_config = AlignmentConfig(
        text_dim=provenance["text_dim"], motion_dim=provenance["motion_dim"],
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        ff_dim=args.ff_dim, dropout=args.dropout,
    )
    config = {
        "model": asdict(model_config), "data": data_config,
        "optimization": {name: getattr(args, name) for name in OPTIMIZATION_FIELDS},
    }
    model = TextMotionAlignment(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: _lr_factor(
            epoch, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
            final_factor=args.final_lr / args.lr,
        ),
    )
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch, best_epoch, best_score, global_step = 0, 0, -math.inf, 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = load_alignment_checkpoint(latest_path)
        if checkpoint["config"] != config or checkpoint["provenance"] != provenance:
            raise ValueError("Resume config or feature provenance differs from the saved run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        best_epoch, best_score = int(checkpoint["best_epoch"]), float(checkpoint["best_val_mean_r1"])
        global_step = int(checkpoint.get("global_step", 0))
        history = checkpoint["history"]
        _restore_rng(checkpoint["rng_state"], generator)
        if not 0 <= start_epoch <= args.epochs or len(history) != start_epoch:
            raise ValueError("Invalid next_epoch/history in resume checkpoint")
        if not best_path.is_file():
            raise FileNotFoundError(f"Resume requires the selected checkpoint: {best_path}")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(config, output / "config.json")
    _atomic_json_save(provenance, output / "provenance.json")
    _atomic_json_save({
        "python": sys.executable, "prefix": sys.prefix, "torch": str(torch.__version__),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }, output / "runtime.json")
    _write_metrics(history, output / "metrics.csv")
    with _tensorboard_writer(args, output, global_step) as writer:
        loader = DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            generator=generator, num_workers=args.num_workers, collate_fn=collate_pairs,
            # Recreate worker iterators each epoch so their generator consumption is
            # identical after an epoch-boundary resume, including with workers > 0.
            pin_memory=device.type == "cuda",
        )
        for epoch in range(start_epoch, args.epochs):
            model.train()
            weighted_loss, sample_count, started = 0.0, 0, time.monotonic()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            for batch in loader:
                count = len(batch["caption_ids"])
                if count < 2:
                    continue  # A singleton has no negative and contributes zero contrastive signal.
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, args.use_bfloat16):
                    motion, text = model(
                        batch["motion_tokens"].to(device, non_blocking=True),
                        batch["motion_mask"].to(device, non_blocking=True),
                        batch["text_tokens"].to(device, non_blocking=True),
                        batch["text_mask"].to(device, non_blocking=True),
                    )
                    loss = symmetric_multi_positive_info_nce(
                        motion, text, batch["caption_ids"], batch["source_ids"],
                        batch["start_frames"], batch["end_frames"], temperature=args.temperature,
                        caption_candidate_ids=batch["caption_candidate_ids"],
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite TMR loss at epoch {epoch + 1}, step {global_step}")
                loss.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                loss_value = float(loss.detach())
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, global_step)
                    writer.add_scalar("train/learning_rate", learning_rate, global_step)
                weighted_loss += loss_value * count
                sample_count += count
            if not sample_count:
                raise ValueError("No trainable contrastive batches; at least two pairs are required")
            validation = evaluate_split(
                model, datasets["val"], device, batch_size=args.eval_batch_size,
                num_workers=args.num_workers, use_bfloat16=args.use_bfloat16, chunk_size=args.chunk_size,
            )
            row = {
                "epoch": epoch + 1, "lr": learning_rate, "train_loss": weighted_loss / sample_count,
                "train_samples": sample_count, **{f"val_{key}": value for key, value in validation.items()},
            }
            history.append(row)
            improved = validation["mean_r1"] > best_score
            if improved:
                best_epoch, best_score = epoch + 1, float(validation["mean_r1"])
            if writer is not None:
                writer.add_scalar("train/epoch", epoch + 1, global_step)
                writer.add_scalar("train/epoch_loss", row["train_loss"], global_step)
                writer.add_scalar("train/epoch_samples", sample_count, global_step)
                writer.add_scalar("train/epoch_seconds", time.monotonic() - started, global_step)
                for key, value in validation.items():
                    writer.add_scalar(f"val/{key}", value, global_step)
                writer.add_scalar("val/best_mean_r1", best_score, global_step)
                writer.add_scalar("val/best_epoch", best_epoch, global_step)
            scheduler.step()
            checkpoint = {
                "format_version": 1, "experiment": "tmr_alignment", "config": config,
                "provenance": provenance, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "next_epoch": epoch + 1, "global_step": global_step,
                "best_epoch": best_epoch, "best_val_mean_r1": best_score,
                "history": history, "rng_state": _capture_rng(generator, device),
            }
            _write_metrics(history, output / "metrics.csv")
            if improved:
                _atomic_torch_save(checkpoint, best_path)
            _atomic_torch_save(checkpoint, latest_path)
            if writer is not None:
                writer.flush()
            print(
                f"TMR {args.input_source} epoch {epoch + 1}/{args.epochs}: "
                f"loss={row['train_loss']:.5f}, val mean R@1={best_score:.4f} "
                f"(current={validation['mean_r1']:.4f}), {time.monotonic() - started:.1f}s",
                flush=True,
            )
        selected = load_alignment_checkpoint(best_path)
        if selected["config"] != config or selected["provenance"] != provenance:
            raise ValueError("Best checkpoint does not belong to this TMR run")
        model.load_state_dict(selected["model"], strict=True)
        test = evaluate_split(
            model, datasets["test"], device, batch_size=args.eval_batch_size,
            num_workers=args.num_workers, use_bfloat16=args.use_bfloat16, chunk_size=args.chunk_size,
        ) if len(datasets["test"]) else None
        if writer is not None and test is not None:
            for key, value in test.items():
                writer.add_scalar(f"test/{key}", value, global_step)
        summary = {
            "input_source": args.input_source, "output_root": str(output),
            "selection": "mean_bidirectional_val_r1", "best_epoch": best_epoch,
            "best_val": {key.removeprefix("val_"): value for key, value in history[best_epoch - 1].items() if key.startswith("val_")},
            "test": test, "epochs_completed": len(history), "global_step": global_step,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path),
            "tensorboard_dir": str(Path(args.tensorboard_dir or output / "tensorboard").expanduser().resolve()) if args.tensorboard else None,
        }
        _atomic_json_save(summary, output / "summary.json")
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-source", choices=("raw", "jepa"), default="raw")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotations-path", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--checkpoint-key", choices=("target_encoder", "encoder"), default="target_encoder")
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--text-revision")
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--use-bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
