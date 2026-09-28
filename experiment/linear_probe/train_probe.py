"""Train a linear probe on frozen MotionJEPA features."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import MultiLabelIndex, SingleLabelIndex, build_classification_datasets
from .features import (
    PROJECT_ROOT,
    SPLITS,
    GLOBAL_MEAN_POOLING,
    SPATIAL_FLATTEN_POOLING,
    Metrics,
    _atomic_json_save,
    _atomic_torch_save,
    _seed_all,
    build_cache_metadata,
    load_frozen_encoder,
    load_or_extract_split,
    resolve_device,
    resolve_pretraining_stats,
)


RESULT_FILENAMES = (
    "metrics.csv",
    "summary.json",
    "linear-probe-best.pth.tar",
    "linear-probe-final.pth.tar",
    "class-index.json",
)


def evaluate_classifier(
    classifier: nn.Linear,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Metrics:
    classifier.eval()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    loss_sum = 0.0
    total = 0
    correct = 0
    top5_correct = 0
    class_total = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = classifier(batch_features)
            loss_sum += float(F.cross_entropy(logits, batch_labels, reduction="sum"))
            predictions = logits.argmax(dim=1)
            matches = predictions.eq(batch_labels)
            correct += int(matches.sum())
            total += len(batch_labels)
            topk = min(5, num_classes)
            top5_correct += int(
                logits.topk(topk, dim=1)
                .indices.eq(batch_labels[:, None])
                .any(dim=1)
                .sum()
            )
            cpu_labels = batch_labels.cpu()
            class_total += torch.bincount(cpu_labels, minlength=num_classes)
            class_correct += torch.bincount(
                cpu_labels[matches.cpu()], minlength=num_classes
            )
    if total == 0:
        raise ValueError("Cannot evaluate an empty feature split")
    present = class_total > 0
    macro = (class_correct[present].float() / class_total[present].float()).mean()
    return Metrics(
        loss=loss_sum / total,
        top1_accuracy=correct / total,
        macro_accuracy=float(macro),
        top5_accuracy=top5_correct / total,
    )


def train_linear_probe(
    caches: dict[str, dict[str, Any]],
    *,
    output: Path | None,
    checkpoint_path: Path | None,
    checkpoint_key: str,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    run_args: dict[str, Any],
) -> dict[str, Any]:
    train_features = caches["train"]["features"]
    train_labels = caches["train"]["labels"]
    feature_dim = int(train_features.shape[1])
    num_classes = len(class_names)
    _seed_all(seed)
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_used = int(len(caches["val"]["labels"])) > 0
    selection = "validation_best" if validation_used else "fixed_last_epoch"
    metrics_path = None if output is None else output / "metrics.csv"
    best_path = (
        None
        if output is None
        else output
        / ("linear-probe-best.pth.tar" if validation_used else "linear-probe-final.pth.tar")
    )
    best_accuracy = -1.0
    best_epoch = 0
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_top1_accuracy",
        "train_macro_accuracy",
        "train_top5_accuracy",
        "val_loss",
        "val_top1_accuracy",
        "val_macro_accuracy",
        "val_top5_accuracy",
    ]
    file = (
        None
        if metrics_path is None
        else metrics_path.open("w", encoding="utf-8", newline="")
    )
    try:
        writer = None if file is None else csv.DictWriter(file, fieldnames=fields)
        if writer is not None:
            writer.writeheader()
        best = None
        for epoch in range(1, epochs + 1):
            classifier.train()
            current_lr = float(optimizer.param_groups[0]["lr"])
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(classifier(batch_features), batch_labels)
                loss.backward()
                optimizer.step()
            train_metrics = evaluate_classifier(
                classifier,
                train_features,
                train_labels,
                device=device,
                batch_size=batch_size,
                num_classes=num_classes,
            )
            val_metrics = None
            if validation_used:
                val_metrics = evaluate_classifier(
                    classifier,
                    caches["val"]["features"],
                    caches["val"]["labels"],
                    device=device,
                    batch_size=batch_size,
                    num_classes=num_classes,
                )
            row = {
                "epoch": epoch,
                "learning_rate": current_lr,
                **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
                **(
                    {f"val_{key}": value for key, value in asdict(val_metrics).items()}
                    if val_metrics is not None
                    else {f"val_{key}": "" for key in asdict(train_metrics)}
                ),
            }
            if writer is not None:
                writer.writerow(row)
                file.flush()
            if val_metrics is not None and val_metrics.top1_accuracy > best_accuracy:
                best_accuracy = val_metrics.top1_accuracy
                best_epoch = epoch
                best = {
                    "format_version": 1,
                    "classifier": copy.deepcopy(classifier.state_dict()),
                    "feature_dim": feature_dim,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_metrics": asdict(val_metrics),
                    "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
                    "checkpoint_key": checkpoint_key,
                    "run_args": run_args,
                }
                if best_path is not None:
                    _atomic_torch_save(best, best_path)
            scheduler.step()
        if not validation_used:
            best_epoch = epochs
            best = {
                "format_version": 1,
                "classifier": copy.deepcopy(classifier.state_dict()),
                "feature_dim": feature_dim,
                "num_classes": num_classes,
                "class_names": class_names,
                "epoch": epochs,
                "val_metrics": None,
                "selection": selection,
                "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
                "checkpoint_key": checkpoint_key,
                "run_args": run_args,
            }
            if best_path is not None:
                _atomic_torch_save(best, best_path)
    finally:
        if file is not None:
            file.close()

    if best_path is not None:
        best = torch.load(best_path, map_location=device, weights_only=False)
    if best is None:
        raise RuntimeError("Linear probe did not produce a final head")
    classifier.load_state_dict(best["classifier"], strict=True)
    test_metrics = evaluate_classifier(
        classifier,
        caches["test"]["features"],
        caches["test"]["labels"],
        device=device,
        batch_size=batch_size,
        num_classes=num_classes,
    )
    return {
        "best_epoch": best_epoch,
        "best_val": best["val_metrics"],
        "selection": selection,
        "validation_used": validation_used,
        "head_filename": None if best_path is None else best_path.name,
        "test": asdict(test_metrics),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "split_counts": {
            split: int(len(caches[split]["labels"])) for split in SPLITS
        },
    }


def _evaluate_multilabel_linear_probe(
    classifier: nn.Linear,
    cache: dict[str, Any],
    *,
    label_index: MultiLabelIndex,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    # Share BABEL's AP and hit definitions with the offline classifier runner.
    from .train_classifier import MultiLabelMetricAccumulator

    classifier.eval()
    features = cache["features"]
    labels = cache["labels"]
    sample_ids = cache["sample_ids"]
    metrics = MultiLabelMetricAccumulator(
        label_index.num_classes, label_index.row_labels_by_path
    )
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            inputs = features[start:end].to(device=device, dtype=torch.float32)
            targets = labels[start:end].to(device=device, dtype=torch.float32)
            logits = classifier(inputs)
            loss_sum = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum"
            ) / label_index.num_classes
            metrics.update(logits, targets, float(loss_sum), sample_ids[start:end])
    return asdict(metrics.compute())


def train_multilabel_probe(
    caches: dict[str, dict[str, Any]],
    *,
    label_index: MultiLabelIndex,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    """Fit a fresh in-memory BABEL head and select it by validation mAP."""
    train = caches["train"]
    validation = caches["val"]
    feature_dim = int(train["features"].shape[1])
    num_classes = label_index.num_classes
    if not len(train["labels"]) or not len(validation["labels"]):
        raise ValueError("BABEL online probing requires nonempty train and val splits")
    for split, cache in (("train", train), ("val", validation)):
        if (
            cache["features"].ndim != 2
            or cache["features"].shape[1] != feature_dim
            or cache["labels"].shape != (len(cache["features"]), num_classes)
            or cache["labels"].dtype != torch.float32
            or len(cache["sample_ids"]) != len(cache["features"])
        ):
            raise ValueError(f"Invalid BABEL online feature cache for {split}")

    _seed_all(seed)
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=learning_rate, momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(
        TensorDataset(train["features"], train["labels"]),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_score = float("-inf")
    best_epoch = None
    best_metrics = None
    for epoch in range(1, epochs + 1):
        classifier.train()
        for features, labels in loader:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(classifier(features), labels)
            loss.backward()
            optimizer.step()
        metrics = _evaluate_multilabel_linear_probe(
            classifier, validation, label_index=label_index,
            device=device, batch_size=batch_size,
        )
        score = float(metrics["mean_average_precision"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
        scheduler.step()
    return {
        "best_epoch": best_epoch,
        "best_val": best_metrics,
        "selection": "validation_best",
        "validation_used": True,
        "test": None,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "split_counts": {"train": len(train["labels"]), "val": len(validation["labels"]), "test": 0},
    }


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("Epoch and batch-size arguments must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pooling = str(getattr(args, "pooling", GLOBAL_MEAN_POOLING))
    output = (
        checkpoint_path.parent
        / ("linear-probe-2d" if pooling == SPATIAL_FLATTEN_POOLING else "linear-probe")
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    existing_results = [name for name in RESULT_FILENAMES if (output / name).exists()]
    if existing_results and not args.overwrite:
        raise FileExistsError(
            f"Linear-probe results already exist under {output}: {existing_results}; "
            "use --overwrite to replace result files"
        )
    if args.overwrite:
        for name in RESULT_FILENAMES:
            path = output / name
            if path.is_file():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = output / "features"
    cache_root.mkdir(exist_ok=True)
    device = resolve_device(args.device)
    _seed_all(args.seed)
    encoder, config, model_info = load_frozen_encoder(
        checkpoint_path,
        args.checkpoint_key,
        device,
    )
    if pooling == SPATIAL_FLATTEN_POOLING and model_info["kind"] != "2d":
        raise ValueError(
            f"{SPATIAL_FLATTEN_POOLING} requires a 2D encoder, "
            f"got {model_info['model_name']}"
        )
    stats_root = resolve_pretraining_stats(config, args.stats_path)
    datasets, label_index = build_classification_datasets(
        dataset_root,
        splits=SPLITS,
        num_frames=model_info["num_frames"],
        fps=model_info["fps"],
        motion_dim=model_info["motion_dim"],
        stats_root=stats_root,
    )
    if not isinstance(label_index, SingleLabelIndex):
        raise ValueError("This probe command requires single-label classification data")
    class_names = list(label_index.class_names)
    _atomic_json_save(label_index.to_json(), output / "class-index.json")
    train_classes = set(datasets["train"].labels)
    expected_classes = set(range(len(class_names)))
    if train_classes != expected_classes:
        missing = [class_names[index] for index in sorted(expected_classes - train_classes)]
        raise ValueError(f"Training split does not contain every class: {missing}")

    caches = {}
    for split in SPLITS:
        metadata = build_cache_metadata(
            split=split,
            checkpoint_path=checkpoint_path,
            checkpoint_key=args.checkpoint_key,
            dataset_root=dataset_root,
            stats_root=stats_root,
            model_info=model_info,
            class_names=class_names,
            pooling=pooling,
        )
        if len(datasets[split]) == 0:
            caches[split] = {
                "metadata": metadata,
                "features": torch.empty((0, int(metadata["feature_dim"]))),
                "labels": torch.empty((0,), dtype=torch.long),
                "sample_ids": [],
            }
        else:
            caches[split] = load_or_extract_split(
                split=split,
                cache_path=cache_root / f"{split}.pt",
                metadata=metadata,
                dataset=datasets[split],
                encoder=encoder,
                device=device,
                batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
                use_bfloat16=model_info["use_bfloat16"],
                recompute=args.recompute_features,
                pooling=pooling,
            )

    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("Frozen encoder unexpectedly accumulated gradients")
    summary = train_linear_probe(
        caches,
        output=output,
        checkpoint_path=checkpoint_path,
        checkpoint_key=args.checkpoint_key,
        class_names=class_names,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        run_args=_serializable_args(args),
    )
    dataset_metadata = json.loads((dataset_root / "meta.json").read_text(encoding="utf-8"))
    test_contents = dataset_metadata.get("test_contents", [])
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_key": args.checkpoint_key,
            "dataset_root": str(dataset_root),
            "stats_root": str(stats_root),
            "model_name": model_info["model_name"],
            "pooling": pooling,
            "seed": args.seed,
            "test_content": test_contents[0] if len(test_contents) == 1 else test_contents,
        }
    )
    _atomic_json_save(summary, output / "summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linear-probe MotionJEPA features")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-soma77-processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Linear-probe result directory "
            "(default: <checkpoint directory>/linear-probe, or linear-probe-2d "
            "for temporal_mean_spatial_flatten)"
        ),
    )
    parser.add_argument(
        "--checkpoint-key",
        choices=("target_encoder", "encoder"),
        default="target_encoder",
    )
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pooling",
        choices=(GLOBAL_MEAN_POOLING, SPATIAL_FLATTEN_POOLING),
        default=GLOBAL_MEAN_POOLING,
        help=(
            "Feature pooling before the linear head. temporal_mean_spatial_flatten "
            "is the group-aware 2D probe."
        ),
    )
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
