"""Resumable linear-probe learning-rate sweeps over every latest checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SIZE_RANK = {
    name: rank
    for rank, name in enumerate(("tiny", "small", "base", "large", "huge", "giant"))
}
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from . import features  # noqa: E402
from .dataset import (  # noqa: E402
    SingleLabelIndex,
    SingleLabelMotionDataset,
    build_classification_datasets,
)
from .train_probe import train_linear_probe  # noqa: E402


DEFAULT_LRS = (0.1, 0.3, 0.5, 0.7, 1.0)
DEFAULT_SEEDS = (42,)
METRIC_NAMES = (
    "loss",
    "top1_accuracy",
    "macro_accuracy",
    "top5_accuracy",
)


def require_validation_dataset(dataset_root: Path) -> None:
    """Reject LR selection when the dataset intentionally has no validation set."""
    meta_path = dataset_root / "meta.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("validation_enabled") is False:
            raise ValueError(
                "LR sweep is disabled for validation-free datasets; keep the "
                "pre-registered LR instead of selecting it on the test content"
            )
    val_manifest = dataset_root / "val.txt"
    if val_manifest.is_file() and not val_manifest.read_text(encoding="utf-8").strip():
        raise ValueError(
            "LR sweep requires a non-empty validation split and may not use test "
            "performance for model selection"
        )


def has_validation_dataset(dataset_root: Path) -> bool:
    """Return whether the probe dataset has a usable validation split."""
    meta_path = dataset_root / "meta.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("validation_enabled") is False:
            return False
    val_manifest = dataset_root / "val.txt"
    return val_manifest.is_file() and bool(
        val_manifest.read_text(encoding="utf-8").strip()
    )


def resolve_checkpoints(
    output_root: Path, checkpoint_values: list[Path] | None
) -> list[tuple[str, Path]]:
    """Resolve explicit immutable checkpoints or retain legacy latest discovery."""
    if not checkpoint_values:
        return discover_latest_checkpoints(output_root)
    resolved: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for value in checkpoint_values:
        checkpoint = Path(value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        label = f"{checkpoint.parent.name}--{checkpoint.name.removesuffix('.pth.tar')}"
        resolved.append((label, checkpoint))
    return resolved


def discover_latest_checkpoints(output_root: Path) -> list[tuple[str, Path]]:
    """Find one latest checkpoint in each direct child training directory."""
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")
    discovered: list[tuple[str, Path]] = []
    for run_root in sorted(path for path in output_root.iterdir() if path.is_dir()):
        matches = sorted(run_root.glob("*-latest.pth.tar"))
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(
                f"Expected one latest checkpoint under {run_root}, found {matches}"
            )
        discovered.append((run_root.name, matches[0].resolve()))
    if not discovered:
        raise FileNotFoundError(f"No *-latest.pth.tar checkpoints under {output_root}")
    return discovered


def lr_slug(value: float) -> str:
    return f"{value:.10g}".replace(".", "p").replace("-", "m")


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
    )


def _feature_batch_candidates(initial: int) -> list[int]:
    candidates = []
    value = initial
    while value >= 1:
        candidates.append(value)
        if value == 1:
            break
        value = max(1, value // 2)
    return candidates


def load_or_extract_adaptive(
    *,
    split: str,
    cache_path: Path,
    metadata: dict[str, Any],
    dataset: SingleLabelMotionDataset,
    encoder: torch.nn.Module,
    device: torch.device,
    initial_batch_size: int,
    num_workers: int,
    use_bfloat16: bool,
    recompute: bool,
    pooling: str = features.GLOBAL_MEAN_POOLING,
) -> tuple[dict[str, Any], int]:
    """Reuse a valid cache or reduce feature batch size after CUDA OOM."""
    if cache_path.is_file() and not recompute:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            features._validate_feature_cache(payload, metadata)
            return payload, initial_batch_size
        except (ValueError, KeyError, TypeError):
            recompute = True

    last_error: BaseException | None = None
    for batch_size in _feature_batch_candidates(initial_batch_size):
        try:
            payload = features.load_or_extract_split(
                split=split,
                cache_path=cache_path,
                metadata=metadata,
                dataset=dataset,
                encoder=encoder,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                use_bfloat16=use_bfloat16,
                recompute=True,
                pooling=pooling,
            )
            return payload, batch_size
        except BaseException as error:
            if not _is_cuda_oom(error) or batch_size == 1:
                raise
            last_error = error
            if device.type == "cuda":
                torch.cuda.empty_cache()
    assert last_error is not None
    raise last_error


def prepare_checkpoint_features(
    *,
    run_name: str,
    checkpoint_path: Path,
    dataset_root: Path,
    device: torch.device,
    feature_batch_size: int,
    num_workers: int,
    recompute_features: bool,
    pooling: str = features.GLOBAL_MEAN_POOLING,
    cache_root: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    encoder, config, model_info = features.load_frozen_encoder(
        checkpoint_path, "target_encoder", device
    )
    stats_root = features.resolve_pretraining_stats(config, None)
    datasets, label_index = build_classification_datasets(
        dataset_root,
        splits=features.SPLITS,
        num_frames=model_info["num_frames"],
        fps=model_info["fps"],
        motion_dim=model_info["motion_dim"],
        stats_root=stats_root,
    )
    if not isinstance(label_index, SingleLabelIndex):
        raise ValueError("The top-1 LR sweep requires single-label classification data")
    class_names = list(label_index.class_names)
    train_classes = set(datasets["train"].labels)
    if train_classes != set(range(len(class_names))):
        missing = [
            class_names[index]
            for index in sorted(set(range(len(class_names))) - train_classes)
        ]
        raise ValueError(f"Training split is missing classes: {missing}")

    cache_root = (
        checkpoint_path.parent / "linear-probe" / "features"
        if cache_root is None
        else cache_root
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    caches: dict[str, dict[str, Any]] = {}
    used_batch_sizes: dict[str, int] = {}
    next_batch_size = feature_batch_size
    for split in features.SPLITS:
        metadata = features.build_cache_metadata(
            split=split,
            checkpoint_path=checkpoint_path,
            checkpoint_key="target_encoder",
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
            used = next_batch_size
        else:
            caches[split], used = load_or_extract_adaptive(
                split=split,
                cache_path=cache_root / f"{split}.pt",
                metadata=metadata,
                dataset=datasets[split],
                encoder=encoder,
                device=device,
                initial_batch_size=next_batch_size,
                num_workers=num_workers,
                use_bfloat16=model_info["use_bfloat16"],
                recompute=recompute_features,
                pooling=pooling,
            )
        used_batch_sizes[split] = used
        next_batch_size = min(next_batch_size, used)

    info = {
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": features._sha256_file(checkpoint_path),
        "model_name": model_info["model_name"],
        "feature_dim": int(caches["train"]["features"].shape[1]),
        "pooling": pooling,
        "dataset_root": str(dataset_root),
        "dataset_index_sha256": features._sha256_file(dataset_root / "index.json"),
        "stats_root": str(stats_root),
        "stats_mean_sha256": features._sha256_file(stats_root / "mean.npy"),
        "stats_std_sha256": features._sha256_file(stats_root / "std.npy"),
        "split_counts": {split: len(datasets[split]) for split in features.SPLITS},
        "feature_batch_sizes": used_batch_sizes,
    }
    del encoder, datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return caches, class_names, info


def _run_signature(
    checkpoint_info: dict[str, Any], *, lr: float, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "dataset_index_sha256": checkpoint_info["dataset_index_sha256"],
        "stats_mean_sha256": checkpoint_info["stats_mean_sha256"],
        "stats_std_sha256": checkpoint_info["stats_std_sha256"],
        "lr": lr,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "pooling": checkpoint_info.get("pooling", features.GLOBAL_MEAN_POOLING),
    }


def run_one_probe(
    *,
    caches: dict[str, dict[str, Any]],
    class_names: list[str],
    checkpoint_info: dict[str, Any],
    lr: float,
    seed: int,
    run_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    signature = _run_signature(checkpoint_info, lr=lr, seed=seed, args=args)
    summary_path = run_root / "summary.json"
    if summary_path.is_file() and not args.overwrite_runs:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete" and summary.get("signature") == signature:
            head_filename = summary.get("head_filename", "linear-probe-best.pth.tar")
            if (run_root / "metrics.csv").is_file() and (
                run_root / head_filename
            ).is_file():
                return summary

    run_root.mkdir(parents=True, exist_ok=True)
    summary = train_linear_probe(
        caches,
        output=run_root,
        checkpoint_path=Path(checkpoint_info["checkpoint"]),
        checkpoint_key="target_encoder",
        class_names=class_names,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=seed,
        run_args=signature,
    )
    summary.update(
        {
            "status": "complete",
            "run_name": checkpoint_info["run_name"],
            "model_name": checkpoint_info["model_name"],
            "checkpoint": checkpoint_info["checkpoint"],
            "lr": lr,
            "seed": seed,
            "signature": signature,
        }
    )
    _atomic_json(summary, summary_path)
    return summary


def _result_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": summary["run_name"],
        "model_name": summary["model_name"],
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["signature"]["checkpoint_sha256"],
        "lr": summary["lr"],
        "seed": summary["seed"],
        "best_epoch": summary["best_epoch"],
    }
    for split in ("best_val", "test"):
        prefix = "val" if split == "best_val" else "test"
        for metric in METRIC_NAMES:
            values = summary.get(split)
            row[f"{prefix}_{metric}"] = (
                float("nan") if values is None else values[metric]
            )
    return row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run_name"]), float(row["lr"]))].append(row)
    aggregates = []
    for (run_name, lr), group in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "run_name": run_name,
            "model_name": group[0]["model_name"],
            "lr": lr,
            "num_seeds": len(group),
        }
        numeric_fields = ["best_epoch"] + [
            f"{split}_{metric}"
            for split in ("val", "test")
            for metric in METRIC_NAMES
        ]
        for field in numeric_fields:
            values = [float(row[field]) for row in group]
            aggregate[f"{field}_mean"] = mean(values)
            aggregate[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        aggregates.append(aggregate)
    return aggregates


def merge_epoch_metrics(
    findings_root: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = []
    for result in rows:
        run_root = (
            findings_root
            / "runs"
            / str(result["run_name"])
            / f"lr-{lr_slug(float(result['lr']))}"
            / f"seed-{int(result['seed'])}"
        )
        with (run_root / "metrics.csv").open(encoding="utf-8", newline="") as file:
            for epoch_row in csv.DictReader(file):
                merged.append(
                    {
                        "run_name": result["run_name"],
                        "model_name": result["model_name"],
                        "lr": result["lr"],
                        "seed": result["seed"],
                        **epoch_row,
                    }
                )
    return merged


def _plot_heatmap(
    aggregates: list[dict[str, Any]],
    run_names: list[str],
    lrs: list[float],
    metric: str,
    output: Path,
    title: str,
) -> None:
    lookup = {(row["run_name"], float(row["lr"])): row for row in aggregates}
    values = np.array(
        [
            [float(lookup[(run, lr)][metric]) * 100.0 for lr in lrs]
            for run in run_names
        ]
    )
    figure, axis = plt.subplots(figsize=(12, max(7, len(run_names) * 0.48)))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(lrs)))
    axis.set_xticklabels([f"{lr:g}" for lr in lrs])
    axis.set_yticks(range(len(run_names)))
    axis.set_yticklabels(run_names)
    axis.set_xlabel("Initial learning rate")
    axis.set_title(title)
    for row_index in range(len(run_names)):
        for column_index in range(len(lrs)):
            color = "white" if values[row_index, column_index] < values.mean() else "black"
            axis.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    figure.colorbar(image, ax=axis, label="Top-1 accuracy (%)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_architecture_sort_key(run_name: str) -> tuple[int, int, str]:
    """Order larger encoders first, then larger predictors within an encoder."""
    architecture = run_name.split("-bs.", 1)[0]
    architecture = architecture.removeprefix("mot_").split("_1d", 1)[0]
    names = architecture.split("-", 1)
    encoder_name = names[0]
    predictor_name = names[1] if len(names) == 2 else encoder_name
    return (
        -MODEL_SIZE_RANK.get(encoder_name, -1),
        -MODEL_SIZE_RANK.get(predictor_name, -1),
        run_name,
    )


def create_plots(
    findings_root: Path,
    aggregates: list[dict[str, Any]],
    run_names: list[str],
    lrs: list[float],
    *,
    validation_used: bool = True,
) -> None:
    if validation_used:
        _plot_heatmap(
            aggregates,
            run_names,
            lrs,
            "val_top1_accuracy_mean",
            findings_root / "validation-top1-heatmap.png",
            "Validation top-1 by checkpoint and learning rate",
        )
    _plot_heatmap(
        aggregates,
        run_names,
        lrs,
        "test_top1_accuracy_mean",
        findings_root / "test-top1-heatmap.png",
        "Test top-1 by checkpoint and learning rate",
    )
    plot_root = findings_root / "plots"
    plot_root.mkdir(exist_ok=True)
    for run_name in run_names:
        group = sorted(
            (row for row in aggregates if row["run_name"] == run_name),
            key=lambda row: float(row["lr"]),
        )
        figure, axis = plt.subplots(figsize=(7, 4.5))
        multiple_seeds = any(int(row["num_seeds"]) > 1 for row in group)
        splits = [("test", "tab:orange")]
        if validation_used:
            splits.insert(0, ("val", "tab:blue"))
        for split, color in splits:
            x_values = [float(row["lr"]) for row in group]
            y_values = [
                float(row[f"{split}_top1_accuracy_mean"]) * 100 for row in group
            ]
            if multiple_seeds:
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=[
                        float(row[f"{split}_top1_accuracy_std"]) * 100
                        for row in group
                    ],
                    marker="o",
                    capsize=3,
                    label=split,
                    color=color,
                )
            else:
                axis.plot(x_values, y_values, marker="o", label=split, color=color)
        axis.set_xscale("log")
        axis.set_xlabel("Initial learning rate")
        axis.set_ylabel("Top-1 accuracy (%)")
        axis.set_title(run_name)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_root / f"{run_name}.png", dpi=180)
        plt.close(figure)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}"


def _format_aggregate_metric(
    row: dict[str, Any], field: str, *, multiple_seeds: bool
) -> str:
    value = _percent(float(row[field + "_mean"]))
    if not multiple_seeds:
        return value
    return f"{value} +/- {_percent(float(row[field + '_std']))}"


def write_readme(
    findings_root: Path,
    checkpoint_infos: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    lrs: list[float],
    seeds: list[int],
    args: argparse.Namespace,
    *,
    validation_used: bool = True,
) -> None:
    lookup = {(row["run_name"], float(row["lr"])): row for row in aggregates}
    multiple_seeds = len(seeds) > 1
    lines = [
        "# 100STYLE MotionJEPA Linear-Probe LR Sweep",
        "",
        "## Settings",
        "",
        f"- Checkpoints: {len(checkpoint_infos)}",
        f"- Dataset: `{args.dataset_root}`",
        f"- Learning rates: {', '.join(f'`{lr:g}`' for lr in lrs)}",
        f"- Seeds: {', '.join(map(str, seeds))}",
        f"- Epochs: {args.epochs}",
        f"- Optimizer: SGD, momentum {args.momentum}, weight decay {args.weight_decay}",
        "- Schedule: cosine decay; loss: ordinary cross entropy",
        f"- Pooling: `{getattr(args, 'pooling', features.GLOBAL_MEAN_POOLING)}`",
        (
            "- Downstream comparisons select LR by validation top-1"
            if validation_used
            else "- No validation split: FW test results are diagnostic only and must not be used to select LR"
        ),
        "",
        "## Overall comparison",
        "",
        "![Test top-1 heatmap](test-top1-heatmap.png)",
        "",
        (
            f"Values are means over {len(seeds)} seeds and are reported in percent."
            if multiple_seeds
            else f"Values are from seed {seeds[0]} and are reported in percent."
        ),
        "",
        "## Checkpoints",
        "",
        "| Run | Encoder | Feature dim | SHA256 |",
        "|---|---:|---:|---|",
    ]
    if validation_used:
        insertion = lines.index("![Test top-1 heatmap](test-top1-heatmap.png)")
        lines[insertion:insertion] = [
            "![Validation top-1 heatmap](validation-top1-heatmap.png)",
            "",
        ]
    for info in checkpoint_infos:
        lines.append(
            f"| `{info['run_name']}` | `{info['model_name']}` | "
            f"{info['feature_dim']} | `{info['checkpoint_sha256']}` |"
        )
    lines.extend(["", "## Results by learning rate", ""])
    for info in checkpoint_infos:
        run_name = info["run_name"]
        lines.extend(
            [
                f"### {run_name}",
                "",
                f"![{run_name} LR curve](plots/{run_name}.png)",
                "",
                (
                    "| LR | Val top-1 | Val macro | Val top-5 | Test top-1 | Test macro | Test top-5 |"
                    if validation_used
                    else "| LR | Test top-1 | Test macro | Test top-5 |"
                ),
                (
                    "|---:|---:|---:|---:|---:|---:|---:|"
                    if validation_used
                    else "|---:|---:|---:|---:|"
                ),
            ]
        )
        for lr in lrs:
            row = lookup[(run_name, lr)]
            values = []
            fields = [
                "test_top1_accuracy",
                "test_macro_accuracy",
                "test_top5_accuracy",
            ]
            if validation_used:
                fields[0:0] = [
                    "val_top1_accuracy",
                    "val_macro_accuracy",
                    "val_top5_accuracy",
                ]
            for field in fields:
                values.append(
                    _format_aggregate_metric(
                        row, field, multiple_seeds=multiple_seeds
                    )
                )
            lines.append(f"| {lr:g} | " + " | ".join(values) + " |")
        test_values = [float(lookup[(run_name, lr)]["test_top1_accuracy_mean"]) for lr in lrs]
        span = (
            "Across the evaluated range, test top-1 spans "
            f"{(max(test_values) - min(test_values)) * 100:.2f} percentage points."
        )
        if validation_used:
            val_values = [
                float(lookup[(run_name, lr)]["val_top1_accuracy_mean"]) for lr in lrs
            ]
            span = (
                "Across the evaluated range, validation top-1 spans "
                f"{(max(val_values) - min(val_values)) * 100:.2f} percentage points "
                "and test top-1 spans "
                f"{(max(test_values) - min(test_values)) * 100:.2f} percentage points."
            )
        lines.extend(["", span, ""])
    lines.extend(
        [
            "## Raw artifacts",
            "",
            "- [Per-seed results](sweep-results.csv)",
            "- [Per-LR aggregates](aggregate-results.csv)",
            "- [Per-epoch metrics](epoch-metrics.csv)",
            "- [Experiment configuration](sweep-config.json)",
            "- Best linear heads and individual metrics are stored under `runs/<run>/lr-*/seed-*`",
            "",
        ]
    )
    (findings_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_reports(
    findings_root: Path,
    summaries: list[dict[str, Any]],
    checkpoint_infos: list[dict[str, Any]],
    lrs: list[float],
    seeds: list[int],
    args: argparse.Namespace,
    *,
    validation_used: bool = True,
) -> None:
    checkpoint_infos = sorted(
        checkpoint_infos,
        key=lambda info: run_architecture_sort_key(str(info["run_name"])),
    )
    rows = [_result_row(summary) for summary in summaries]
    result_fields = list(rows[0])
    _atomic_csv(findings_root / "sweep-results.csv", rows, result_fields)
    aggregates = aggregate_rows(rows)
    _atomic_csv(
        findings_root / "aggregate-results.csv",
        aggregates,
        list(aggregates[0]),
    )
    epoch_rows = merge_epoch_metrics(findings_root, rows)
    _atomic_csv(
        findings_root / "epoch-metrics.csv",
        epoch_rows,
        list(epoch_rows[0]),
    )
    run_names = [info["run_name"] for info in checkpoint_infos]
    create_plots(
        findings_root,
        aggregates,
        run_names,
        lrs,
        validation_used=validation_used,
    )
    write_readme(
        findings_root,
        checkpoint_infos,
        aggregates,
        lrs,
        seeds,
        args,
        validation_used=validation_used,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    validation_used = has_validation_dataset(dataset_root)
    diagnostic_no_validation = bool(
        getattr(args, "diagnostic_no_validation", False)
    )
    if not validation_used and not diagnostic_no_validation:
        require_validation_dataset(dataset_root)
    findings_root = Path(args.findings_root).expanduser().resolve()
    findings_root.mkdir(parents=True, exist_ok=True)
    args.dataset_root = dataset_root
    lrs = sorted(set(float(value) for value in args.lrs))
    seeds = sorted(set(int(value) for value in args.seeds))
    if not lrs or any(not math.isfinite(lr) or lr <= 0 for lr in lrs):
        raise ValueError("Learning rates must be finite and positive")
    if args.epochs <= 0 or args.batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("Epoch and batch sizes must be positive")
    checkpoints = resolve_checkpoints(
        output_root, getattr(args, "checkpoints", None)
    )
    device = features.resolve_device(args.device)
    checkpoint_infos: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for run_name, checkpoint_path in checkpoints:
        print(f"\n=== {run_name}: feature preparation ===", flush=True)
        try:
            caches, class_names, info = prepare_checkpoint_features(
                run_name=run_name,
                checkpoint_path=checkpoint_path,
                dataset_root=dataset_root,
                device=device,
                feature_batch_size=args.feature_batch_size,
                num_workers=args.num_workers,
                recompute_features=args.recompute_features,
                pooling=getattr(args, "pooling", features.GLOBAL_MEAN_POOLING),
                cache_root=findings_root / "feature-cache" / run_name,
            )
            checkpoint_infos.append(info)
            for lr in lrs:
                for seed in seeds:
                    print(f"{run_name}: lr={lr:g}, seed={seed}", flush=True)
                    run_root = (
                        findings_root
                        / "runs"
                        / run_name
                        / f"lr-{lr_slug(lr)}"
                        / f"seed-{seed}"
                    )
                    summaries.append(
                        run_one_probe(
                            caches=caches,
                            class_names=class_names,
                            checkpoint_info=info,
                            lr=lr,
                            seed=seed,
                            run_root=run_root,
                            device=device,
                            args=args,
                        )
                    )
            del caches
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            failure = {
                "run_name": run_name,
                "checkpoint": str(checkpoint_path),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            with (findings_root / "errors.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(failure, ensure_ascii=False) + "\n")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    config = {
        "format_version": 1,
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "findings_root": str(findings_root),
        "lrs": lrs,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "feature_batch_size": args.feature_batch_size,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "pooling": getattr(args, "pooling", features.GLOBAL_MEAN_POOLING),
        "validation_used": validation_used,
        "diagnostic_no_validation": diagnostic_no_validation,
        "checkpoints": checkpoint_infos,
        "failures": failures,
    }
    _atomic_json(config, findings_root / "sweep-config.json")
    if summaries:
        _write_reports(
            findings_root,
            summaries,
            checkpoint_infos,
            lrs,
            seeds,
            args,
            validation_used=validation_used,
        )
    expected = len(checkpoints) * len(lrs) * len(seeds)
    result = {
        "num_checkpoints": len(checkpoints),
        "num_completed": len(summaries),
        "num_expected": expected,
        "num_failures": len(failures),
        "findings_root": str(findings_root),
    }
    if failures or len(summaries) != expected:
        raise RuntimeError(f"Sweep incomplete: {result}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep linear probes over latest checkpoints")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit immutable checkpoints; default discovers direct-child latest files",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/100style-soma77-processed",
    )
    parser.add_argument(
        "--findings-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "findings/000-100style-classification/linear-probe"
        ),
    )
    parser.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--pooling",
        choices=(features.GLOBAL_MEAN_POOLING, features.SPATIAL_FLATTEN_POOLING),
        default=features.GLOBAL_MEAN_POOLING,
    )
    parser.add_argument(
        "--diagnostic-no-validation",
        action="store_true",
        help="Allow an LR sensitivity diagnostic without selecting LR on test metrics",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--overwrite-runs", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
