"""Build the unified 100STYLE probe and classifier findings report."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINDINGS_ROOT = PROJECT_ROOT / "findings/000-100style-classification"
FIXED_LR_COMPARISON = 0.3
METRICS = ("loss", "top1_accuracy", "macro_accuracy", "top5_accuracy")
MODEL_SIZE_RANK = {
    name: rank
    for rank, name in enumerate(("tiny", "small", "base", "large", "huge", "giant"))
}
SELECTED_FIELDS = (
    "kind",
    "name",
    "model_name",
    "selected_lr",
    "seed",
    "best_epoch",
    "val_loss",
    "val_top1_accuracy",
    "val_macro_accuracy",
    "val_top5_accuracy",
    "test_loss",
    "test_top1_accuracy",
    "test_macro_accuracy",
    "test_top5_accuracy",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required report input does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Report input CSV is empty: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required report input does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Report input JSON must be an object: {path}")
    return value


def select_probe_rows(
    aggregates: list[dict[str, str]], *, seed: int | str = 42
) -> list[dict[str, Any]]:
    """Select one LR per checkpoint by validation top-1, then lower-LR tie break."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in aggregates:
        grouped[row["run_name"]].append(row)
    selected = []
    for run_name, rows in grouped.items():
        best = max(
            rows,
            key=lambda row: (
                float(row["val_top1_accuracy_mean"]),
                -float(row["lr"]),
            ),
        )
        item: dict[str, Any] = {
            "kind": "linear_probe",
            "name": run_name,
            "model_name": best["model_name"],
            "selected_lr": float(best["lr"]),
            "seed": seed,
            "best_epoch": float(best["best_epoch_mean"]),
        }
        for split in ("val", "test"):
            for metric in METRICS:
                item[f"{split}_{metric}"] = float(
                    best[f"{split}_{metric}_mean"]
                )
        selected.append(item)
    return sorted(selected, key=_probe_architecture_sort_key)


def _probe_architecture_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    """Order larger encoders first, then larger predictors within an encoder."""
    run_name = str(row["name"])
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


def load_classifier_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = results.get("summaries")
    if not isinstance(summaries, dict) or set(summaries) != {"cnn", "transformer"}:
        raise ValueError("Classifier results must contain cnn and transformer summaries")
    seed = int(results["seed"])
    rows = []
    for model_name in ("transformer", "cnn"):
        summary = summaries[model_name]
        row: dict[str, Any] = {
            "kind": "raw_classifier",
            "name": model_name,
            "model_name": summary["signature"]["architecture"]["name"],
            "selected_lr": "",
            "seed": seed,
            "best_epoch": int(summary["best_epoch"]),
        }
        for split, source in (("val", summary["best_val"]), ("test", summary["test"])):
            for metric in METRICS:
                row[f"{split}_{metric}"] = float(source[metric])
        rows.append(row)
    return rows


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in SELECTED_FIELDS} for row in rows)
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}"


def _plot_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [
        "CLS Transformer" if row["name"] == "transformer" else
        "CNN" if row["name"] == "cnn" else str(row["name"])
        for row in rows
    ]
    values = [float(row["test_top1_accuracy"]) * 100 for row in rows]
    colors = ["tab:orange", "tab:blue"] + ["0.55"] * (len(rows) - 2)
    figure, axis = plt.subplots(figsize=(12, max(7, len(rows) * 0.47)))
    bars = axis.barh(range(len(rows)), values, color=colors)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Test top-1 accuracy (%)")
    axis.set_title("100STYLE classification with validation-selected models")
    axis.set_xlim(max(0.0, min(values) - 5.0), 100.0)
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            value + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def probe_rows_at_lr(
    aggregates: list[dict[str, str]], lr: float
) -> list[dict[str, Any]]:
    rows = [
        {
            "name": row["run_name"],
            "val_top1_accuracy": float(row["val_top1_accuracy_mean"]),
            "test_top1_accuracy": float(row["test_top1_accuracy_mean"]),
        }
        for row in aggregates
        if float(row["lr"]) == lr
    ]
    if not rows:
        raise ValueError(f"No linear-probe aggregate results found for LR {lr:g}")
    return sorted(rows, key=_probe_architecture_sort_key)


def _plot_fixed_lr_model_comparison(
    rows: list[dict[str, Any]], *, lr: float, output: Path
) -> None:
    labels = [str(row["name"]) for row in rows]
    validation = [float(row["val_top1_accuracy"]) * 100 for row in rows]
    test = [float(row["test_top1_accuracy"]) * 100 for row in rows]
    positions = list(range(len(rows)))
    height = 0.38
    figure, axis = plt.subplots(figsize=(12, max(7, len(rows) * 0.52)))
    axis.barh(
        [position - height / 2 for position in positions],
        validation,
        height=height,
        label="Validation",
        color="tab:blue",
    )
    axis.barh(
        [position + height / 2 for position in positions],
        test,
        height=height,
        label="Test",
        color="tab:orange",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Top-1 accuracy (%)")
    axis.set_title(f"Motion-JEPA linear probes at LR {lr:g}")
    axis.set_xlim(max(0.0, min(validation + test) - 5.0), 100.0)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _metric_text(row: dict[str, str], field: str, multiple_seeds: bool) -> str:
    value = _percent(float(row[field + "_mean"]))
    if not multiple_seeds:
        return value
    return f"{value} +/- {_percent(float(row[field + '_std']))}"


def _comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | Selection | Best epoch | Val top-1 | Val macro | Test top-1 | Test macro | Test top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["kind"] == "raw_classifier":
            method = "CLS Transformer" if row["name"] == "transformer" else "CNN"
            selection = "best val epoch"
        else:
            method = f"`{row['name']}`"
            selection = f"LR {float(row['selected_lr']):g}"
        lines.append(
            f"| {method} | {selection} | {float(row['best_epoch']):g} | "
            f"{_percent(float(row['val_top1_accuracy']))} | "
            f"{_percent(float(row['val_macro_accuracy']))} | "
            f"{_percent(float(row['test_top1_accuracy']))} | "
            f"{_percent(float(row['test_macro_accuracy']))} | "
            f"{_percent(float(row['test_top5_accuracy']))} |"
        )
    return lines


def build_readme(
    *,
    sweep_config: dict[str, Any],
    aggregates: list[dict[str, str]],
    classifier_results: dict[str, Any],
    classifier_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
) -> str:
    seeds = [int(value) for value in sweep_config["seeds"]]
    if seeds != [int(classifier_results["seed"])]:
        raise ValueError("Probe and classifier seeds do not match")
    if len(seeds) != 1:
        raise ValueError("The unified report currently requires one shared seed")
    seed = seeds[0]
    all_rows = classifier_rows + probe_rows
    transformer, cnn = classifier_rows
    best_probe = max(probe_rows, key=lambda row: float(row["val_top1_accuracy"]))
    transformer_gap = (
        float(transformer["test_top1_accuracy"])
        - float(best_probe["test_top1_accuracy"])
    ) * 100
    cnn_gap = (
        float(cnn["test_top1_accuracy"])
        - float(best_probe["test_top1_accuracy"])
    ) * 100
    lrs = [float(value) for value in sweep_config["lrs"]]
    lookup = {(row["run_name"], float(row["lr"])): row for row in aggregates}
    checkpoint_lookup = {
        item["run_name"]: item for item in sweep_config["checkpoints"]
    }

    lines = [
        "# 100STYLE Classification Evaluation",
        "",
        "This report compares frozen Motion-JEPA linear probes with supervised raw-motion CNN and CLS-Transformer classifiers. All values were regenerated from the current experiment artifacts; no historical metrics are reused.",
        "",
        "## Evaluation protocol",
        "",
        f"- Dataset: `dataset/100style-soma77-processed` ({sweep_config['checkpoints'][0]['split_counts']['train']:,} train, {sweep_config['checkpoints'][0]['split_counts']['val']:,} validation, {sweep_config['checkpoints'][0]['split_counts']['test']:,} test windows)",
        "- Split unit: non-overlapping 90-frame windows; source recordings may occur in multiple splits",
        f"- Seed: {seed} for every probe and classifier run",
        "- Selection: probe LR and training epoch are selected only by validation top-1; classifier epoch is selected only by validation top-1",
        "- Test metrics are descriptive single-seed results; no variance estimate is reported",
        "",
        "## Main comparison",
        "",
        "![Validation-selected test top-1 comparison](test-top1-comparison.png)",
        "",
        *_comparison_table(all_rows),
        "",
        "## Key findings",
        "",
        f"- The CLS Transformer is the strongest method at {_percent(float(transformer['test_top1_accuracy']))}% test top-1, followed by the CNN at {_percent(float(cnn['test_top1_accuracy']))}%.",
        f"- The strongest validation-selected linear probe is `{best_probe['name']}` with LR {float(best_probe['selected_lr']):g}, reaching {_percent(float(best_probe['test_top1_accuracy']))}% test top-1.",
        f"- The CNN and CLS Transformer exceed that probe by {cnn_gap:.2f} and {transformer_gap:.2f} percentage points, respectively.",
        "- These results compare temporal classifiers trained end to end with a linear head on globally mean-pooled frozen features; they do not isolate encoder quality from the pooling bottleneck.",
        "",
        "## Supervised raw-motion classifiers",
        "",
        "Both classifiers use normalized `[90,366]` motion windows, ordinary cross entropy, AdamW with LR `3e-4` and weight decay `0.05`, five warmup epochs, cosine decay over 100 epochs, batch size 256, BF16 autocast, and seed 42.",
        "",
        "![CNN and Transformer training curves](classifiers/training-curves.png)",
        "",
        *_comparison_table(classifier_rows),
        "",
        "Architecture details:",
        "",
        "- CNN: temporal ResNet with widths 256/384/512, two residual blocks per stage, and masked mean pooling.",
        "- CLS Transformer: dimension 256, eight blocks, eight heads, MLP ratio 4, learnable CLS token, and CLS-position embedding.",
        "",
        "## Linear-probe sweep",
        "",
        f"The sweep evaluates {len(probe_rows)} latest Motion-JEPA checkpoints at initial learning rates {', '.join(f'`{lr:g}`' for lr in lrs)}. Each biased linear head is trained for {int(sweep_config['epochs'])} epochs with SGD, momentum {float(sweep_config['momentum']):g}, zero weight decay, cosine decay, batch size {int(sweep_config['batch_size'])}, and seed {seed}. Frozen EMA target-encoder outputs are mean-pooled over valid tokens.",
        "",
        f"### Model comparison at LR {FIXED_LR_COMPARISON:g}",
        "",
        f"![Motion-JEPA model comparison at LR {FIXED_LR_COMPARISON:g}](linear-probe/lr-0p3-model-comparison.png)",
        "",
        "![Validation top-1 heatmap](linear-probe/validation-top1-heatmap.png)",
        "",
        "![Test top-1 heatmap](linear-probe/test-top1-heatmap.png)",
        "",
        "### Checkpoints",
        "",
        "| Run | Encoder | Feature dim | SHA256 |",
        "|---|---:|---:|---|",
    ]
    for row in probe_rows:
        info = checkpoint_lookup[row["name"]]
        lines.append(
            f"| `{row['name']}` | `{info['model_name']}` | {info['feature_dim']} | `{info['checkpoint_sha256']}` |"
        )

    lines.extend(["", "## Full learning-rate results", ""])
    multiple_seeds = len(seeds) > 1
    for selected in probe_rows:
        run_name = selected["name"]
        lines.extend(
            [
                f"### {run_name}",
                "",
                f"Validation-selected LR: `{float(selected['selected_lr']):g}`.",
                "",
                f"![{run_name} LR curve](linear-probe/plots/{run_name}.png)",
                "",
                "| LR | Val top-1 | Val macro | Val top-5 | Test top-1 | Test macro | Test top-5 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for lr in lrs:
            row = lookup[(run_name, lr)]
            values = [
                _metric_text(row, field, multiple_seeds)
                for field in (
                    "val_top1_accuracy",
                    "val_macro_accuracy",
                    "val_top5_accuracy",
                    "test_top1_accuracy",
                    "test_macro_accuracy",
                    "test_top5_accuracy",
                )
            ]
            lines.append(f"| {lr:g} | " + " | ".join(values) + " |")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- [Validation-selected results](selected-results.csv)",
            "- [Probe per-run results](linear-probe/sweep-results.csv)",
            "- [Probe per-LR aggregates](linear-probe/aggregate-results.csv)",
            "- [Probe per-epoch metrics](linear-probe/epoch-metrics.csv)",
            "- [Probe configuration](linear-probe/sweep-config.json)",
            "- [CNN metrics](classifiers/cnn-metrics.csv)",
            "- [Transformer metrics](classifiers/transformer-metrics.csv)",
            "- [Classifier summaries](classifiers/results.json)",
            "",
        ]
    )
    return "\n".join(lines)


def validate_local_links(markdown_path: Path) -> list[Path]:
    text = markdown_path.read_text(encoding="utf-8")
    targets = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", text)
    missing = []
    for target in targets:
        if "://" in target or target.startswith("#"):
            continue
        path = markdown_path.parent / target
        if not path.exists():
            missing.append(path)
    return missing


def run(findings_root: Path) -> dict[str, Any]:
    findings_root = findings_root.expanduser().resolve()
    sweep_root = findings_root / "linear-probe"
    classifier_root = findings_root / "classifiers"
    sweep_config = _read_json(sweep_root / "sweep-config.json")
    checkpoints = sweep_config.get("checkpoints", [])
    if any(item.get("split_counts", {}).get("val") == 0 for item in checkpoints):
        raise ValueError(
            "Validation-selected reports are disabled for validation-free datasets"
        )
    aggregates = _read_csv(sweep_root / "aggregate-results.csv")
    classifier_results = _read_json(classifier_root / "results.json")
    seeds = [int(value) for value in sweep_config["seeds"]]
    probe_seed: int | str = seeds[0] if len(seeds) == 1 else "multiple"
    probe_rows = select_probe_rows(aggregates, seed=probe_seed)
    classifier_rows = load_classifier_rows(classifier_results)
    selected = classifier_rows + probe_rows
    _atomic_csv(findings_root / "selected-results.csv", selected)
    _plot_comparison(selected, findings_root / "test-top1-comparison.png")
    _plot_fixed_lr_model_comparison(
        probe_rows_at_lr(aggregates, FIXED_LR_COMPARISON),
        lr=FIXED_LR_COMPARISON,
        output=sweep_root / "lr-0p3-model-comparison.png",
    )
    readme = build_readme(
        sweep_config=sweep_config,
        aggregates=aggregates,
        classifier_results=classifier_results,
        classifier_rows=classifier_rows,
        probe_rows=probe_rows,
    )
    _atomic_text(findings_root / "README.md", readme)
    missing = [
        missing_path
        for markdown_path in findings_root.rglob("README.md")
        for missing_path in validate_local_links(markdown_path)
    ]
    if missing:
        raise FileNotFoundError(
            "Findings contain missing local links: "
            + ", ".join(str(path) for path in missing)
        )
    return {
        "findings_root": str(findings_root),
        "num_probes": len(probe_rows),
        "num_classifiers": len(classifier_rows),
        "best_probe": max(
            probe_rows, key=lambda row: float(row["val_top1_accuracy"])
        )["name"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the unified 100STYLE evaluation report"
    )
    parser.add_argument("--findings-root", type=Path, default=DEFAULT_FINDINGS_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args.findings_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
