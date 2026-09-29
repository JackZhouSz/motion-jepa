"""Offline attentive-probe LR sweeps and comparison with frozen linear probes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import train_classifier as classifier
from .features import _atomic_json_save, _sha256_file, resolve_device


def verify_linear_baselines(root: Path, checkpoint: Path, epoch: int,
                            dataset: str, prepared) -> list[dict]:
    """Reject comparisons that use different checkpoints, data or statistics."""
    candidates = [json.loads(path.read_text()) for path in sorted((root / "results").glob("*.json"))]
    candidates = [r for r in candidates if r["dataset"] == dataset and r["encoder_epoch"] == epoch]
    if not candidates:
        raise ValueError(f"No linear baseline for {dataset} epoch {epoch}")
    source = prepared.jepa_source
    for result in candidates:
        signature = result["signature"]
        expected = {
            "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_key": "target_encoder",
            "stats_mean_sha256": source["stats_mean_sha256"],
            "stats_std_sha256": source["stats_std_sha256"],
        }
        if any(signature.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Linear baseline checkpoint/statistics mismatch: {dataset}, epoch {epoch}")
        # The sweep supplies the actual input dataset path independently of old aliases.
        current_root = Path(source["comparison_dataset_root"])
        for relative, digest in signature["dataset_files"].items():
            path = current_root / relative
            if not path.is_file() or _sha256_file(path) != digest:
                raise ValueError(f"Linear baseline dataset mismatch: {path}")
        if signature["dataset_files"].get("index.json") != _sha256_file(current_root / "index.json"):
            raise ValueError("Linear baseline lacks a matching dataset index hash")
        if result["protocol"]["pooling"] != "valid_token_mean":
            raise ValueError("Expected a mean-pooled linear baseline")
    return candidates


def write_report(output: Path, manifest: dict) -> None:
    results = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("**/result.json"))]
    rows = []
    for r in results:
        s = r["summary"]
        rows.append(dict(dataset=r["dataset"], encoder_epoch=r["encoder_epoch"], lr=r["lr"],
                         best_head_epoch=s["best_epoch"], num_parameters=s["num_parameters"], **s["best_val"]))
        metrics_path = output / r["metrics_path"]
        curve_path = metrics_path.with_name("learning-curves.png")
        if not curve_path.exists():
            with metrics_path.open() as file:
                history = list(csv.DictReader(file))
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
            epochs = [int(h["epoch"]) for h in history]
            for ax, metric, label in zip(axes, ("mean_average_precision", "loss"), ("mAP", "BCE loss")):
                for split in ("train", "val"):
                    ax.plot(epochs, [float(h[f"{split}_{metric}"]) for h in history], label=split)
                ax.axvline(s["best_epoch"], color="gray", linestyle=":", label="Selected head epoch")
                ax.set(xlabel="Head epoch", ylabel=label)
                ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.suptitle(f"{r['dataset']} | encoder epoch {r['encoder_epoch']} | LR {r['lr']:g}")
            fig.savefig(curve_path, dpi=160)
            plt.close(fig)
    rows.sort(key=lambda r: (r["dataset"], r["encoder_epoch"], r["lr"]))
    if not rows:
        return
    with (output / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = []
    accuracy = []
    for name in ("babel-60", "babel-120"):
        for epoch in sorted({r["encoder_epoch"] for r in results}):
            group = [r for r in results if r["dataset"] == name and r["encoder_epoch"] == epoch]
            if not group:
                continue
            chosen = max(group, key=lambda r: r["summary"]["best_val"]["mean_average_precision"])
            linear = max(chosen["linear_baselines"], key=lambda r: r["summary"]["best_val"]["mean_average_precision"])
            item = dict(dataset=name, encoder_epoch=epoch, attentive_lr=chosen["lr"],
                        attentive_mAP=chosen["summary"]["best_val"]["mean_average_precision"],
                        linear_lr=linear["lr"], linear_mAP=linear["summary"]["best_val"]["mean_average_precision"],
                        best_head_epoch=chosen["summary"]["best_epoch"],
                        top1_hit=chosen["summary"]["best_val"]["top1_hit"],
                        top1_label_row_accuracy=chosen["summary"]["best_val"]["top1_label_row_accuracy"],
                        lr_boundary=chosen["lr"] in (min(manifest["lrs"]), max(manifest["lrs"])),
                        last_head_epoch=chosen["summary"]["best_epoch"] == manifest["epochs"])
            best.append(item)
            attentive_candidates = []
            for r in group:
                with (output / r["metrics_path"]).open() as f:
                    for row in csv.DictReader(f):
                        attentive_candidates.append((float(row["val_top1_label_row_accuracy"]), r["lr"], int(row["epoch"])))
            linear_candidates = [(h["top1_label_row_accuracy"], r["lr"], h["head_epoch"])
                                 for r in chosen["linear_baselines"] for h in r["history"]]
            a, alr, ae = max(attentive_candidates)
            l, llr, le = max(linear_candidates)
            accuracy.append(dict(dataset=name, encoder_epoch=epoch, attentive_accuracy=a,
                                 attentive_lr=alr, attentive_head_epoch=ae, linear_accuracy=l,
                                 linear_lr=llr, linear_head_epoch=le))
    _atomic_json_save({"mAP_selected": best, "accuracy_selected": accuracy}, output / "comparison.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, name in enumerate(("babel-60", "babel-120")):
        selected = [r for r in best if r["dataset"] == name]
        ax = axes[0, col]
        for epoch in sorted({r["encoder_epoch"] for r in rows}):
            group = [r for r in rows if r["dataset"] == name and r["encoder_epoch"] == epoch]
            if group:
                ax.plot([r["lr"] for r in group], [100*r["mean_average_precision"] for r in group], "o-", label=f"Epoch {epoch}")
        ax.set(xscale="log", xlabel="Attentive-head initial LR", ylabel="Best validation mAP (%)", title=name.upper())
        ax.legend(); ax.grid(alpha=.2)
        ax = axes[1, col]
        for key, label in (("linear_mAP", "Mean + linear (best LR)"), ("attentive_mAP", "Attentive (best LR)")):
            ax.plot([r["encoder_epoch"] for r in selected], [100*r[key] for r in selected], "o-", label=label)
        ax.set(xlabel="Encoder epoch", ylabel="Best validation mAP (%)")
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(f"Frozen EMA encoder: linear vs attentive probing (seed {manifest['seed']})")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# Frozen-checkpoint attentive probing", "",
             f"Completed {len(results)}/{manifest['expected_runs']} runs; seed {manifest['seed']}.", "",
             f"One learned query, 6-head cross-attention, residual MLP (ratio 4, GELU), LayerNorm and classifier. No extra temporal self-attention, positional embeddings or dropout. Frozen EMA encoder; BONES-SEED statistics; BF16 token cache. Head training uses FP32 parameters with BF16 autocast on CUDA, unweighted BCE, AdamW (betas 0.9/0.999, WD 0.05), batch 256, {manifest['epochs']} epochs, 5-epoch warmup then cosine to 1e-6, gradient clipping 1.0.", "",
             f"Attentive LR grid: {manifest['lrs']}. Head parameter count(s): {sorted({r['num_parameters'] for r in rows})}. The existing linear sweep used its own recorded LR grid and SGD protocol.", "",
             "Each LR has the same seed and sample order. Select the best validation mAP over head epochs, then across the agreed LR grid. No test evaluation. Linear results are reused only after checkpoint/data/statistics hash verification. Different head capacity, normalization and optimizer mean this is not an isolated causal test of attention alone. Validation selection and a single seed do not provide independent test estimates or confidence intervals.", "",
             "## Best validation mAP", "",
             "| Dataset | Encoder epoch | Linear LR | Linear mAP % | Attentive LR | Attentive mAP % | Head epoch | Top-1 hit % | Label-row accuracy % | Boundary flags |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in best:
        flags = ", ".join(label for key, label in (("lr_boundary", "LR boundary"), ("last_head_epoch", "last head epoch")) if r[key]) or "none"
        lines.append(f"| {r['dataset']} | {r['encoder_epoch']} | {r['linear_lr']:g} | {100*r['linear_mAP']:.3f} | {r['attentive_lr']:g} | {100*r['attentive_mAP']:.3f} | {r['best_head_epoch']} | {100*r['top1_hit']:.3f} | {100*r['top1_label_row_accuracy']:.3f} | {flags} |")
    lines += ["", "## Epoch trend", ""]
    for name in ("babel-60", "babel-120"):
        start = next((r for r in best if r["dataset"] == name and r["encoder_epoch"] == 100), None)
        end = next((r for r in best if r["dataset"] == name and r["encoder_epoch"] == 300), None)
        if start and end:
            ld = 100*(end["linear_mAP"]-start["linear_mAP"])
            ad = 100*(end["attentive_mAP"]-start["attentive_mAP"])
            lines.append(f"- {name}, epoch 100 -> 300: linear {ld:+.3f} pp; attentive {ad:+.3f} pp. Difference in changes: {ad-ld:+.3f} pp.")
    lines += ["", "Absolute gains and changes in the epoch trend must be interpreted separately. An improving attentive trend would support a readout contribution to the earlier decline, but would not prove that all useful information is preserved. LR boundary/last-epoch optima remain limited by the agreed search/training budget; the sweep does not expand automatically.", "",
              "## Accuracy selected independently", "", "Both LR and head epoch below maximize validation label-row accuracy, not mAP.", "",
              "| Dataset | Encoder epoch | Linear accuracy % | Attentive accuracy % | Attentive LR | Head epoch |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in accuracy:
        lines.append(f"| {r['dataset']} | {r['encoder_epoch']} | {100*r['linear_accuracy']:.3f} | {100*r['attentive_accuracy']:.3f} | {r['attentive_lr']:g} | {r['attentive_head_epoch']} |")
    lines += ["", "## Artifacts and reproduction", "",
              "- `manifest.json`: exact inputs, hashes, grid and command.",
              "- `runs/`: metrics.csv, learning-curves.png, best/latest heads, model configuration and per-run summaries.",
              "- `features/`: checkpoint/dataset-specific BF16 token caches with valid lengths.",
              "- `results.csv`, `comparison.json`, `comparison.png`: all results and selected comparisons.", "",
              "```bash", manifest["command"], "```", ""]
    (output / "README.md").write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--babel-60-root", type=Path, required=True)
    parser.add_argument("--babel-120-root", type=Path, required=True)
    parser.add_argument("--linear-baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers for in-memory head training")
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recompute-features", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.epochs <= 5 or not args.lrs or any(not math.isfinite(lr) or lr < 1e-6 for lr in args.lrs):
        raise ValueError("Require epochs > 5 and finite LRs >= final LR 1e-6")
    if len(set(args.lrs)) != len(args.lrs) or min(args.num_workers, args.feature_workers) < 0:
        raise ValueError("Duplicate LRs or negative workers")
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    output = args.output_root.resolve()
    checkpoints = []
    for path in args.checkpoints:
        path = path.resolve()
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoints.append(dict(path=str(path), epoch=int(state["next_epoch"]), sha256=_sha256_file(path)))
        del state
    if len({c["epoch"] for c in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoint epochs must be unique")
    datasets = {name: str(getattr(args, name.replace("-", "_") + "_root").resolve()) for name in ("babel-60", "babel-120")}
    manifest = dict(version=1, checkpoints=checkpoints, datasets=datasets, lrs=args.lrs,
                    epochs=args.epochs, seed=args.seed, expected_runs=len(checkpoints)*2*len(args.lrs),
                    linear_baseline_root=str(args.linear_baseline_root.resolve()),
                    command="python -m experiment.linear_probe.attentive_sweep " + shlex.join(sys.argv[1:]))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if {k:v for k,v in old.items() if k != "command"} != {k:v for k,v in manifest.items() if k != "command"}:
            raise ValueError("Sweep manifest differs; use a separate output directory")
        manifest = old
    else:
        _atomic_json_save(manifest, manifest_path)
    for checkpoint in checkpoints:
        epoch = checkpoint["epoch"]
        for name, root in datasets.items():
            base = classifier.build_parser().parse_args([])
            base.model = "attentive"
            base.input_source = "jepa"
            base.jepa_checkpoint = Path(checkpoint["path"])
            base.dataset_root = Path(root)
            base.device = str(device)
            base.num_workers = args.feature_workers
            base.feature_cache_root = output / "features" / f"ep{epoch}" / name
            base.recompute_features = args.recompute_features
            base.feature_cache_root.mkdir(parents=True, exist_ok=True)
            print(f"PREPARE encoder_epoch={epoch} dataset={name}", flush=True)
            prepared = classifier._prepare_input(base, device=device)
            if prepared.task != "multilabel" or prepared.dataset_name != name or not all(len(prepared.datasets[s]) for s in ("train", "val")):
                raise ValueError("Expected matching BABEL data with nonempty train and val")
            if len(prepared.datasets["test"]):
                raise ValueError("This comparison expects BABEL's empty test split")
            prepared.jepa_source["comparison_dataset_root"] = root
            baselines = verify_linear_baselines(args.linear_baseline_root, base.jepa_checkpoint, epoch, name, prepared)
            base.num_workers = args.num_workers
            base.epochs = args.epochs
            base.seed = args.seed
            base.resume = args.resume
            for lr in args.lrs:
                base.lr = lr
                run_root = output / "runs" / f"ep{epoch}" / name / f"lr-{lr:g}"
                print(f"FIT encoder_epoch={epoch} dataset={name} lr={lr:g}", flush=True)
                summary = classifier.run_model(base, "attentive", prepared=prepared, output_root=run_root, device=device)
                directory = run_root / "attentive" / f"seed-{args.seed}"
                _atomic_json_save(dict(dataset=name, encoder_epoch=epoch, lr=lr, summary=summary,
                                       metrics_path=str((directory / "metrics.csv").relative_to(output)),
                                       linear_baselines=baselines), directory / "result.json")
                write_report(output, manifest)
                print(f"RESULT encoder_epoch={epoch} dataset={name} lr={lr:g} mAP={summary['best_val']['mean_average_precision']:.6f}", flush=True)
            del prepared
    print("SWEEP_COMPLETE", flush=True)
    return json.loads((output / "comparison.json").read_text())


if __name__ == "__main__":
    run(build_parser().parse_args())
