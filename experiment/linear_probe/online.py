"""In-memory 100STYLE and BABEL linear probing during pretraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dataset import (
    MultiLabelIndex,
    SingleLabelIndex,
    build_classification_datasets,
    classification_dataset_kind,
)
from .features import (
    GLOBAL_MEAN_POOLING,
    PROJECT_ROOT,
    SPLITS,
    extract_features,
    resolve_pretraining_stats,
)
from .train_probe import train_linear_probe, train_multilabel_probe


class OnlineLinearProbe:
    """Evaluate an in-memory frozen encoder with a fixed linear-probe protocol."""

    def __init__(
        self,
        training_config: dict[str, Any],
        probe_config: dict[str, Any],
        *,
        device: torch.device,
    ) -> None:
        self.device = device
        self.epochs = int(probe_config.get("epochs", 50))
        self.feature_batch_size = int(probe_config.get("feature_batch_size", 256))
        self.batch_size = int(probe_config.get("batch_size", 256))
        self.num_workers = int(probe_config.get("num_workers", 8))
        self.learning_rate = float(probe_config.get("lr", 0.3))
        self.momentum = float(probe_config.get("momentum", 0.9))
        self.weight_decay = float(probe_config.get("weight_decay", 0.0))
        self.seed = int(probe_config.get("seed", 42))
        self.pooling = str(probe_config.get("pooling", GLOBAL_MEAN_POOLING))
        if min(self.epochs, self.feature_batch_size, self.batch_size) <= 0:
            raise ValueError("Linear-probe epochs and batch sizes must be positive")
        if self.num_workers < 0:
            raise ValueError("linear_probe.num_workers must be non-negative")

        dataset_root = Path(
            str(probe_config.get("dataset_root", "dataset/100style-soma77-processed"))
        ).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root
        self.dataset_root = dataset_root.resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Linear-probe dataset does not exist: {self.dataset_root}"
            )
        dataset_metadata = json.loads(
            (self.dataset_root / "meta.json").read_text(encoding="utf-8")
        )
        test_contents = dataset_metadata.get("test_contents", [])
        self.test_content = test_contents[0] if len(test_contents) == 1 else test_contents

        data_config = training_config["data"]
        meta_config = training_config["meta"]
        stats_root = resolve_pretraining_stats(training_config, None)
        self.datasets, label_index = build_classification_datasets(
            self.dataset_root,
            splits=SPLITS,
            num_frames=int(data_config["num_frames"]),
            fps=int(data_config["fps"]),
            motion_dim=int(data_config["motion_dim"]),
            stats_root=stats_root,
        )
        if not isinstance(label_index, SingleLabelIndex):
            raise ValueError("The single-label online probe requires single-label data")
        self.class_names = list(label_index.class_names)
        train_classes = set(self.datasets["train"].labels)
        expected_classes = set(range(len(self.class_names)))
        if train_classes != expected_classes:
            missing = [
                self.class_names[index]
                for index in sorted(expected_classes - train_classes)
            ]
            raise ValueError(
                f"Linear-probe training split is missing classes: {missing}"
            )
        self.use_bfloat16 = bool(meta_config.get("use_bfloat16", False))

    def evaluate(self, encoder: nn.Module) -> dict[str, Any]:
        if encoder.training:
            raise ValueError("Online linear probe requires an eval-mode encoder")
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("Online linear probe requires a frozen encoder")
        caches = {}
        feature_dim = int(encoder.embed_dim)
        if self.pooling != GLOBAL_MEAN_POOLING:
            feature_dim *= int(encoder.token_layout.token_num_joints)
        for split in SPLITS:
            if len(self.datasets[split]) == 0:
                caches[split] = {
                    "features": torch.empty((0, feature_dim), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.long),
                    "sample_ids": [],
                }
            else:
                caches[split] = extract_features(
                    encoder,
                    self.datasets[split],
                    device=self.device,
                    batch_size=self.feature_batch_size,
                    num_workers=self.num_workers,
                    use_bfloat16=self.use_bfloat16,
                    show_progress=False,
                    pooling=self.pooling,
                )
        summary = train_linear_probe(
            caches,
            output=None,
            checkpoint_path=None,
            checkpoint_key="target_encoder",
            class_names=self.class_names,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            seed=self.seed,
            run_args={
                "mode": "online",
                "dataset_root": str(self.dataset_root),
                "epochs": self.epochs,
                "feature_batch_size": self.feature_batch_size,
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "lr": self.learning_rate,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
                "pooling": self.pooling,
            },
        )
        if any(parameter.grad is not None for parameter in encoder.parameters()):
            raise RuntimeError("Online linear probe accumulated encoder gradients")
        summary["pooling"] = self.pooling
        summary["test_content"] = self.test_content
        return summary


class OnlineBabelProbes:
    """Evaluate BABEL-60 and BABEL-120 separately with frozen EMA features."""

    def __init__(
        self,
        training_config: dict[str, Any],
        probe_config: dict[str, Any],
        *,
        device: torch.device,
    ) -> None:
        self.device = device
        self.epochs = int(probe_config.get("epochs", 50))
        self.feature_batch_size = int(probe_config.get("feature_batch_size", 256))
        self.batch_size = int(probe_config.get("batch_size", 256))
        self.num_workers = int(probe_config.get("num_workers", 8))
        self.learning_rate = float(probe_config.get("lr", 0.3))
        self.momentum = float(probe_config.get("momentum", 0.9))
        self.weight_decay = float(probe_config.get("weight_decay", 0.0))
        self.seed = int(probe_config.get("seed", 42))
        self.pooling = str(probe_config.get("pooling", GLOBAL_MEAN_POOLING))
        if min(self.epochs, self.feature_batch_size, self.batch_size) <= 0:
            raise ValueError("Linear-probe epochs and batch sizes must be positive")
        if self.num_workers < 0:
            raise ValueError("linear_probe.num_workers must be non-negative")

        configured = probe_config.get("datasets")
        if not isinstance(configured, dict) or set(configured) != {"babel-60", "babel-120"}:
            raise ValueError("linear_probe.datasets must contain babel-60 and babel-120")
        self.dataset_roots: dict[str, Path] = {}
        self.datasets: dict[str, dict[str, Any]] = {}
        self.label_indices: dict[str, Any] = {}
        data_config = training_config["data"]
        stats_root = resolve_pretraining_stats(training_config, None)
        for name in ("babel-60", "babel-120"):
            root = Path(str(configured[name])).expanduser()
            if not root.is_absolute():
                root = PROJECT_ROOT / root
            root = root.resolve()
            task, dataset_name = classification_dataset_kind(root)
            if task != "multilabel" or dataset_name != name:
                raise ValueError(f"linear_probe.datasets[{name}] is not {name}: {root}")
            datasets, label_index = build_classification_datasets(
                root,
                splits=("train", "val"),
                num_frames=int(data_config["num_frames"]),
                fps=int(data_config["fps"]),
                motion_dim=int(data_config["motion_dim"]),
                stats_root=stats_root,
            )
            if not isinstance(label_index, MultiLabelIndex):
                raise ValueError(f"Online multi-label probe requires multi-label data: {root}")
            if not len(datasets["train"]) or not len(datasets["val"]):
                raise ValueError(f"BABEL online probing needs train and val data: {root}")
            self.dataset_roots[name] = root
            self.datasets[name] = datasets
            self.label_indices[name] = label_index
        self.use_bfloat16 = bool(training_config["meta"].get("use_bfloat16", False))

    def evaluate(self, encoder: nn.Module) -> dict[str, dict[str, Any]]:
        if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("Online BABEL probing requires a frozen eval-mode encoder")
        summaries = {}
        for name in ("babel-60", "babel-120"):
            caches = {
                split: extract_features(
                    encoder,
                    self.datasets[name][split],
                    device=self.device,
                    batch_size=self.feature_batch_size,
                    num_workers=self.num_workers,
                    use_bfloat16=self.use_bfloat16,
                    show_progress=False,
                    pooling=self.pooling,
                )
                for split in ("train", "val")
            }
            summary = train_multilabel_probe(
                caches,
                label_index=self.label_indices[name],
                device=self.device,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                seed=self.seed,
            )
            summary["dataset_root"] = str(self.dataset_roots[name])
            summary["pooling"] = self.pooling
            summaries[name] = summary
            del caches
        if any(parameter.grad is not None for parameter in encoder.parameters()):
            raise RuntimeError("Online BABEL probing accumulated encoder gradients")
        return summaries


__all__ = ["OnlineLinearProbe", "OnlineBabelProbes"]
