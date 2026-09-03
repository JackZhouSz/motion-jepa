"""In-memory 100STYLE linear probing during Motion-JEPA pretraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dataset import build_style_datasets, load_style_label_index
from .features import (
    GLOBAL_MEAN_POOLING,
    PROJECT_ROOT,
    SPLITS,
    extract_features,
    resolve_pretraining_stats,
)
from .train_probe import train_linear_probe


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
        label_index = load_style_label_index(self.dataset_root)
        self.class_names = list(label_index.class_names)
        self.datasets, _ = build_style_datasets(
            self.dataset_root,
            splits=SPLITS,
            num_frames=int(data_config["num_frames"]),
            fps=int(data_config["fps"]),
            motion_dim=int(data_config["motion_dim"]),
            stats_root=stats_root,
            label_index=label_index,
        )
        train_classes = set(self.datasets["train"].labels)
        expected_classes = set(range(len(self.class_names)))
        if train_classes != expected_classes:
            missing = [
                self.class_names[index]
                for index in sorted(expected_classes - train_classes)
            ]
            raise ValueError(
                f"Linear-probe training split is missing style classes: {missing}"
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


__all__ = ["OnlineLinearProbe"]
