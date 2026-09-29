"""In-memory BABEL attentive probing on a frozen EMA encoder."""

from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .attentive import AttentiveProbe
from .dataset import ClassificationTokenDataset
from .features import _seed_all
from .online import OnlineBabelProbes
from .train_classifier import _extract_token_features, _lr_factor, evaluate, train_epoch


class OnlineAttentiveBabelProbes(OnlineBabelProbes):
    """Reuse BABEL splits and train fresh heads on unpooled, masked tokens."""

    def __init__(self, training_config, probe_config, *, device):
        defaults = {"lr": 3e-4, "weight_decay": 0.05, **probe_config}
        super().__init__(training_config, defaults, device=device)
        self.warmup_epochs = int(probe_config.get("warmup_epochs", 5))
        self.final_lr = float(probe_config.get("final_lr", 1e-6))
        self.gradient_clip = float(probe_config.get("gradient_clip", 1.0))
        self.num_heads = int(probe_config.get("num_heads", 6))
        self.fps = int(training_config["data"]["fps"])
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("attentive_probe.warmup_epochs must be in [0, epochs)")
        if not 0 <= self.final_lr <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError("Require 0 <= attentive final_lr <= lr and lr > 0")
        if self.gradient_clip <= 0 or self.num_heads <= 0:
            raise ValueError("Attentive gradient_clip and num_heads must be positive")

    def evaluate(self, encoder):
        if encoder.training or any(p.requires_grad for p in encoder.parameters()):
            raise ValueError("Attentive probing requires a frozen eval-mode encoder")
        summaries = {}
        for name, splits in self.datasets.items():
            # Retain only one dataset's token cache at a time, in CPU BF16 RAM.
            caches = {
                split: _extract_token_features(
                    encoder, dataset, device=self.device,
                    batch_size=self.feature_batch_size, num_workers=self.num_workers,
                    use_bfloat16=self.use_bfloat16, show_progress=False,
                )
                for split, dataset in splits.items()
            }
            if any((cache["lengths"] < 1).any() for cache in caches.values()):
                raise ValueError("Attentive probing requires at least one valid token per motion")
            index = self.label_indices[name]
            datasets = {
                split: ClassificationTokenDataset(cache, label_index=index, fps=self.fps)
                for split, cache in caches.items()
            }
            _seed_all(self.seed)
            _, tokens, dim = caches["train"]["features"].shape
            head = AttentiveProbe(dim, tokens, index.num_classes, num_heads=self.num_heads).to(self.device)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: _lr_factor(
                    epoch, epochs=self.epochs, warmup_epochs=self.warmup_epochs,
                    final_factor=self.final_lr / self.learning_rate,
                ),
            )
            generator = torch.Generator().manual_seed(self.seed)
            loaders = {
                split: DataLoader(dataset, batch_size=self.batch_size,
                    shuffle=split == "train", generator=generator if split == "train" else None,
                    num_workers=0, pin_memory=self.device.type == "cuda")
                for split, dataset in datasets.items()
            }
            kwargs = dict(device=self.device, num_classes=index.num_classes,
                use_bfloat16=self.use_bfloat16, task="multilabel",
                row_labels_by_sample=index.row_labels_by_path)
            best_score, best_epoch, best_metrics = float("-inf"), None, None
            for epoch in range(1, self.epochs + 1):
                train_epoch(head, loaders["train"], optimizer,
                    gradient_clip=self.gradient_clip, **kwargs)
                metrics = asdict(evaluate(head, loaders["val"], **kwargs))
                if metrics["mean_average_precision"] > best_score:
                    best_score = metrics["mean_average_precision"]
                    best_epoch, best_metrics = epoch, metrics
                scheduler.step()
            summaries[name] = {
                "best_epoch": best_epoch, "best_val": best_metrics,
                "selection": "validation_best", "validation_used": True, "test": None,
                "num_classes": index.num_classes, "feature_dim": dim,
                "split_counts": {"train": len(datasets["train"]), "val": len(datasets["val"]), "test": 0},
                "dataset_root": str(self.dataset_roots[name]), "pooling": "attentive",
                "standardization": "none", "num_heads": self.num_heads,
            }
            del head, optimizer, scheduler, loaders, datasets, caches
        if any(p.grad is not None for p in encoder.parameters()):
            raise RuntimeError("Attentive probing accumulated encoder gradients")
        return summaries
