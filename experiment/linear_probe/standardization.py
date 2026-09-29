"""Train-fitted channel standardization for pooled encoder features."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChannelStandardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    epsilon: float = 1e-6

    @classmethod
    def fit(cls, train_features: torch.Tensor, epsilon: float = 1e-6):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if train_features.ndim != 2 or min(train_features.shape) == 0:
            raise ValueError("Expected a nonempty [samples, channels] training matrix")
        values = train_features.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("Training features must be finite")
        mean = values.mean(dim=0)
        scale = values.std(dim=0, correction=0).clamp_min(epsilon)
        return cls(mean.float(), scale.float(), epsilon)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError("Feature channels do not match the fitted standardizer")
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite")
        return (features.float() - self.mean.to(features.device)) / self.scale.to(features.device)


def standardize_feature_caches(caches: dict) -> tuple[dict, dict]:
    """Fit on train only, preserving labels/IDs and leaving raw caches untouched."""
    scaler = ChannelStandardizer.fit(caches["train"]["features"])
    transformed = {
        split: {**cache, "features": scaler.transform(cache["features"])}
        for split, cache in caches.items()
    }
    return transformed, {
        "mean": scaler.mean, "scale": scaler.scale, "epsilon": scaler.epsilon,
        "fit_split": "train", "correction": 0,
    }
