"""Single-layer classifier over temporal-mean normalized raw motion."""

from __future__ import annotations

import json

import torch
from torch import nn


class RawMotionLinearClassifier(nn.Module):
    """Classify the masked temporal mean with one affine layer."""

    def __init__(
        self,
        motion_dim: int | None = None,
        num_frames: int = 90,
        num_classes: int = 100,
        *,
        input_dim: int | None = None,
        pooling: str = "valid_frame_mean",
    ) -> None:
        super().__init__()
        if motion_dim is not None and input_dim is not None and motion_dim != input_dim:
            raise ValueError("motion_dim and input_dim must match when both are provided")
        self.input_dim = int(input_dim if input_dim is not None else motion_dim or 366)
        self.motion_dim = self.input_dim
        self.num_frames = int(num_frames)
        self.num_classes = int(num_classes)
        if pooling != "valid_frame_mean":
            raise ValueError(f"Unknown raw linear pooling: {pooling!r}")
        self.pooling = pooling
        if self.input_dim <= 0 or self.num_frames <= 0 or self.num_classes <= 0:
            raise ValueError("input_dim, num_frames, and num_classes must be positive")
        self.head = nn.Linear(self.input_dim, self.num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        motion: torch.Tensor,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.num_frames, self.input_dim)
        if motion.ndim != 3 or tuple(motion.shape[1:]) != expected:
            raise ValueError(
                f"Expected input [B,{self.num_frames},{self.input_dim}], "
                f"got {tuple(motion.shape)}"
            )
        if valid_frames is None:
            active = torch.ones(
                motion.shape[:2], device=motion.device, dtype=torch.bool
            )
        else:
            active = valid_frames.to(device=motion.device, dtype=torch.bool)
            if active.shape != motion.shape[:2]:
                raise ValueError(
                    f"valid_frames must have shape {tuple(motion.shape[:2])}, "
                    f"got {tuple(active.shape)}"
                )
        weights = active.unsqueeze(-1).to(dtype=motion.dtype)
        temporal_mean = (motion * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(temporal_mean)


def main() -> None:
    """Run the shared raw-classifier training entry point for this baseline."""
    from .train_classifier import build_parser, run

    parser = build_parser()
    parser.set_defaults(model="linear", input_source="raw")
    args = parser.parse_args()
    args.model = "linear"
    if args.input_source != "raw":
        parser.error("experiment.linear_probe.linear supports only normalized raw motion")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


__all__ = ["RawMotionLinearClassifier"]


if __name__ == "__main__":
    main()
