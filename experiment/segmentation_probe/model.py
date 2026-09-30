"""One affine readout per within-patch frame, without temporal mixing."""

import torch
from torch import nn


class FrameLinearProbe(nn.Module):
    """Read p frame predictions from each token using p independent row groups."""

    def __init__(self, feature_dim: int, temporal_patch_size: int, num_classes: int = 120):
        super().__init__()
        if min(feature_dim, temporal_patch_size, num_classes) < 1:
            raise ValueError("Feature, patch and class dimensions must be positive")
        self.temporal_patch_size = int(temporal_patch_size)
        self.num_classes = int(num_classes)
        self.linear = nn.Linear(feature_dim, temporal_patch_size * num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("Frame probe expects [batch, tokens, channels]")
        # Output channel order is phase-major, class-minor: [t, phase, class].
        return self.linear(tokens).reshape(
            tokens.shape[0], tokens.shape[1] * self.temporal_patch_size, self.num_classes
        )


def complete_patch_frame_mask(lengths: torch.Tensor, token_layout) -> torch.Tensor:
    """Exclude all frames of incomplete patches, including their real prefix."""
    count = token_layout.token_num_frames * token_layout.temporal_patch_size
    complete_lengths = token_layout.valid_token_lengths(lengths) * token_layout.temporal_patch_size
    return torch.arange(count, device=lengths.device)[None] < complete_lengths[:, None]
