"""Symmetric multi-positive contrastive alignment with overlap exclusions."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def _encode_ids(ids: Sequence[str], size: int, name: str, device: torch.device) -> torch.Tensor:
    if len(ids) != size or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{name} must contain one nonempty string per sample")
    mapping: dict[str, int] = {}
    encoded = [mapping.setdefault(value, len(mapping)) for value in ids]
    return torch.tensor(encoded, device=device, dtype=torch.long)


def symmetric_multi_positive_info_nce(
    motion_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    caption_ids: Sequence[str],
    source_ids: Sequence[str],
    start_frames: Sequence[int] | torch.Tensor,
    end_frames: Sequence[int] | torch.Tensor,
    temperature: float = 0.1,
    *,
    caption_candidate_ids: Sequence[Sequence[str]] | None = None,
) -> torch.Tensor:
    """Align sampled text with motion, accepting each motion's candidate captions.

    Different captions from intersecting half-open intervals of the same source
    are excluded from the denominator. Positive matches always take precedence.
    The numerator sums the probability mass of every positive, in each direction.
    """
    if (
        motion_embeddings.ndim != 2
        or text_embeddings.shape != motion_embeddings.shape
        or min(motion_embeddings.shape) <= 0
    ):
        raise ValueError("Motion and text embeddings must have equal nonempty shape [B,D]")
    if motion_embeddings.device != text_embeddings.device:
        raise ValueError("Motion and text embeddings must be on the same device")
    if not motion_embeddings.is_floating_point() or not text_embeddings.is_floating_point():
        raise ValueError("Embeddings must have floating-point dtypes")
    if not torch.isfinite(motion_embeddings).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("Embeddings must be finite")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    size = len(motion_embeddings)
    device = motion_embeddings.device
    captions = _encode_ids(caption_ids, size, "caption_ids", device)
    sources = _encode_ids(source_ids, size, "source_ids", device)
    starts = torch.as_tensor(start_frames, device=device)
    ends = torch.as_tensor(end_frames, device=device)
    if starts.shape != (size,) or ends.shape != (size,):
        raise ValueError("Frame intervals must contain one start and end per sample")
    for value in (starts, ends):
        if value.is_complex() or not torch.isfinite(value).all() or (value != value.long()).any():
            raise ValueError("Frame intervals must contain finite integer frame indices")
    starts, ends = starts.long(), ends.long()
    if (starts < 0).any() or (ends <= starts).any():
        raise ValueError("Frame intervals require 0 <= start < end")

    positive = captions[:, None].eq(captions[None, :])
    if caption_candidate_ids is not None:
        if len(caption_candidate_ids) != size or any(
            isinstance(candidates, str) or not candidates
            or any(not isinstance(value, str) or not value for value in candidates)
            or caption_ids[row] not in candidates
            for row, candidates in enumerate(caption_candidate_ids)
        ):
            raise ValueError("Caption candidates must include the sampled caption for each motion")
        candidate_sets = [set(candidates) for candidates in caption_candidate_ids]
        positive = torch.tensor(
            [[caption in candidates for caption in caption_ids] for candidates in candidate_sets],
            device=device, dtype=torch.bool,
        )
    same_source = sources[:, None].eq(sources[None, :])
    overlap = (starts[:, None] < ends[None, :]) & (starts[None, :] < ends[:, None])
    allowed = positive | ~(same_source & overlap)
    # Disable surrounding AMP: logits and logsumexp remain float32.
    with torch.autocast(device_type=device.type, enabled=False):
        logits = motion_embeddings.float() @ text_embeddings.float().T / temperature
        positive_logits = logits.masked_fill(~positive, -torch.inf)
        allowed_logits = logits.masked_fill(~allowed, -torch.inf)
        motion_loss = torch.logsumexp(allowed_logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
        text_loss = torch.logsumexp(allowed_logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
        return 0.5 * (motion_loss.mean() + text_loss.mean())


__all__ = ["symmetric_multi_positive_info_nce"]
