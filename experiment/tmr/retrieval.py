"""Complete-gallery retrieval with caption-group positives and bounded tiles."""

from __future__ import annotations

from typing import Sequence

import torch


def _rank_metrics(ranks: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_r1": float((ranks <= 1).float().mean()),
        f"{prefix}_r5": float((ranks <= 5).float().mean()),
        f"{prefix}_r10": float((ranks <= 10).float().mean()),
        f"{prefix}_medr": float(torch.quantile(ranks.float(), 0.5)),
    }


@torch.no_grad()
def evaluate_retrieval(
    motion_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    motion_caption_ids: Sequence[str | Sequence[str]],
    text_caption_ids: Sequence[str],
    chunk_size: int = 512,
) -> dict[str, float]:
    """Return recall fractions and 1-based median ranks in both directions.

    Text gallery/query IDs must be unique. A text query succeeds with any motion
    carrying its caption ID among its candidates. Motion-to-text accepts any of
    the motion's candidates. Score ties use ascending gallery index. Similarity
    matrices are at most ``chunk_size x chunk_size``; source-overlap filtering is
    exclusively a training-loss behavior and is not applied to retrieval.
    """
    if (
        motion_embeddings.ndim != 2 or text_embeddings.ndim != 2
        or motion_embeddings.shape[1] != text_embeddings.shape[1]
        or min(*motion_embeddings.shape, *text_embeddings.shape) <= 0
    ):
        raise ValueError("Embeddings must be nonempty [N,D] and [U,D] with equal feature widths")
    if motion_embeddings.device != text_embeddings.device:
        raise ValueError("Motion and text embeddings must be on the same device")
    if not motion_embeddings.is_floating_point() or not text_embeddings.is_floating_point():
        raise ValueError("Embeddings must have floating-point dtypes")
    if not torch.isfinite(motion_embeddings).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("Embeddings must be finite")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if len(text_caption_ids) != len(text_embeddings) or any(
        not isinstance(value, str) or not value for value in text_caption_ids
    ):
        raise ValueError("text_caption_ids must contain one nonempty string per embedding")
    if len(motion_caption_ids) != len(motion_embeddings):
        raise ValueError("motion_caption_ids must contain candidates for each motion")
    motion_candidates = []
    for value in motion_caption_ids:
        candidates = [value] if isinstance(value, str) else value
        if (
            not isinstance(candidates, Sequence) or not candidates
            or any(not isinstance(caption, str) or not caption for caption in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("motion_caption_ids must contain nonempty unique caption candidates")
        motion_candidates.append(candidates)
    if len(set(text_caption_ids)) != len(text_caption_ids):
        raise ValueError("text_caption_ids must be unique")
    if {caption for candidates in motion_candidates for caption in candidates} != set(text_caption_ids):
        raise ValueError("Every motion and text query must have a positive in its complete gallery")

    device = motion_embeddings.device
    motions, texts = motion_embeddings.float(), text_embeddings.float()
    count_m, count_t = len(motions), len(texts)
    caption_index = {caption: index for index, caption in enumerate(text_caption_ids)}
    max_candidates = max(map(len, motion_candidates))
    motion_positive_text = torch.tensor([
        [caption_index[caption] for caption in candidates] + [-1] * (max_candidates - len(candidates))
        for candidates in motion_candidates
    ], device=device, dtype=torch.long)
    motion_best_score = torch.full((count_m,), -torch.inf, device=device)
    motion_best_index = torch.full((count_m,), count_t, device=device, dtype=torch.long)
    text_best_score = torch.full((count_t,), -torch.inf, device=device)
    text_best_index = torch.full((count_t,), count_m, device=device, dtype=torch.long)

    # First pass: best positive score and first positive at that score. Computing
    # these with the same GEMM tiling used for ranking also preserves exact ties.
    with torch.autocast(device_type=device.type, enabled=False):
        for mi in range(0, count_m, chunk_size):
            mend = min(mi + chunk_size, count_m)
            motion_ids = torch.arange(mi, mend, device=device)
            for ti in range(0, count_t, chunk_size):
                tend = min(ti + chunk_size, count_t)
                text_ids = torch.arange(ti, tend, device=device)
                scores = motions[mi:mend] @ texts[ti:tend].T
                positive = torch.zeros_like(scores, dtype=torch.bool)
                for slot in range(motion_positive_text.shape[1]):
                    positive |= motion_positive_text[mi:mend, slot, None].eq(text_ids[None, :])
                positive_scores = scores.masked_fill(~positive, -torch.inf)
                best_score, best_local_index = positive_scores.max(dim=1)
                best_index = text_ids[best_local_index]
                previous_score = motion_best_score[mi:mend]
                previous_index = motion_best_index[mi:mend]
                better = (best_score > previous_score) | (
                    (best_score == previous_score) & (best_index < previous_index)
                    & torch.isfinite(best_score)
                )
                motion_best_score[mi:mend] = torch.where(better, best_score, previous_score)
                motion_best_index[mi:mend] = torch.where(better, best_index, previous_index)
                best_score, best_local_index = positive_scores.max(dim=0)
                best_index = motion_ids[best_local_index]
                previous_score = text_best_score[ti:tend]
                previous_index = text_best_index[ti:tend]
                better = (best_score > previous_score) | (
                    (best_score == previous_score) & (best_index < previous_index)
                    & torch.isfinite(best_score)
                )
                text_best_score[ti:tend] = torch.where(better, best_score, previous_score)
                text_best_index[ti:tend] = torch.where(better, best_index, previous_index)

        motion_ranks = torch.ones(count_m, device=device, dtype=torch.long)
        text_ranks = torch.ones(count_t, device=device, dtype=torch.long)
        for mi in range(0, count_m, chunk_size):
            mend = min(mi + chunk_size, count_m)
            motion_ids = torch.arange(mi, mend, device=device)
            for ti in range(0, count_t, chunk_size):
                tend = min(ti + chunk_size, count_t)
                text_ids = torch.arange(ti, tend, device=device)
                scores = motions[mi:mend] @ texts[ti:tend].T
                mscore = motion_best_score[mi:mend, None]
                trank = (scores > mscore) | (
                    (scores == mscore) & (text_ids[None, :] < motion_best_index[mi:mend, None])
                )
                motion_ranks[mi:mend] += trank.sum(dim=1)
                tscore = text_best_score[None, ti:tend]
                mrank = (scores > tscore) | (
                    (scores == tscore) & (motion_ids[:, None] < text_best_index[None, ti:tend])
                )
                text_ranks[ti:tend] += mrank.sum(dim=0)

    metrics = {**_rank_metrics(text_ranks, "t2m"), **_rank_metrics(motion_ranks, "m2t")}
    metrics["mean_r1"] = 0.5 * (metrics["t2m_r1"] + metrics["m2t_r1"])
    return metrics


__all__ = ["evaluate_retrieval"]
