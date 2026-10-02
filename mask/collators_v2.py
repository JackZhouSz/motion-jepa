"""Length-aware 1D masks with padded indices instead of batch truncation.

All blocks of the same role have a common width; ``-1`` marks padding. Initial
blocks use each sample's valid length. Optional context selection reduces their
counts without shortening target blocks or violating the minimum context.
Consumers must exclude padded entries from attention and loss.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from model.token_layout import TokenLayout

from .collators import _StatefulMaskCollator, _block_length, _sample_ratio, _valid_lengths


def _ratio_bounds(values, name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two bounds")
    bounds = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in bounds) or not (
        0.0 <= bounds[0] <= bounds[1] <= 1.0
    ):
        raise ValueError(f"{name} must satisfy 0 <= min <= max <= 1, got {values}")
    return bounds


class MaskCollator1DV2(_StatefulMaskCollator):
    """Sample temporal blocks independently, with optional context selection."""

    def __init__(
        self,
        num_frames: int,
        enc_frame_mask_ratio: tuple[float, float] = (0.85, 1.0),
        pred_frame_mask_ratio: tuple[float, float] = (0.15, 0.2),
        nenc: int = 1,
        npred: int = 4,
        allow_overlap: bool = False,
        min_context_tokens: int = 1,
        min_context_ratio: float = 0.2,
        context_selection: str = "all",
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.nenc = int(nenc)
        self.npred = int(npred)
        self.allow_overlap = bool(allow_overlap)
        self.min_context_tokens = int(min_context_tokens)
        self.min_context_ratio = float(min_context_ratio)
        if context_selection not in ("all", "prefix", "random"):
            raise ValueError("context_selection must be 'all', 'prefix', or 'random'")
        self.context_selection = context_selection
        self.enc_frame_mask_ratio = _ratio_bounds(
            enc_frame_mask_ratio, "enc_frame_mask_ratio"
        )
        self.pred_frame_mask_ratio = _ratio_bounds(
            pred_frame_mask_ratio, "pred_frame_mask_ratio"
        )
        if min(self.num_frames, self.nenc, self.npred, self.min_context_tokens) <= 0:
            raise ValueError(
                "num_frames, nenc, npred, and min_context_tokens must be positive"
            )
        if not math.isfinite(self.min_context_ratio) or not (
            0.0 <= self.min_context_ratio <= 1.0
        ):
            raise ValueError("min_context_ratio must be in [0, 1]")
        self._check_feasible(self.num_frames)
        self._configuration = {
            "variant": "1d_v2",
            "num_frames": self.num_frames,
            "enc_frame_mask_ratio": self.enc_frame_mask_ratio,
            "pred_frame_mask_ratio": self.pred_frame_mask_ratio,
            "nenc": self.nenc,
            "npred": self.npred,
            "allow_overlap": self.allow_overlap,
            "min_context_tokens": self.min_context_tokens,
            "min_context_ratio": self.min_context_ratio,
            "context_selection": self.context_selection,
            "padding_index": -1,
            "max_sampling_attempts": 128,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        configuration = state.get("configuration")
        if not isinstance(configuration, dict) or configuration.get("variant") != (
            self._configuration["variant"]
        ):
            raise ValueError("V2 mask state requires a matching V2 mask configuration")
        # V2 checkpoints from before context selection kept every context token.
        configuration = dict(configuration)
        configuration.setdefault("context_selection", "all")
        super().load_state_dict({**state, "configuration": configuration})

    def _minimum_context(self, valid_length: int) -> int:
        return max(
            self.min_context_tokens, math.ceil(valid_length * self.min_context_ratio)
        )

    def _check_feasible(self, valid_length: int) -> None:
        minimum = self._minimum_context(valid_length)
        maximum_context = _block_length(valid_length, self.enc_frame_mask_ratio[1])
        if not self.allow_overlap:
            # Target blocks may overlap each other, so their smallest possible
            # union is one target block, regardless of the number of blocks.
            minimum_target = _block_length(
                valid_length, self.pred_frame_mask_ratio[0]
            )
            maximum_context = min(maximum_context, valid_length - minimum_target)
        if maximum_context < minimum:
            raise ValueError(
                "V2 mask configuration cannot satisfy minimum context: "
                f"valid_tokens={valid_length}, required={minimum}, "
                f"maximum_possible={maximum_context}"
            )

    @staticmethod
    def _interval(
        valid_length: int, length: int, generator: torch.Generator
    ) -> torch.Tensor:
        start = int(torch.randint(valid_length - length + 1, (), generator=generator))
        return torch.arange(start, start + length, dtype=torch.long)

    def _sample(
        self, valid_length: int, generator: torch.Generator
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self._check_feasible(valid_length)
        minimum = self._minimum_context(valid_length)
        for _attempt in range(128):
            pred_length = _block_length(
                valid_length, _sample_ratio(generator, self.pred_frame_mask_ratio)
            )
            enc_length = _block_length(
                valid_length, _sample_ratio(generator, self.enc_frame_mask_ratio)
            )
            targets = [
                self._interval(valid_length, pred_length, generator)
                for _ in range(self.npred)
            ]
            union = torch.zeros(valid_length, dtype=torch.bool)
            for target in targets:
                union[target] = True
            contexts = []
            for _ in range(self.nenc):
                context = self._interval(valid_length, enc_length, generator)
                if not self.allow_overlap:
                    context = context[~union[context]]
                contexts.append(context)
            if all(len(context) >= minimum for context in contexts):
                return contexts, targets
        raise ValueError(
            "V2 mask sampling could not satisfy minimum context after 128 attempts: "
            f"valid_tokens={valid_length}, required={minimum}. "
            "Reduce the target ratio or minimum context, or increase the context ratio."
        )

    @staticmethod
    def _pack(
        samples: list[list[torch.Tensor]], count: int
    ) -> list[torch.Tensor]:
        width = max(len(block) for sample in samples for block in sample)
        blocks = []
        for block_index in range(count):
            packed = torch.full((len(samples), width), -1, dtype=torch.long)
            for sample_index, sample in enumerate(samples):
                indices = sample[block_index]
                packed[sample_index, : len(indices)] = indices
            blocks.append(packed)
        return blocks

    def _select_contexts(
        self,
        samples: list[list[torch.Tensor]],
        valid_lengths: list[int],
        generator: torch.Generator,
    ) -> list[list[torch.Tensor]]:
        if self.context_selection == "all":
            return samples
        minimum = min(len(block) for sample in samples for block in sample)
        selected = []
        for blocks, valid_length in zip(samples, valid_lengths):
            # A short sample may set the batch minimum below a longer sample's
            # context guard. Keep the guard and pad any remaining differences.
            keep = max(minimum, self._minimum_context(valid_length))
            selected_blocks = []
            for indices in blocks:
                if len(indices) > keep and self.context_selection == "random":
                    chosen = torch.randperm(len(indices), generator=generator)[:keep]
                    indices = indices[chosen].sort().values
                else:
                    indices = indices[:keep]
                selected_blocks.append(indices)
            selected.append(selected_blocks)
        return selected

    def _masks(self, valid_lengths: list[int]):
        if not valid_lengths:
            raise ValueError("V2 mask collators require a nonempty batch")
        generator = torch.Generator().manual_seed(self.step())
        # Draw all seeds first.  Rejections or lengths in one sample therefore
        # cannot change another sample's geometry at the same batch position.
        seeds = torch.randint(
            torch.iinfo(torch.int64).max, (len(valid_lengths),), generator=generator
        )
        contexts_by_sample = []
        targets_by_sample = []
        for valid_length, seed in zip(valid_lengths, seeds.tolist()):
            contexts, targets = self._sample(
                valid_length, torch.Generator().manual_seed(seed)
            )
            contexts_by_sample.append(contexts)
            targets_by_sample.append(targets)
        contexts_by_sample = self._select_contexts(
            contexts_by_sample, valid_lengths, generator
        )
        return (
            self._pack(contexts_by_sample, self.nenc),
            self._pack(targets_by_sample, self.npred),
        )

    def __call__(self, batch):
        contexts, targets = self._masks(_valid_lengths(batch, self.num_frames))
        return torch.utils.data.default_collate(batch), contexts, targets


class PatchMaskCollator1DV2(MaskCollator1DV2):
    """Sample V2 masks on complete temporal patches and keep raw lengths."""

    def __init__(
        self,
        raw_num_frames: int,
        temporal_patch_size: int = 3,
        **kwargs,
    ) -> None:
        raw_num_frames = int(raw_num_frames)
        temporal_patch_size = int(temporal_patch_size)
        if raw_num_frames <= 0 or temporal_patch_size <= 0:
            raise ValueError("raw_num_frames and temporal_patch_size must be positive")
        self.layout = TokenLayout(
            kind="1d",
            patchified=True,
            raw_num_frames=raw_num_frames,
            token_num_frames=raw_num_frames // temporal_patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        super().__init__(num_frames=self.layout.token_num_frames, **kwargs)
        self._configuration = {
            **self._configuration,
            "variant": "patch_1d_v2",
            "raw_num_frames": self.layout.raw_num_frames,
            "token_num_frames": self.layout.token_num_frames,
            "temporal_patch_size": self.layout.temporal_patch_size,
        }

    def __call__(self, batch):
        raw_lengths = torch.tensor(
            _valid_lengths(batch, self.layout.raw_num_frames), dtype=torch.long
        )
        token_lengths = self.layout.valid_token_lengths(raw_lengths)
        if bool((token_lengths < 1).any()):
            raise ValueError("Every sample must contain at least one complete temporal patch")
        contexts, targets = self._masks(token_lengths.tolist())
        return torch.utils.data.default_collate(batch), contexts, targets


__all__ = ["MaskCollator1DV2", "PatchMaskCollator1DV2"]
