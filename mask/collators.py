"""Deterministic structured mask collators for MotionJEPA."""

from __future__ import annotations

from multiprocessing import Value
from typing import Any

import torch


def _sample_ratio(generator: torch.Generator, bounds: tuple[float, float]) -> float:
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(f"Mask ratios must satisfy 0 <= min <= max <= 1, got {bounds}")
    return lower + torch.rand((), generator=generator).item() * (upper - lower)


def _block_length(total: int, ratio: float) -> int:
    return min(total, max(1, int(round(total * ratio))))


def _valid_lengths(batch, num_frames: int) -> list[int]:
    lengths = [int(sample[2]) if len(sample) >= 3 else num_frames for sample in batch]
    if any(length < 1 or length > num_frames for length in lengths):
        raise ValueError(f"Valid lengths must be in [1, {num_frames}], got {lengths}")
    return lengths


class _StatefulMaskCollator:
    def __init__(self) -> None:
        self._counter = Value("q", -1)
        self._configuration: dict = {}

    def step(self) -> int:
        with self._counter.get_lock():
            self._counter.value += 1
            return int(self._counter.value)

    def state_dict(self) -> dict[str, Any]:
        with self._counter.get_lock():
            return {
                "counter": int(self._counter.value),
                "configuration": self._configuration,
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("configuration", self._configuration) != self._configuration:
            raise ValueError("Mask configuration differs from the saved checkpoint")
        with self._counter.get_lock():
            self._counter.value = int(state["counter"])


class MaskCollator1D(_StatefulMaskCollator):
    """Sample temporal context and target blocks for frame-token JEPA."""

    def __init__(
        self,
        num_frames: int,
        enc_frame_mask_ratio: tuple[float, float] = (0.85, 1.0),
        pred_frame_mask_ratio: tuple[float, float] = (0.15, 0.2),
        nenc: int = 1,
        npred: int = 4,
        allow_overlap: bool = False,
        context_selection: str = "prefix",
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.enc_frame_mask_ratio = tuple(enc_frame_mask_ratio)
        self.pred_frame_mask_ratio = tuple(pred_frame_mask_ratio)
        self.nenc = int(nenc)
        self.npred = int(npred)
        self.allow_overlap = bool(allow_overlap)
        if context_selection not in ("prefix", "random"):
            raise ValueError("context_selection must be 'prefix' or 'random'")
        self.context_selection = context_selection
        if self.num_frames <= 0 or self.nenc <= 0 or self.npred <= 0:
            raise ValueError("num_frames, nenc, and npred must be positive")
        self._configuration = {
            "variant": "1d",
            "num_frames": self.num_frames,
            "enc_frame_mask_ratio": self.enc_frame_mask_ratio,
            "pred_frame_mask_ratio": self.pred_frame_mask_ratio,
            "nenc": self.nenc,
            "npred": self.npred,
            "allow_overlap": self.allow_overlap,
            "context_selection": self.context_selection,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # Old 1D/patch checkpoints always used prefix truncation.
        configuration = dict(state.get("configuration", self._configuration))
        configuration.setdefault("context_selection", "prefix")
        super().load_state_dict({**state, "configuration": configuration})

    @staticmethod
    def _interval(
        total: int,
        valid_length: int,
        length: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        start = int(torch.randint(valid_length - length + 1, (), generator=generator).item())
        active = torch.zeros(total, dtype=torch.bool)
        active[start : start + length] = True
        return active

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        generator = torch.Generator().manual_seed(self.step())
        valid_lengths = _valid_lengths(batch, self.num_frames)
        shortest = min(valid_lengths)
        pred_length = _block_length(
            shortest, _sample_ratio(generator, self.pred_frame_mask_ratio)
        )
        enc_length = _block_length(
            shortest, _sample_ratio(generator, self.enc_frame_mask_ratio)
        )
        batch_pred: list[list[torch.Tensor]] = []
        batch_enc: list[list[torch.Tensor]] = []
        for valid_length in valid_lengths:
            valid = torch.arange(self.num_frames) < valid_length
            for _attempt in range(64):
                targets = [
                    self._interval(self.num_frames, valid_length, pred_length, generator)
                    for _ in range(self.npred)
                ]
                target_union = torch.stack(targets).any(dim=0)
                if self.allow_overlap or (valid & ~target_union).any():
                    break
            else:
                raise ValueError("Target mask configuration leaves no possible context frame")
            contexts = []
            for _ in range(self.nenc):
                context = self._interval(
                    self.num_frames, valid_length, enc_length, generator
                )
                if not self.allow_overlap:
                    context &= ~target_union
                if not context.any():
                    available = valid & ~target_union
                    if not available.any():
                        raise ValueError("Mask configuration leaves no context frames")
                    context = available
                contexts.append(context)
            batch_pred.append(targets)
            batch_enc.append(contexts)

        def pack(
            blocks: list[list[torch.Tensor]], count: int, selection: str = "prefix"
        ) -> list[torch.Tensor]:
            indices_by_block = [
                [torch.nonzero(sample[index], as_tuple=False).flatten() for sample in blocks]
                for index in range(count)
            ]
            minimum = min(len(indices) for block in indices_by_block for indices in block)
            if minimum <= 0:
                raise ValueError("Mask configuration generated an empty block")
            def select(indices: torch.Tensor) -> torch.Tensor:
                if selection == "random" and len(indices) > minimum:
                    chosen = torch.randperm(len(indices), generator=generator)[:minimum]
                    return indices[chosen].sort().values
                return indices[:minimum]

            return [torch.stack([select(indices) for indices in block]) for block in indices_by_block]

        return (
            collated_batch,
            pack(batch_enc, self.nenc, self.context_selection),
            pack(batch_pred, self.npred),
        )


class MaskCollator2D(_StatefulMaskCollator):
    """Sample structured frame-by-joint masks for skeletal-temporal JEPA."""

    def __init__(
        self,
        num_frames: int,
        num_joints: int = 30,
        enc_frame_mask_ratio: tuple[float, float] = (0.85, 1.0),
        enc_joint_mask_ratio: tuple[float, float] = (0.85, 1.0),
        pred_frame_mask_ratio: tuple[float, float] = (0.15, 0.2),
        pred_joint_mask_ratio: tuple[float, float] = (0.15, 0.2),
        nenc: int = 1,
        npred: int = 4,
        allow_overlap: bool = False,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.num_joints = int(num_joints)
        self.enc_frame_mask_ratio = tuple(enc_frame_mask_ratio)
        self.enc_joint_mask_ratio = tuple(enc_joint_mask_ratio)
        self.pred_frame_mask_ratio = tuple(pred_frame_mask_ratio)
        self.pred_joint_mask_ratio = tuple(pred_joint_mask_ratio)
        self.nenc = int(nenc)
        self.npred = int(npred)
        self.allow_overlap = bool(allow_overlap)
        if min(self.num_frames, self.num_joints, self.nenc, self.npred) <= 0:
            raise ValueError("Frame, joint, and mask counts must be positive")
        self._configuration = {
            "variant": "2d",
            "num_frames": self.num_frames,
            "num_joints": self.num_joints,
            "enc_frame_mask_ratio": self.enc_frame_mask_ratio,
            "enc_joint_mask_ratio": self.enc_joint_mask_ratio,
            "pred_frame_mask_ratio": self.pred_frame_mask_ratio,
            "pred_joint_mask_ratio": self.pred_joint_mask_ratio,
            "nenc": self.nenc,
            "npred": self.npred,
            "allow_overlap": self.allow_overlap,
        }

    def _rectangle(
        self,
        valid_length: int,
        frame_count: int,
        joint_count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        start = int(
            torch.randint(valid_length - frame_count + 1, (), generator=generator).item()
        )
        joints = torch.randperm(self.num_joints, generator=generator)[:joint_count]
        active = torch.zeros((self.num_frames, self.num_joints), dtype=torch.bool)
        active[start : start + frame_count, joints] = True
        return active

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        generator = torch.Generator().manual_seed(self.step())
        valid_lengths = _valid_lengths(batch, self.num_frames)
        shortest = min(valid_lengths)
        pred_frames = _block_length(
            shortest, _sample_ratio(generator, self.pred_frame_mask_ratio)
        )
        pred_joints = _block_length(
            self.num_joints, _sample_ratio(generator, self.pred_joint_mask_ratio)
        )
        enc_frames = _block_length(
            shortest, _sample_ratio(generator, self.enc_frame_mask_ratio)
        )
        enc_joints = _block_length(
            self.num_joints, _sample_ratio(generator, self.enc_joint_mask_ratio)
        )
        targets_by_sample: list[list[torch.Tensor]] = []
        contexts_by_sample: list[list[torch.Tensor]] = []
        for valid_length in valid_lengths:
            valid = torch.arange(self.num_frames)[:, None] < valid_length
            for _attempt in range(64):
                targets = [
                    self._rectangle(valid_length, pred_frames, pred_joints, generator)
                    for _ in range(self.npred)
                ]
                target_union = torch.stack(targets).any(dim=0)
                if self.allow_overlap or (valid & ~target_union).any():
                    break
            else:
                raise ValueError("Target mask configuration leaves no possible context cell")
            contexts = []
            for _ in range(self.nenc):
                context = self._rectangle(
                    valid_length, enc_frames, enc_joints, generator
                )
                if not self.allow_overlap:
                    context &= ~target_union
                if not context.any():
                    available = valid & ~target_union
                    if not available.any():
                        raise ValueError("Mask configuration leaves no context cells")
                    context = available
                contexts.append(context)
            targets_by_sample.append(targets)
            contexts_by_sample.append(contexts)

        contexts = [
            torch.stack([sample[index] for sample in contexts_by_sample])
            for index in range(self.nenc)
        ]
        targets = [
            torch.stack([sample[index] for sample in targets_by_sample])
            for index in range(self.npred)
        ]
        return collated_batch, contexts, targets


__all__ = ["MaskCollator1D", "MaskCollator2D"]
