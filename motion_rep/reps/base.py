"""Base class for MotionJEPA feature representations."""

from abc import ABC, abstractmethod

import torch


class MotionRepBase(ABC):
    def __init__(self, skeleton, fps: int):
        self.skeleton = skeleton
        self.fps = int(fps)

    @abstractmethod
    def __call__(self, local_rotations, root_positions, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def inverse(self, features: torch.Tensor):
        raise NotImplementedError


__all__ = ["MotionRepBase"]
