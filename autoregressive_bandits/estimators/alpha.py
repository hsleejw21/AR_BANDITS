from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AlphaEstimator(ABC):
    @abstractmethod
    def estimate(self, rewards: np.ndarray) -> float:
        raise NotImplementedError


class LeastSquaresAlphaEstimator(AlphaEstimator):
    def estimate(self, rewards: np.ndarray) -> float:
        return estimate_alpha_ls(rewards)


def estimate_alpha_ls(rewards: np.ndarray) -> float:
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 1 or len(rewards) < 2:
        raise ValueError("rewards must be a one-dimensional sequence with at least two values")
    x = rewards[:-1]
    y = rewards[1:]
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.5
    alpha = float(np.dot(x, y) / denom)
    return float(np.clip(alpha, 1e-6, 1 - 1e-6))
