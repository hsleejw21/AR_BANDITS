from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationContext:
    t: int
    n_arms: int
    horizon: int


class Agent(ABC):
    name = "agent"

    def reset(self, n_arms: int, horizon: int, seed: int | None = None, **params: Any) -> "Agent":
        self.n_arms = int(n_arms)
        self.horizon = int(horizon)
        self.t = 0
        self.last_action: int | None = None
        self.actions: list[int] = []
        self.rng = np.random.default_rng(seed)
        return self

    @abstractmethod
    def select_action(self, context: ObservationContext) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        raise NotImplementedError

    def _record_action(self, action: int) -> int:
        self.last_action = int(action)
        self.actions.append(int(action))
        return int(action)
