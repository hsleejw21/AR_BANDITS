from __future__ import annotations

from typing import Any

import numpy as np

from autoregressive_bandits.algorithms.base import Agent, ObservationContext


class RandomAgent(Agent):
    name = "random"

    def select_action(self, context: ObservationContext) -> int:
        return self._record_action(int(self.rng.integers(self.n_arms)))

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        self.t += 1


class EpsilonGreedyAgent(Agent):
    name = "epsilon_greedy"

    def reset(self, n_arms: int, horizon: int, seed: int | None = None, epsilon: float = 0.1, **params: Any):
        super().reset(n_arms, horizon, seed)
        self.epsilon = float(epsilon)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        return self

    def select_action(self, context: ObservationContext) -> int:
        unpulled = np.flatnonzero(self.counts == 0)
        if len(unpulled) > 0:
            return self._record_action(int(unpulled[0]))
        if self.rng.random() < self.epsilon:
            return self._record_action(int(self.rng.integers(self.n_arms)))
        return self._record_action(int(np.argmax(self.values)))

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
        self.t += 1


class UCB1Agent(Agent):
    name = "ucb1"

    def reset(self, n_arms: int, horizon: int, seed: int | None = None, exploration: float = 2.0, **params: Any):
        super().reset(n_arms, horizon, seed)
        self.exploration = float(exploration)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        return self

    def select_action(self, context: ObservationContext) -> int:
        unpulled = np.flatnonzero(self.counts == 0)
        if len(unpulled) > 0:
            return self._record_action(int(unpulled[0]))
        bonus = np.sqrt(self.exploration * np.log(max(2, self.t + 1)) / self.counts)
        return self._record_action(int(np.argmax(self.values + bonus)))

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
        self.t += 1
