from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from autoregressive_bandits.algorithms.base import Agent, ObservationContext


class DynLinUCBAgent(Agent):
    """DynLin-UCB stock adaptation over a finite portfolio action set."""

    name = "dynlin_ucb"

    def reset(
        self,
        n_arms: int,
        horizon: int,
        seed: int | None = None,
        lambda_: float | None = None,
        rho_bar: float = 0.5,
        sigma: float | list[float] | np.ndarray = 0.1,
        action_set: str | list[list[float]] = "one_hot_and_equal_baskets",
        basket_size: int = 2,
        include_cash: bool = False,
        beta: float | None = None,
        exploration_scale: float = 1.0,
        **params: Any,
    ) -> "DynLinUCBAgent":
        super().reset(n_arms, horizon, seed)
        if not 0 <= rho_bar < 1:
            raise ValueError("rho_bar must be in [0, 1)")
        self.rho_bar = float(rho_bar)
        self.lambda_ = float(lambda_ if lambda_ is not None else max(1.0, np.log(max(3, horizon))))
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        self.sigma_ref = float(np.mean(np.asarray(sigma, dtype=float)))
        self.beta_override = None if beta is None else float(beta)
        self.exploration_scale = float(exploration_scale)
        self.action_set = self._build_action_set(action_set, basket_size, include_cash)
        self.action_dim = self.action_set.shape[1]

        self.v_inv = np.eye(self.action_dim, dtype=float) / self.lambda_
        self.b = np.zeros(self.action_dim, dtype=float)
        self.h_hat = np.zeros(self.action_dim, dtype=float)
        self.epoch = 1
        self.epoch_action: np.ndarray | None = None
        self.epoch_action_index: int | None = None
        self.epoch_remaining = 0
        self.epoch_last_reward = 0.0
        return self

    def _build_action_set(
        self,
        action_set: str | list[list[float]],
        basket_size: int,
        include_cash: bool,
    ) -> np.ndarray:
        if isinstance(action_set, str):
            if action_set not in {"one_hot", "one_hot_and_equal_baskets"}:
                raise ValueError("action_set must be 'one_hot', 'one_hot_and_equal_baskets', or an explicit list")
            actions = [np.eye(self.n_arms, dtype=float)[i] for i in range(self.n_arms)]
            if action_set == "one_hot_and_equal_baskets" and basket_size > 1:
                size = min(int(basket_size), self.n_arms)
                for combo in combinations(range(self.n_arms), size):
                    action = np.zeros(self.n_arms, dtype=float)
                    action[list(combo)] = 1.0 / size
                    actions.append(action)
            if include_cash:
                actions.append(np.zeros(self.n_arms, dtype=float))
            return np.asarray(actions, dtype=float)
        actions = np.asarray(action_set, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != self.n_arms:
            raise ValueError("explicit action_set must have shape (n_actions, n_arms)")
        if len(actions) == 0:
            raise ValueError("action_set must contain at least one action")
        return actions

    def select_action(self, context: ObservationContext) -> np.ndarray:
        if self.epoch_action is not None and self.epoch_remaining > 0:
            return self._record_vector_action(self.epoch_action)

        beta = self._beta()
        scores = np.array(
            [
                float(action @ self.h_hat + beta * np.sqrt(max(0.0, action @ self.v_inv @ action)))
                for action in self.action_set
            ]
        )
        max_score = np.max(scores)
        ties = np.flatnonzero(np.isclose(scores, max_score))
        index = int(self.rng.choice(ties))
        self.epoch_action_index = index
        self.epoch_action = self.action_set[index].copy()
        self.epoch_remaining = self._epoch_length(self.epoch)
        return self._record_vector_action(self.epoch_action)

    def update(self, action: np.ndarray, reward: float, info: dict[str, Any]) -> None:
        self.epoch_last_reward = float(reward)
        self.t += 1
        self.epoch_remaining = max(0, self.epoch_remaining - 1)
        if self.epoch_remaining == 0 and self.epoch_action is not None:
            self._ridge_update(self.epoch_action, self.epoch_last_reward)
            self.epoch += 1
            self.epoch_action = None

    def _record_vector_action(self, action: np.ndarray) -> np.ndarray:
        self.last_action = action.copy()
        self.actions.append(action.copy())
        return action.copy()

    def _epoch_length(self, epoch: int) -> int:
        if self.rho_bar <= 0:
            return 1
        extra = int(np.floor(np.log(max(1, epoch)) / np.log(1.0 / self.rho_bar)))
        return max(1, 1 + extra)

    def _beta(self) -> float:
        if self.beta_override is not None:
            return self.beta_override
        t = max(1, self.t + 1)
        radius = np.sqrt(self.action_dim * np.log(1.0 + t / self.lambda_) + 2.0 * np.log(max(2, self.horizon)))
        return self.exploration_scale * (np.sqrt(self.lambda_) + max(self.sigma_ref, 1e-9) * radius)

    def _ridge_update(self, action: np.ndarray, reward: float) -> None:
        q = self.v_inv @ action
        denom = 1.0 + float(action @ q)
        self.v_inv -= np.outer(q, q) / denom
        self.b += action * reward
        self.h_hat = self.v_inv @ self.b
