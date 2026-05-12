from __future__ import annotations

from typing import Any

import numpy as np

from autoregressive_bandits.algorithms.base import Agent, ObservationContext


class ARUCBAgent(Agent):
    """Autoregressive UCB adapted to stock-picking rewards.

    Each arm has its own ridge model over the global observed reward history
    vector z_t = (1, r_{t-1}, ..., r_{t-k}). Only the selected arm's model is
    updated, matching the bandit feedback available in the stock environment.
    """

    name = "ar_ucb"

    def reset(
        self,
        n_arms: int,
        horizon: int,
        seed: int | None = None,
        ar_order: int = 2,
        m_bound: float = 1.0,
        sigma: float | list[float] | np.ndarray = 0.1,
        lambda_: float = 1.0,
        delta: float | None = None,
        reward_bound: float = 1.0,
        bootstrap_each_arm: bool = True,
        **params: Any,
    ) -> "ARUCBAgent":
        super().reset(n_arms, horizon, seed)
        if ar_order < 0:
            raise ValueError("ar_order must be non-negative")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        if m_bound < 0:
            raise ValueError("m_bound must be non-negative")
        self.ar_order = int(ar_order)
        self.dim = self.ar_order + 1
        self.m_bound = float(m_bound)
        self.sigma = self._as_arm_vector(sigma, "sigma")
        if np.any(self.sigma < 0):
            raise ValueError("sigma values must be non-negative")
        self.lambda_ = float(lambda_)
        self.delta = float(delta) if delta is not None else 1.0 / max(1, int(horizon))
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.reward_bound = float(reward_bound)
        self.bootstrap_each_arm = bool(bootstrap_each_arm)

        self.v_inv = np.repeat((np.eye(self.dim) / self.lambda_)[None, :, :], self.n_arms, axis=0)
        self.b = np.zeros((self.n_arms, self.dim), dtype=float)
        self.gamma_hat = np.zeros((self.n_arms, self.dim), dtype=float)
        self.det_v = np.full(self.n_arms, self.lambda_**self.dim, dtype=float)
        self.history = np.zeros(self.ar_order, dtype=float)
        self._last_z = np.concatenate(([1.0], self.history))
        return self

    def _as_arm_vector(self, value: float | list[float] | np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n_arms, float(arr))
        if arr.shape != (self.n_arms,):
            raise ValueError(f"{name} must be scalar or length n_arms")
        return arr

    def select_action(self, context: ObservationContext) -> int:
        if self.bootstrap_each_arm and self.t < self.n_arms:
            return self._record_action(self.t)

        z = self._current_z()
        scores = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            z_norm = float(z @ self.v_inv[arm] @ z)
            beta = self._beta(arm)
            scores[arm] = float(self.gamma_hat[arm] @ z + beta * np.sqrt(max(0.0, z_norm)))
        max_score = np.max(scores)
        ties = np.flatnonzero(np.isclose(scores, max_score))
        return self._record_action(int(self.rng.choice(ties)))

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        action = int(action)
        z = self._current_z()
        q = self.v_inv[action] @ z
        denom = 1.0 + float(z @ q)
        self.v_inv[action] -= np.outer(q, q) / denom
        self.det_v[action] *= denom
        self.b[action] += z * float(reward)
        self.gamma_hat[action] = self.v_inv[action] @ self.b[action]

        if self.ar_order:
            self.history = np.concatenate(([float(np.clip(reward, -self.reward_bound, self.reward_bound))], self.history[:-1]))
        self._last_z = z
        self.t += 1

    def _current_z(self) -> np.ndarray:
        if self.ar_order == 0:
            return np.array([1.0], dtype=float)
        return np.concatenate(([1.0], self.history))

    def _beta(self, arm: int) -> float:
        det_ratio = max(self.det_v[arm] / (self.lambda_**self.dim), 1.0)
        return float(
            np.sqrt(self.lambda_ * (self.m_bound**2 + 1.0))
            + self.sigma[arm] * np.sqrt(2.0 * np.log(self.n_arms / self.delta) + np.log(det_ratio))
        )
