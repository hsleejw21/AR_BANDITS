from __future__ import annotations

from typing import Any

import numpy as np

from autoregressive_bandits.algorithms.base import Agent, ObservationContext


class VAROracleAR2Agent(Agent):
    """AR2-style agent that knows the VAR transition matrix.

    The agent observes only the pulled reward. It uses the oracle transition
    matrix to propagate a full reward-vector estimate between observations.
    """

    name = "var_oracle_ar2"

    def reset(
        self,
        n_arms: int,
        horizon: int,
        seed: int | None = None,
        transition_matrix: list[list[float]] | np.ndarray | None = None,
        noise_scale: float | list[float] | np.ndarray = 0.1,
        reward_bound: float = 1.0,
        epoch_size: int | None = None,
        c0: float = 0.01,
        superior_mode: str = "max_estimate",
        triggered_selection: str = "ucb",
        **params: Any,
    ) -> "VAROracleAR2Agent":
        super().reset(n_arms, horizon, seed)
        if transition_matrix is None:
            raise ValueError("VAROracleAR2Agent requires transition_matrix")
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        if self.transition_matrix.shape != (self.n_arms, self.n_arms):
            raise ValueError("transition_matrix must have shape (n_arms, n_arms)")
        self.noise_scale = self._as_arm_vector(noise_scale, "noise_scale")
        if np.any(self.noise_scale < 0):
            raise ValueError("noise_scale values must be non-negative")
        if superior_mode not in {"max_estimate", "recent_two"}:
            raise ValueError("superior_mode must be 'max_estimate' or 'recent_two'")
        if triggered_selection not in {"ucb", "earliest"}:
            raise ValueError("triggered_selection must be 'ucb' or 'earliest'")
        self.reward_bound = float(reward_bound)
        self.c0 = float(c0)
        self.c1 = 24.0 * self.c0
        self.superior_mode = superior_mode
        self.triggered_selection = triggered_selection
        if epoch_size is None:
            radius = max(abs(np.linalg.eigvals(self.transition_matrix)))
            persistence = min(0.99, max(0.05, float(radius)))
            sigma_ref = max(float(np.mean(self.noise_scale)), 1e-12)
            epoch_size = int(np.ceil(self.n_arms * persistence**-3 * sigma_ref**-3))
        self.epoch_size = max(self.n_arms + 1, int(epoch_size))
        self._restart_epoch()
        return self

    def _as_arm_vector(self, value: float | list[float] | np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n_arms, float(arr))
        if arr.shape != (self.n_arms,):
            raise ValueError(f"{name} must be scalar or length n_arms")
        return arr

    def _restart_epoch(self) -> None:
        self.epoch_start_t = self.t
        self.est_rewards = np.zeros(self.n_arms, dtype=float)
        self.est_cov = np.eye(self.n_arms, dtype=float) * self.reward_bound**2
        self.last_pull_round = np.full(self.n_arms, -np.inf, dtype=float)
        self.trigger_round = np.full(self.n_arms, np.inf, dtype=float)
        self.triggered: set[int] = set()
        self.prev_actions: list[int] = []
        self.superior_arm: int | None = None

    def select_action(self, context: ObservationContext) -> int:
        epoch_pos = self.t - self.epoch_start_t
        if epoch_pos == 0 and self.t != 0:
            self._restart_epoch()
            epoch_pos = 0

        if epoch_pos < self.n_arms:
            return self._record_action(epoch_pos)

        self._update_superior_arm()
        self._refresh_triggered_set()
        if self.superior_arm in self.triggered:
            self.triggered.remove(self.superior_arm)
            self.trigger_round[self.superior_arm] = np.inf

        if (self.t + 1) % 2 == 1 and self.triggered:
            candidates = np.array(sorted(self.triggered), dtype=int)
            if self.triggered_selection == "ucb":
                scores = self.est_rewards[candidates] + self._uncertainty(candidates)
                action = int(candidates[np.argmax(scores)])
            else:
                action = int(candidates[np.argmin(self.trigger_round[candidates])])
        else:
            action = int(self.superior_arm)
        return self._record_action(action)

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        action = int(action)
        observed_state = self.est_rewards.copy()
        observed_state[action] = float(np.clip(reward, -self.reward_bound, self.reward_bound))
        observed_cov = self.est_cov.copy()
        observed_cov[action, :] = 0.0
        observed_cov[:, action] = 0.0
        observed_cov[action, action] = 0.0

        self.est_rewards = np.clip(
            self.transition_matrix @ observed_state,
            -self.reward_bound,
            self.reward_bound,
        )
        noise_cov = np.diag(self.noise_scale**2)
        self.est_cov = self.transition_matrix @ observed_cov @ self.transition_matrix.T + noise_cov
        self.est_cov = (self.est_cov + self.est_cov.T) / 2.0

        self.last_pull_round[action] = self.t + 1
        self.triggered.discard(action)
        self.trigger_round[action] = np.inf
        self.prev_actions.append(action)
        self.prev_actions = self.prev_actions[-2:]
        self.t += 1
        if self.t - self.epoch_start_t >= self.epoch_size and self.t < self.horizon:
            self._restart_epoch()

    def _update_superior_arm(self) -> None:
        if self.superior_mode == "max_estimate":
            self.superior_arm = int(np.argmax(self.est_rewards))
        elif len(self.prev_actions) >= 2:
            a, b = self.prev_actions[-1], self.prev_actions[-2]
            self.superior_arm = a if self.est_rewards[a] >= self.est_rewards[b] else b
        elif self.prev_actions:
            self.superior_arm = self.prev_actions[-1]
        else:
            self.superior_arm = int(np.argmax(self.est_rewards))

    def _refresh_triggered_set(self) -> None:
        assert self.superior_arm is not None
        sup_est = self.est_rewards[self.superior_arm]
        uncertainty = self._uncertainty(np.arange(self.n_arms))
        for arm in range(self.n_arms):
            if arm == self.superior_arm or arm in self.triggered:
                continue
            if not np.isfinite(self.last_pull_round[arm]):
                continue
            if sup_est - self.est_rewards[arm] <= uncertainty[arm]:
                self.triggered.add(arm)
                self.trigger_round[arm] = self.t + 1

    def _uncertainty(self, arms: np.ndarray) -> np.ndarray:
        variances = np.maximum(0.0, np.diag(self.est_cov)[arms])
        return self.c1 * np.sqrt(variances)
