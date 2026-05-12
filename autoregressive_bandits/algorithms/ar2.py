from __future__ import annotations

from typing import Any

import numpy as np

from autoregressive_bandits.algorithms.base import Agent, ObservationContext


class AR2Agent(Agent):
    """Alternating and Restarting algorithm for restless AR(1) bandits."""

    name = "ar2"

    def reset(
        self,
        n_arms: int,
        horizon: int,
        seed: int | None = None,
        alpha: float | list[float] | np.ndarray = 0.9,
        sigma: float | list[float] | np.ndarray = 0.1,
        reward_bound: float = 1.0,
        epoch_size: int | None = None,
        c0: float | None = None,
        c1_multiplier: float = 24.0,
        theoretical_c0: bool = False,
        superior_mode: str = "recent_two",
        triggered_selection: str = "earliest",
        **params: Any,
    ) -> "AR2Agent":
        super().reset(n_arms, horizon, seed)
        self.alpha = self._as_arm_vector(alpha, "alpha")
        self.sigma = self._as_arm_vector(sigma, "sigma")
        if np.any((self.alpha <= 0) | (self.alpha >= 1)):
            raise ValueError("alpha values must be in (0, 1)")
        if np.any(self.sigma < 0):
            raise ValueError("sigma values must be non-negative")
        if superior_mode not in {"recent_two", "max_estimate"}:
            raise ValueError("superior_mode must be 'recent_two' or 'max_estimate'")
        if triggered_selection not in {"earliest", "ucb"}:
            raise ValueError("triggered_selection must be 'earliest' or 'ucb'")
        self.superior_mode = superior_mode
        self.triggered_selection = triggered_selection
        self.reward_bound = float(reward_bound)
        alpha_ref = float(np.mean(self.alpha))
        sigma_ref = float(np.mean(self.sigma))
        if epoch_size is None:
            safe_sigma = max(sigma_ref, 1e-12)
            epoch_size = int(np.ceil(self.n_arms * alpha_ref**-3 * safe_sigma**-3))
        self.epoch_size = max(self.n_arms + 1, int(epoch_size))
        if c0 is None:
            if theoretical_c0:
                safe = max(alpha_ref * max(sigma_ref, 1e-12), 1e-12)
                c0 = np.sqrt(4 * np.log(1 / safe) + 4 * np.log(self.epoch_size) + 2 * np.log(4 * self.n_arms))
            else:
                c0 = 0.01
        self.c0 = float(c0)
        self.c1_multiplier = float(c1_multiplier)
        self.c1 = self.c1_multiplier * self.c0
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

        current_round = self.t + 1
        if current_round % 2 == 1 and self.triggered:
            candidates = np.array(sorted(self.triggered), dtype=int)
            if self.triggered_selection == "ucb":
                ucbs = self.est_rewards[candidates] + self._uncertainty(candidates, current_round)
                action = int(candidates[np.argmax(ucbs)])
            else:
                action = int(candidates[np.argmin(self.trigger_round[candidates])])
        else:
            action = int(self.superior_arm)
        return self._record_action(action)

    def update(self, action: int, reward: float, info: dict[str, Any]) -> None:
        action = int(action)
        self.est_rewards *= self.alpha
        self.est_rewards[action] = self._clip(self.alpha[action] * reward)
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
            return
        if len(self.prev_actions) >= 2:
            a, b = self.prev_actions[-1], self.prev_actions[-2]
            self.superior_arm = a if self.est_rewards[a] >= self.est_rewards[b] else b
        elif self.prev_actions:
            self.superior_arm = self.prev_actions[-1]
        else:
            self.superior_arm = int(np.argmax(self.est_rewards))

    def _refresh_triggered_set(self) -> None:
        assert self.superior_arm is not None
        current_round = self.t + 1
        sup_est = self.est_rewards[self.superior_arm]
        for arm in range(self.n_arms):
            if arm == self.superior_arm or arm in self.triggered:
                continue
            tau = self.last_pull_round[arm]
            if not np.isfinite(tau):
                continue
            gap = max(1.0, current_round - tau + 1.0)
            threshold = self._uncertainty(np.array([arm]), current_round)[0]
            if sup_est - self.est_rewards[arm] <= threshold:
                self.triggered.add(arm)
                self.trigger_round[arm] = current_round

    def _uncertainty(self, arms: np.ndarray, current_round: int) -> np.ndarray:
        gaps = np.maximum(1.0, current_round - self.last_pull_round[arms] + 1.0)
        alpha = self.alpha[arms]
        sigma = self.sigma[arms]
        variance_sum = (alpha**2 - alpha ** (2 * gaps)) / np.maximum(1e-12, 1 - alpha**2)
        return self.c1 * sigma * np.sqrt(np.maximum(0.0, variance_sum))

    def _clip(self, value: float) -> float:
        return float(np.clip(value, -self.reward_bound, self.reward_bound))
