from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StepObservation:
    """Observation returned after one bandit action."""

    t: int
    action: Any
    reward: float
    best_arm: int
    best_reward: float
    regret: float
    optimal: bool
    state: np.ndarray
    next_state: np.ndarray


class RestlessAR1BanditEnv:
    """Restless per-arm truncated AR(1) bandit environment.

    At each round, the learner observes the current reward of the selected arm.
    After the action, all arms evolve according to
    r_i(t+1) = clip(alpha * r_i(t) + epsilon_i(t), -reward_bound, reward_bound)
    by default.
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float | list[float] | np.ndarray,
        sigma: float | list[float] | np.ndarray,
        reward_bound: float = 1.0,
        horizon: int | None = None,
        initial_state: list[float] | np.ndarray | str | None = None,
        initial_state_mode: str = "uniform",
        noise_model: str = "additive",
        seed: int | None = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if reward_bound <= 0:
            raise ValueError("reward_bound must be positive")
        if initial_state_mode not in {"uniform", "stationary"}:
            raise ValueError("initial_state_mode must be 'uniform' or 'stationary'")
        if noise_model not in {"additive", "scaled_noise"}:
            raise ValueError("noise_model must be 'additive' or 'scaled_noise'")

        self.n_arms = int(n_arms)
        self.alpha = self._as_arm_vector(alpha, "alpha")
        self.sigma = self._as_arm_vector(sigma, "sigma")
        if np.any((self.alpha <= 0) | (self.alpha >= 1)):
            raise ValueError("alpha values must be in (0, 1)")
        if np.any(self.sigma < 0):
            raise ValueError("sigma values must be non-negative")
        self.reward_bound = float(reward_bound)
        self.horizon = horizon
        self.initial_state_mode = initial_state_mode
        self.noise_model = noise_model
        if isinstance(initial_state, str):
            self.initial_state_mode = initial_state
            self.initial_state = None
        else:
            self.initial_state = None if initial_state is None else np.asarray(initial_state, dtype=float)
        if self.initial_state is not None and self.initial_state.shape != (self.n_arms,):
            raise ValueError("initial_state must have length n_arms")
        self.seed = seed
        self.reset(seed)

    def _as_arm_vector(self, value: float | list[float] | np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n_arms, float(arr))
        if arr.shape != (self.n_arms,):
            raise ValueError(f"{name} must be scalar or length n_arms")
        return arr

    def reset(self, seed: int | None = None) -> "RestlessAR1BanditEnv":
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self.t = 0
        if self.initial_state is None:
            if self.initial_state_mode == "stationary":
                self.state = self._sample_stationary_state()
            else:
                self.state = self.rng.uniform(-self.reward_bound, self.reward_bound, size=self.n_arms)
        else:
            self.state = np.clip(self.initial_state.astype(float), -self.reward_bound, self.reward_bound)
        self.state_history = [self.state.copy()]
        return self

    def _sample_stationary_state(self, burn_in: int = 5000) -> np.ndarray:
        state = np.zeros(self.n_arms, dtype=float)
        for _ in range(burn_in):
            noise = self.rng.normal(0.0, self.sigma, size=self.n_arms)
            if self.noise_model == "additive":
                state = self.alpha * state + noise
            else:
                state = self.alpha * (state + noise)
            state = np.clip(state, -self.reward_bound, self.reward_bound)
        return state

    def clone_with_seed(self, seed: int) -> "RestlessAR1BanditEnv":
        return RestlessAR1BanditEnv(
            n_arms=self.n_arms,
            alpha=self.alpha,
            sigma=self.sigma,
            reward_bound=self.reward_bound,
            horizon=self.horizon,
            initial_state=self.initial_state,
            initial_state_mode=self.initial_state_mode,
            noise_model=self.noise_model,
            seed=seed,
        )

    def step(self, action: int) -> StepObservation:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")
        current = self.state.copy()
        reward = float(current[action])
        best_arm = int(np.argmax(current))
        best_reward = float(current[best_arm])
        noise = self.rng.normal(0.0, self.sigma, size=self.n_arms)
        if self.noise_model == "additive":
            next_state = self.alpha * self.state + noise
        else:
            next_state = self.alpha * (self.state + noise)
        self.state = np.clip(
            next_state,
            -self.reward_bound,
            self.reward_bound,
        )
        self.t += 1
        self.state_history.append(self.state.copy())
        return StepObservation(
            t=self.t - 1,
            action=int(action),
            reward=reward,
            best_arm=best_arm,
            best_reward=best_reward,
            regret=best_reward - reward,
            optimal=int(action) == best_arm,
            state=current,
            next_state=self.state.copy(),
        )
