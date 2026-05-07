from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from autoregressive_bandits.environments.restless_ar1 import StepObservation


@dataclass(frozen=True)
class ClusteredVARSpec:
    n_clusters: int = 2
    self_persistence: float = 0.72
    within_cluster_strength: float = 0.12
    cross_cluster_strength: float = 0.015
    spectral_radius: float = 0.9
    noise_scale: float = 0.1


def generate_clustered_transition_matrix(
    n_arms: int,
    seed: int | None = None,
    n_clusters: int = 2,
    self_persistence: float = 0.72,
    within_cluster_strength: float = 0.12,
    cross_cluster_strength: float = 0.015,
    spectral_radius: float = 0.9,
    **params: Any,
) -> np.ndarray:
    """Generate a stable stock-like clustered VAR transition matrix."""

    if n_arms <= 0:
        raise ValueError("n_arms must be positive")
    if not 1 <= n_clusters <= n_arms:
        raise ValueError("n_clusters must be in [1, n_arms]")
    if spectral_radius <= 0:
        raise ValueError("spectral_radius must be positive")

    rng = np.random.default_rng(seed)
    clusters = np.arange(n_arms) % n_clusters
    matrix = np.zeros((n_arms, n_arms), dtype=float)
    for i in range(n_arms):
        for j in range(n_arms):
            if i == j:
                matrix[i, j] = self_persistence
            elif clusters[i] == clusters[j]:
                matrix[i, j] = rng.uniform(0.5, 1.0) * within_cluster_strength
            else:
                matrix[i, j] = rng.uniform(0.0, 1.0) * cross_cluster_strength

    current_radius = max(abs(np.linalg.eigvals(matrix)))
    if current_radius > 0:
        matrix *= min(1.0, spectral_radius / float(current_radius))
    return matrix


class RestlessVARBanditEnv:
    """Restless vector-autoregressive bandit environment.

    The latent reward vector evolves as
    r(t+1) = clip(A @ r(t) + epsilon(t), -reward_bound, reward_bound).
    """

    def __init__(
        self,
        n_arms: int,
        transition_matrix: list[list[float]] | np.ndarray | None = None,
        var_generation: dict[str, Any] | None = None,
        reward_bound: float = 1.0,
        horizon: int | None = None,
        initial_state: list[float] | np.ndarray | str | None = None,
        initial_state_mode: str = "stationary",
        noise_scale: float | list[float] | np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if reward_bound <= 0:
            raise ValueError("reward_bound must be positive")
        if initial_state_mode not in {"uniform", "stationary"}:
            raise ValueError("initial_state_mode must be 'uniform' or 'stationary'")

        self.n_arms = int(n_arms)
        self.reward_bound = float(reward_bound)
        self.horizon = horizon
        self.seed = seed
        self.var_generation = dict(var_generation or {})
        if transition_matrix is None:
            self.transition_matrix = generate_clustered_transition_matrix(
                self.n_arms,
                seed=seed,
                **{k: v for k, v in self.var_generation.items() if k != "noise_scale"},
            )
        else:
            self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        if self.transition_matrix.shape != (self.n_arms, self.n_arms):
            raise ValueError("transition_matrix must have shape (n_arms, n_arms)")

        if noise_scale is None:
            noise_scale = self.var_generation.get("noise_scale", 0.1)
        self.noise_scale = self._as_arm_vector(noise_scale, "noise_scale")
        if np.any(self.noise_scale < 0):
            raise ValueError("noise_scale values must be non-negative")

        self.initial_state_mode = initial_state_mode
        if isinstance(initial_state, str):
            self.initial_state_mode = initial_state
            self.initial_state = None
        else:
            self.initial_state = None if initial_state is None else np.asarray(initial_state, dtype=float)
        if self.initial_state is not None and self.initial_state.shape != (self.n_arms,):
            raise ValueError("initial_state must have length n_arms")
        self.reset(seed)

    def _as_arm_vector(self, value: float | list[float] | np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.n_arms, float(arr))
        if arr.shape != (self.n_arms,):
            raise ValueError(f"{name} must be scalar or length n_arms")
        return arr

    def reset(self, seed: int | None = None) -> "RestlessVARBanditEnv":
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
            noise = self.rng.normal(0.0, self.noise_scale, size=self.n_arms)
            state = np.clip(
                self.transition_matrix @ state + noise,
                -self.reward_bound,
                self.reward_bound,
            )
        return state

    def clone_with_seed(self, seed: int) -> "RestlessVARBanditEnv":
        return RestlessVARBanditEnv(
            n_arms=self.n_arms,
            transition_matrix=self.transition_matrix,
            var_generation=self.var_generation,
            reward_bound=self.reward_bound,
            horizon=self.horizon,
            initial_state=self.initial_state,
            initial_state_mode=self.initial_state_mode,
            noise_scale=self.noise_scale,
            seed=seed,
        )

    def step(self, action: int) -> StepObservation:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"action must be in [0, {self.n_arms})")
        current = self.state.copy()
        reward = float(current[action])
        best_arm = int(np.argmax(current))
        best_reward = float(current[best_arm])
        noise = self.rng.normal(0.0, self.noise_scale, size=self.n_arms)
        self.state = np.clip(
            self.transition_matrix @ self.state + noise,
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
