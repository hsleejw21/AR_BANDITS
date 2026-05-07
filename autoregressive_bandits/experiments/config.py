from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScenarioConfig:
    name: str
    horizon: int | None = None
    n_arms: int | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    sigma: float | list[float] = 0.1
    alpha: float | list[float] = 0.9
    environment_type: str = "restless_ar1"
    transition_matrix: list[list[float]] | None = None
    var_generation: dict[str, Any] | None = None
    reward_bound: float = 1.0
    noise_model: str = "additive"
    n_runs: int = 1
    seed: int = 1
    initial_state: list[float] | None = None
    initial_state_mode: str = "uniform"
    agents: list[dict[str, Any]] = field(default_factory=lambda: [{"type": "ar2"}])
    estimate_alpha: bool = False
    estimate_transition_matrix: bool = False
    estimation_rounds: int = 100
    calibration_rounds: int = 1000
    estimator: str = "least_squares"
    ridge_lambda: float = 1.0
    transition_spectral_radius: float | None = None
    tickers_preset: str | None = None
    tickers: list[str] | None = None
    calibration_start: str | None = None
    calibration_end: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    decision_frequency: str = "1d"
    reward_type: str = "log_return"
    reward_scaling: str = "standardize_by_calibration"
    benchmark_ticker: str | None = None
    price_field: str = "Close"
    cache_dir: str = "data/cache/yfinance"
    force_download: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioConfig":
        if "environment_type" not in data:
            data = {"environment_type": "restless_ar1", **data}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "horizon": self.horizon,
            "n_arms": self.n_arms,
            "environment_type": self.environment_type,
            "transition_matrix": self.transition_matrix,
            "var_generation": self.var_generation,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "reward_bound": self.reward_bound,
            "noise_model": self.noise_model,
            "n_runs": self.n_runs,
            "seed": self.seed,
            "initial_state": self.initial_state,
            "initial_state_mode": self.initial_state_mode,
            "agents": self.agents,
            "estimate_alpha": self.estimate_alpha,
            "estimate_transition_matrix": self.estimate_transition_matrix,
            "estimation_rounds": self.estimation_rounds,
            "calibration_rounds": self.calibration_rounds,
            "estimator": self.estimator,
            "ridge_lambda": self.ridge_lambda,
            "transition_spectral_radius": self.transition_spectral_radius,
            "tickers_preset": self.tickers_preset,
            "tickers": self.tickers,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.evaluation_end,
            "decision_frequency": self.decision_frequency,
            "reward_type": self.reward_type,
            "reward_scaling": self.reward_scaling,
            "benchmark_ticker": self.benchmark_ticker,
            "price_field": self.price_field,
            "cache_dir": self.cache_dir,
            "force_download": self.force_download,
        }


def load_config(path: str | Path) -> ScenarioConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return ScenarioConfig.from_dict(data)
