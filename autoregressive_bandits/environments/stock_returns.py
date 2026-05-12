from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoregressive_bandits.environments.restless_ar1 import StepObservation


TICKER_PRESETS: dict[str, list[str]] = {
    "tech_core": ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL"],
    "tech_extended": ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM"],
    "finance_core": ["JPM", "BAC", "GS", "MS", "WFC"],
    "energy_core": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "healthcare_core": ["UNH", "JNJ", "PFE", "MRK", "ABBV"],
    "sector_mixed_core": ["AAPL", "JPM", "XOM", "UNH", "WMT", "NEE"],
    "two_cluster_tech_finance": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "GOOGL",
        "META",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
    ],
    "three_cluster_tech_finance_energy": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "GOOGL",
        "META",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "HAL",
    ],
    "sector_clusters_24": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "GOOGL",
        "META",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "UNH",
        "JNJ",
        "PFE",
        "MRK",
        "WMT",
        "PG",
        "KO",
        "COST",
    ],
    "broad_market_36": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "GOOGL",
        "META",
        "AMZN",
        "AVGO",
        "ORCL",
        "CRM",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "HAL",
        "UNH",
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "LLY",
        "WMT",
        "PG",
        "KO",
        "COST",
        "HD",
        "MCD",
        "CAT",
        "DE",
    ],
    "broad_market_50": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "GOOGL",
        "META",
        "AMZN",
        "AVGO",
        "ORCL",
        "CRM",
        "QCOM",
        "TXN",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
        "BLK",
        "AXP",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "HAL",
        "PSX",
        "OXY",
        "UNH",
        "JNJ",
        "PFE",
        "MRK",
        "ABBV",
        "LLY",
        "AMGN",
        "GILD",
        "WMT",
        "PG",
        "KO",
        "COST",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "CAT",
        "DE",
        "BA",
        "HON",
        "UPS",
        "GE",
    ],
    "mega_cap_mixed": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "cyclical_vs_defensive": ["CAT", "DE", "XOM", "WMT", "PG", "KO"],
    "unrelated_mixed_core": ["NVDA", "JPM", "XOM", "PFE", "KO", "NEE"],
}


@dataclass(frozen=True)
class StockDataset:
    tickers: list[str]
    calibration_rewards: pd.DataFrame
    evaluation_rewards: pd.DataFrame
    metadata: dict[str, Any]


class StockReturnsBanditEnv:
    """Historical stock-return bandit environment.

    The hidden state at each step is the vector of same-period rewards across
    all tickers. The learner observes only the reward of the selected ticker.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        tickers_preset: str | None = None,
        calibration_start: str | None = None,
        calibration_end: str | None = None,
        evaluation_start: str | None = None,
        evaluation_end: str | None = None,
        decision_frequency: str = "1d",
        reward_type: str = "log_return",
        reward_scaling: str = "standardize_by_calibration",
        reward_bound: float = 1.0,
        benchmark_ticker: str | None = None,
        price_field: str = "Close",
        cache_dir: str | Path = "data/cache/yfinance",
        force_download: bool = False,
        horizon: int | None = None,
        seed: int | None = None,
        price_data: pd.DataFrame | None = None,
    ) -> None:
        del seed
        if reward_bound <= 0:
            raise ValueError("reward_bound must be positive")
        self.tickers = resolve_tickers(tickers=tickers, tickers_preset=tickers_preset)
        self.tickers_preset = tickers_preset
        self.calibration_start = _require_date(calibration_start, "calibration_start")
        self.calibration_end = _require_date(calibration_end, "calibration_end")
        self.evaluation_start = _require_date(evaluation_start, "evaluation_start")
        self.evaluation_end = _require_date(evaluation_end, "evaluation_end")
        if not self.calibration_start < self.calibration_end < self.evaluation_start < self.evaluation_end:
            raise ValueError("Expected calibration_start < calibration_end < evaluation_start < evaluation_end")
        if decision_frequency not in {"1d", "1wk", "1mo"}:
            raise ValueError("decision_frequency must be one of '1d', '1wk', '1mo'")
        if reward_type not in {"log_return", "simple_return", "excess_return", "risk_adjusted_return", "directional_return"}:
            raise ValueError("Unsupported reward_type")
        if reward_scaling not in {"none", "standardize_by_calibration"}:
            raise ValueError("reward_scaling must be 'none' or 'standardize_by_calibration'")

        self.decision_frequency = decision_frequency
        self.reward_type = reward_type
        self.reward_scaling = reward_scaling
        self.reward_bound = float(reward_bound)
        self.benchmark_ticker = benchmark_ticker
        self.price_field = price_field
        self.cache_dir = Path(cache_dir)
        self.force_download = force_download
        self.horizon = horizon

        dataset = build_stock_dataset(
            tickers=self.tickers,
            calibration_start=self.calibration_start,
            calibration_end=self.calibration_end,
            evaluation_start=self.evaluation_start,
            evaluation_end=self.evaluation_end,
            decision_frequency=decision_frequency,
            reward_type=reward_type,
            reward_scaling=reward_scaling,
            reward_bound=self.reward_bound,
            benchmark_ticker=benchmark_ticker,
            price_field=price_field,
            cache_dir=self.cache_dir,
            force_download=force_download,
            price_data=price_data,
        )
        self.tickers = dataset.tickers
        self.n_arms = len(self.tickers)
        evaluation = dataset.evaluation_rewards
        if horizon is not None and horizon > 0:
            evaluation = evaluation.iloc[: int(horizon)]
        if len(evaluation) == 0:
            raise ValueError("Evaluation period produced no reward rows")
        self.calibration_rewards = dataset.calibration_rewards
        self.evaluation_rewards = evaluation
        self.metadata = {
            **dataset.metadata,
            "horizon": int(len(self.evaluation_rewards)),
            "evaluation_dates": [str(idx.date()) for idx in self.evaluation_rewards.index],
        }
        self.reset()

    def reset(self, seed: int | None = None) -> "StockReturnsBanditEnv":
        del seed
        self.t = 0
        self.state = self.evaluation_rewards.iloc[0].to_numpy(dtype=float)
        self.state_history = [self.state.copy()]
        return self

    def step(self, action: int | list[float] | np.ndarray) -> StepObservation:
        current = self.evaluation_rewards.iloc[self.t].to_numpy(dtype=float)
        action_value: int | list[float]
        if np.isscalar(action):
            action_int = int(action)
            if not 0 <= action_int < self.n_arms:
                raise ValueError(f"action must be in [0, {self.n_arms})")
            reward = float(current[action_int])
            best_arm = int(np.argmax(current))
            best_reward = float(current[best_arm])
            optimal = action_int == best_arm
            action_value = action_int
        else:
            weights = np.asarray(action, dtype=float)
            if weights.shape != (self.n_arms,):
                raise ValueError("vector action must have length n_arms")
            reward = float(weights @ current)
            action_set = getattr(self, "portfolio_action_set", None)
            if action_set is None:
                action_set = np.eye(self.n_arms, dtype=float)
            action_set = np.asarray(action_set, dtype=float)
            if action_set.ndim != 2 or action_set.shape[1] != self.n_arms:
                raise ValueError("portfolio_action_set must have shape (n_actions, n_arms)")
            action_rewards = action_set @ current
            best_arm = int(np.argmax(action_rewards))
            best_reward = float(action_rewards[best_arm])
            optimal = bool(np.isclose(reward, best_reward))
            action_value = weights.tolist()
        next_t = min(self.t + 1, len(self.evaluation_rewards) - 1)
        next_state = self.evaluation_rewards.iloc[next_t].to_numpy(dtype=float)
        obs = StepObservation(
            t=self.t,
            action=action_value,
            reward=reward,
            best_arm=best_arm,
            best_reward=best_reward,
            regret=best_reward - reward,
            optimal=optimal,
            state=current,
            next_state=next_state.copy(),
        )
        self.t += 1
        self.state = next_state.copy()
        self.state_history.append(self.state.copy())
        return obs


def resolve_tickers(tickers: list[str] | None = None, tickers_preset: str | None = None) -> list[str]:
    if tickers:
        resolved = [ticker.upper() for ticker in tickers]
    else:
        preset = tickers_preset or "tech_core"
        if preset not in TICKER_PRESETS:
            raise ValueError(f"Unknown tickers_preset: {preset}")
        resolved = TICKER_PRESETS[preset]
    if len(set(resolved)) != len(resolved):
        raise ValueError("tickers must be unique")
    if len(resolved) < 2:
        raise ValueError("At least two tickers are required")
    return resolved


def build_stock_dataset(
    tickers: list[str],
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    decision_frequency: str = "1d",
    reward_type: str = "log_return",
    reward_scaling: str = "standardize_by_calibration",
    reward_bound: float = 1.0,
    benchmark_ticker: str | None = None,
    price_field: str = "Close",
    cache_dir: str | Path = "data/cache/yfinance",
    force_download: bool = False,
    price_data: pd.DataFrame | None = None,
) -> StockDataset:
    all_tickers = list(tickers)
    if benchmark_ticker and benchmark_ticker.upper() not in all_tickers:
        all_tickers.append(benchmark_ticker.upper())
    if price_data is None:
        prices = load_yfinance_prices(
            all_tickers,
            start=calibration_start,
            end=evaluation_end,
            price_field=price_field,
            cache_dir=cache_dir,
            force_download=force_download,
        )
    else:
        prices = _normalize_price_data(price_data, all_tickers)
    prices = prices.sort_index().loc[:, all_tickers]
    prices = _resample_prices(prices, decision_frequency)
    base_returns = _compute_base_returns(prices, reward_type)

    if reward_type == "excess_return":
        if not benchmark_ticker:
            raise ValueError("benchmark_ticker is required for excess_return")
        benchmark = base_returns[benchmark_ticker.upper()]
        rewards = base_returns[tickers].sub(benchmark, axis=0)
    elif reward_type == "risk_adjusted_return":
        rewards = base_returns[tickers]
    elif reward_type == "directional_return":
        rewards = np.sign(base_returns[tickers]).astype(float)
    else:
        rewards = base_returns[tickers]
    rewards = rewards.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    calibration_raw = rewards.loc[(rewards.index >= calibration_start) & (rewards.index <= calibration_end)]
    evaluation_raw = rewards.loc[(rewards.index >= evaluation_start) & (rewards.index <= evaluation_end)]
    if len(calibration_raw) < 2:
        raise ValueError("Calibration period produced fewer than two reward rows")
    if len(evaluation_raw) < 1:
        raise ValueError("Evaluation period produced no reward rows")

    if reward_type == "risk_adjusted_return":
        volatility = calibration_raw.std(axis=0).replace(0.0, 1.0)
        calibration_raw = calibration_raw / volatility
        evaluation_raw = evaluation_raw / volatility

    calibration_rewards, evaluation_rewards, scaling_metadata = _scale_rewards(
        calibration_raw,
        evaluation_raw,
        reward_scaling=reward_scaling,
        reward_bound=reward_bound,
    )

    metadata = {
        "tickers": tickers,
        "all_downloaded_tickers": all_tickers,
        "calibration_start": str(calibration_start.date()),
        "calibration_end": str(calibration_end.date()),
        "evaluation_start": str(evaluation_start.date()),
        "evaluation_end": str(evaluation_end.date()),
        "actual_calibration_start": str(calibration_rewards.index[0].date()),
        "actual_calibration_end": str(calibration_rewards.index[-1].date()),
        "actual_evaluation_start": str(evaluation_rewards.index[0].date()),
        "actual_evaluation_end": str(evaluation_rewards.index[-1].date()),
        "decision_frequency": decision_frequency,
        "reward_type": reward_type,
        "reward_scaling": reward_scaling,
        "reward_bound": reward_bound,
        "benchmark_ticker": benchmark_ticker,
        "price_field": price_field,
        "calibration_rows": int(len(calibration_rewards)),
        "evaluation_rows": int(len(evaluation_rewards)),
        "return_correlation_matrix": calibration_rewards.corr().fillna(0.0).to_numpy().tolist(),
        **scaling_metadata,
    }
    return StockDataset(
        tickers=tickers,
        calibration_rewards=calibration_rewards,
        evaluation_rewards=evaluation_rewards,
        metadata=metadata,
    )


def load_yfinance_prices(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_field: str = "Close",
    cache_dir: str | Path = "data/cache/yfinance",
    force_download: bool = False,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = {
        "tickers": sorted(tickers),
        "start": str(start.date()),
        "end": str(end.date()),
        "price_field": price_field,
    }
    name = "_".join(cache_key["tickers"]) + f"_{cache_key['start']}_{cache_key['end']}_{price_field}.csv"
    cache_path = cache_dir / name.replace("/", "-")
    if cache_path.exists() and not force_download:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required to download stock data. Install the package or provide cached CSV data.") from exc

    # yfinance treats end as exclusive; adding a day keeps the requested end date available.
    raw = yf.download(
        tickers=tickers,
        start=str(start.date()),
        end=str((end + pd.Timedelta(days=1)).date()),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError(f"yfinance returned no data for tickers: {tickers}")
    if isinstance(raw.columns, pd.MultiIndex):
        if price_field in raw.columns.get_level_values(0):
            prices = raw[price_field]
        elif price_field == "Adj Close" and "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            raise ValueError(f"price_field {price_field!r} not found in yfinance data")
    else:
        if len(tickers) != 1:
            raise ValueError("Expected MultiIndex columns for multiple tickers")
        prices = raw[[price_field]].rename(columns={price_field: tickers[0]})
    prices = _normalize_price_data(prices, tickers)
    prices.to_csv(cache_path)
    (cache_path.with_suffix(".meta.json")).write_text(json.dumps(cache_key, indent=2), encoding="utf-8")
    return prices


def _normalize_price_data(price_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    prices = price_data.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [str(col).upper() for col in prices.columns]
    missing = [ticker for ticker in tickers if ticker.upper() not in prices.columns]
    if missing:
        raise ValueError(f"Missing price columns for tickers: {missing}")
    prices = prices[[ticker.upper() for ticker in tickers]]
    return prices.dropna(axis=0, how="any")


def _resample_prices(prices: pd.DataFrame, decision_frequency: str) -> pd.DataFrame:
    if decision_frequency == "1d":
        return prices
    rule = {"1wk": "W-FRI", "1mo": "M"}[decision_frequency]
    return prices.resample(rule).last().dropna(axis=0, how="any")


def _compute_base_returns(prices: pd.DataFrame, reward_type: str) -> pd.DataFrame:
    if reward_type == "simple_return":
        returns = prices.pct_change()
    else:
        returns = np.log(prices / prices.shift(1))
    return returns.dropna(axis=0, how="any")


def _scale_rewards(
    calibration_raw: pd.DataFrame,
    evaluation_raw: pd.DataFrame,
    reward_scaling: str,
    reward_bound: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if reward_scaling == "none":
        calibration = calibration_raw.copy()
        evaluation = evaluation_raw.copy()
        metadata: dict[str, Any] = {}
    else:
        mean = calibration_raw.mean(axis=0)
        std = calibration_raw.std(axis=0).replace(0.0, 1.0)
        calibration = (calibration_raw - mean) / std
        evaluation = (evaluation_raw - mean) / std
        metadata = {
            "scaling_mean": mean.to_dict(),
            "scaling_std": std.to_dict(),
        }
    calibration = calibration.clip(lower=-reward_bound, upper=reward_bound)
    evaluation = evaluation.clip(lower=-reward_bound, upper=reward_bound)
    return calibration, evaluation, metadata


def _require_date(value: str | None, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{name} is required for stock_returns")
    return pd.Timestamp(value)
