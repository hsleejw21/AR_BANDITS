import csv
import json
from pathlib import Path

import pandas as pd

from autoregressive_bandits.experiments.config import ScenarioConfig
from autoregressive_bandits.experiments.runner import run_experiment


def test_runner_writes_outputs_and_metrics(tmp_path):
    config = ScenarioConfig(
        name="tiny",
        horizon=8,
        n_arms=3,
        alpha=0.8,
        sigma=0.0,
        reward_bound=1.0,
        n_runs=2,
        seed=3,
        initial_state=[0.1, 0.4, -0.2],
        agents=[
            {"type": "ar2", "name": "AR2", "epoch_size": 6},
            {"type": "epsilon_greedy", "name": "EG", "epsilon": 0.0},
            {"type": "ucb1", "name": "UCB1"},
            {"type": "random", "name": "Random"},
        ],
    )

    summary = run_experiment(config, tmp_path)

    assert set(summary["agents"]) == {"AR2", "EG", "UCB1", "Random"}
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "plots" / "cumulative_regret.png").exists()
    assert (tmp_path / "plots" / "optimal_pull_ratio.png").exists()

    with (tmp_path / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == config.horizon * config.n_runs * len(config.agents)

    with (tmp_path / "summary.json").open() as f:
        loaded = json.load(f)
    assert loaded["name"] == "tiny"


def test_runner_uses_identical_hidden_paths_for_all_agents(tmp_path):
    config = ScenarioConfig(
        name="paths",
        horizon=5,
        n_arms=2,
        alpha=0.8,
        sigma=0.01,
        reward_bound=1.0,
        n_runs=1,
        seed=9,
        initial_state=[0.5, -0.5],
        agents=[{"type": "random", "name": "A"}, {"type": "ucb1", "name": "B"}],
    )

    run_experiment(config, tmp_path)

    with (tmp_path / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    states_by_agent = {}
    for row in rows:
        states_by_agent.setdefault(row["agent"], []).append(row["state"])

    assert states_by_agent["A"] == states_by_agent["B"]


def test_runner_runs_tiny_var_scenario_and_writes_outputs(tmp_path):
    config = ScenarioConfig(
        name="tiny_var",
        environment_type="restless_var",
        horizon=8,
        n_arms=2,
        sigma=0.0,
        reward_bound=1.0,
        n_runs=1,
        seed=5,
        initial_state=[0.5, -0.1],
        transition_matrix=[[0.8, 0.1], [0.2, 0.7]],
        var_generation={"noise_scale": 0.0},
        agents=[
            {"type": "var_oracle_ar2", "name": "VAR", "epoch_size": 8},
            {"type": "ar2", "name": "AR2", "epoch_size": 8},
            {"type": "ucb1", "name": "UCB1"},
            {"type": "random", "name": "Random"},
        ],
    )

    summary = run_experiment(config, tmp_path)

    assert set(summary["agents"]) == {"VAR", "AR2", "UCB1", "Random"}
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "plots" / "cumulative_regret.png").exists()


def test_runner_runs_known_unknown_var_scenario_and_writes_estimates(tmp_path):
    config = ScenarioConfig(
        name="tiny_known_unknown",
        environment_type="restless_var",
        horizon=8,
        n_arms=2,
        sigma=0.0,
        reward_bound=1.0,
        n_runs=1,
        seed=6,
        initial_state=[0.5, -0.1],
        transition_matrix=[[0.8, 0.1], [0.2, 0.7]],
        var_generation={"noise_scale": 0.0, "spectral_radius": 0.95},
        estimate_transition_matrix=True,
        calibration_rounds=20,
        agents=[
            {"type": "var_oracle_ar2", "name": "VAR2-Known", "epoch_size": 8},
            {"type": "var_estimated_ar2", "name": "VAR2-Estimated", "epoch_size": 8},
            {"type": "ar2", "name": "AR2-Known", "epoch_size": 8},
            {"type": "ar2", "name": "AR2-Estimated", "estimate_alpha": True, "epoch_size": 8},
            {"type": "ucb1", "name": "UCB1"},
        ],
    )

    summary = run_experiment(config, tmp_path)

    assert set(summary["agents"]) == {"VAR2-Known", "VAR2-Estimated", "AR2-Known", "AR2-Estimated", "UCB1"}
    assert (tmp_path / "estimated_parameters.json").exists()
    with (tmp_path / "estimated_parameters.json").open() as f:
        estimates = json.load(f)
    assert "0" in estimates
    assert "transition_matrix_hat" in estimates["0"]
    assert "alpha_hat" in estimates["0"]


def test_runner_runs_stock_scenario_from_cached_prices(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prices = pd.DataFrame(
        {
            "AAA": [10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18],
            "BBB": [20, 20.2, 20.1, 20.5, 21, 23, 22, 24, 25, 26],
            "CCC": [30, 29, 30, 31, 30, 32, 33, 31, 34, 35],
        },
        index=pd.date_range("2020-01-01", periods=10, freq="D"),
    )
    cache_path = cache_dir / "AAA_BBB_CCC_2020-01-01_2020-01-10_Close.csv"
    prices.to_csv(cache_path)

    config = ScenarioConfig(
        name="tiny_stock",
        environment_type="stock_returns",
        tickers=["AAA", "BBB", "CCC"],
        calibration_start="2020-01-01",
        calibration_end="2020-01-05",
        evaluation_start="2020-01-06",
        evaluation_end="2020-01-10",
        decision_frequency="1d",
        reward_type="simple_return",
        reward_scaling="standardize_by_calibration",
        reward_bound=1.0,
        cache_dir=str(cache_dir),
        n_runs=1,
        seed=10,
        agents=[
            {"type": "var_estimated_ar2", "name": "VAR2-Estimated", "epoch_size": 5},
            {"type": "ar_ucb", "name": "AR-UCB", "ar_order": 2, "m_bound": 1.0},
            {"type": "dynlin_ucb", "name": "DynLin-UCB", "rho_bar": 0.5, "action_set": "one_hot"},
            {"type": "ar2", "name": "AR2-Estimated", "estimate_alpha": True, "epoch_size": 5},
            {"type": "ucb1", "name": "UCB1"},
            {"type": "random", "name": "Random"},
        ],
    )

    summary = run_experiment(config, tmp_path / "out")

    assert set(summary["agents"]) == {"VAR2-Estimated", "AR-UCB", "DynLin-UCB", "AR2-Estimated", "UCB1", "Random"}
    assert (tmp_path / "out" / "stock_metadata.json").exists()
    assert (tmp_path / "out" / "estimated_parameters.json").exists()
    with (tmp_path / "out" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    states_by_agent = {}
    for row in rows:
        states_by_agent.setdefault(row["agent"], []).append(row["state"])
    assert states_by_agent["VAR2-Estimated"] == states_by_agent["UCB1"]


def test_runner_stock_scenario_supports_ridge_transition_estimator(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prices = pd.DataFrame(
        {
            "AAA": [10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18],
            "BBB": [20, 20.2, 20.1, 20.5, 21, 23, 22, 24, 25, 26],
            "CCC": [30, 29, 30, 31, 30, 32, 33, 31, 34, 35],
        },
        index=pd.date_range("2020-01-01", periods=10, freq="D"),
    )
    cache_path = cache_dir / "AAA_BBB_CCC_2020-01-01_2020-01-10_Close.csv"
    prices.to_csv(cache_path)

    config = ScenarioConfig(
        name="tiny_stock_ridge",
        environment_type="stock_returns",
        tickers=["AAA", "BBB", "CCC"],
        calibration_start="2020-01-01",
        calibration_end="2020-01-05",
        evaluation_start="2020-01-06",
        evaluation_end="2020-01-10",
        decision_frequency="1d",
        reward_type="simple_return",
        reward_scaling="standardize_by_calibration",
        reward_bound=1.0,
        cache_dir=str(cache_dir),
        estimator="ridge",
        ridge_lambda=2.0,
        transition_spectral_radius=0.9,
        n_runs=1,
        seed=11,
        agents=[
            {"type": "var_estimated_ar2", "name": "VAR2-Ridge", "epoch_size": 5},
            {"type": "ar2", "name": "AR2-Estimated", "estimate_alpha": True, "epoch_size": 5},
        ],
    )

    run_experiment(config, tmp_path / "out")

    with (tmp_path / "out" / "estimated_parameters.json").open() as f:
        estimates = json.load(f)
    assert estimates["0"]["estimator"] == "ridge"
    assert estimates["0"]["ridge_lambda"] == 2.0
    assert estimates["0"]["transition_spectral_radius"] == 0.9
