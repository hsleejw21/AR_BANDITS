import csv
import json

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
