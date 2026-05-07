from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from autoregressive_bandits.algorithms import AGENT_REGISTRY
from autoregressive_bandits.algorithms.base import ObservationContext
from autoregressive_bandits.environments import (
    RestlessAR1BanditEnv,
    RestlessVARBanditEnv,
    StockReturnsBanditEnv,
    generate_clustered_transition_matrix,
)
from autoregressive_bandits.estimators import LeastSquaresAlphaEstimator
from autoregressive_bandits.estimators import estimate_alpha_vector_ls, estimate_transition_matrix_ls
from autoregressive_bandits.experiments.config import ScenarioConfig


def run_experiment(config: ScenarioConfig, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    with (output_dir / "scenario.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"name": config.name, "agents": {}}
    estimated_parameters: dict[str, Any] = {}
    calibration_by_seed: dict[int, dict[str, Any] | None] = {}
    for agent_cfg in config.agents:
        label = agent_cfg.get("name") or agent_cfg.get("type")
        agent_rows = []
        for run_idx in range(config.n_runs):
            seed = config.seed + run_idx
            if seed not in calibration_by_seed:
                calibration_by_seed[seed] = _get_calibration(config, seed)
            calibration = calibration_by_seed[seed]
            if calibration is not None:
                estimated_parameters.setdefault(str(run_idx), calibration["diagnostics"])
            agent_rows.extend(_run_single_agent(config, agent_cfg, label, run_idx, seed, calibration))
        rows.extend(agent_rows)
        summary["agents"][label] = _summarize(agent_rows)

    _write_rows(output_dir / "metrics.csv", rows)
    if estimated_parameters:
        with (output_dir / "estimated_parameters.json").open("w", encoding="utf-8") as f:
            json.dump(estimated_parameters, f, indent=2)
    if config.environment_type == "stock_returns":
        _write_stock_metadata(output_dir, config)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _plot(rows, output_dir / "plots")
    _write_experiment_readme(output_dir, config, summary, bool(estimated_parameters))
    _update_experiment_index(output_dir, config, summary)
    return summary


def _run_single_agent(
    config: ScenarioConfig,
    agent_cfg: dict[str, Any],
    label: str,
    run_idx: int,
    seed: int,
    calibration: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    env = _build_environment(config, seed)
    n_arms = env.n_arms
    horizon = _environment_horizon(config, env)
    params = {k: v for k, v in agent_cfg.items() if k not in {"type", "name"}}
    if config.environment_type == "restless_var":
        _set_var_agent_defaults(params, agent_cfg, config, env, calibration)
    elif config.environment_type == "stock_returns":
        _set_stock_agent_defaults(params, agent_cfg, config, env, calibration)
    else:
        params.setdefault("alpha", config.alpha)
        params.setdefault("sigma", config.sigma)
    params.setdefault("reward_bound", config.reward_bound)
    if config.estimate_alpha and config.environment_type == "restless_ar1" and agent_cfg.get("type") == "ar2":
        params["alpha"] = _estimate_alpha_for_run(config, seed)

    agent_type = agent_cfg["type"]
    agent_cls = AGENT_REGISTRY[agent_type]
    agent = agent_cls().reset(n_arms, horizon, seed=seed, **params)
    cumulative_regret = 0.0
    rows = []
    for t in range(horizon):
        context = ObservationContext(t=t, n_arms=n_arms, horizon=horizon)
        action = agent.select_action(context)
        obs = env.step(action)
        agent.update(action, obs.reward, {"observation": obs})
        cumulative_regret += obs.regret
        rows.append(
            {
                "agent": label,
                "agent_type": agent_type,
                "run": run_idx,
                "t": t,
                "action": action,
                "reward": obs.reward,
                "best_arm": obs.best_arm,
                "best_reward": obs.best_reward,
                "regret": obs.regret,
                "cumulative_regret": cumulative_regret,
                "optimal": int(obs.optimal),
                "state": json.dumps(obs.state.tolist()),
            }
        )
    return rows


def _build_environment(config: ScenarioConfig, seed: int):
    if config.environment_type == "restless_ar1":
        if config.n_arms is None or config.horizon is None:
            raise ValueError("restless_ar1 requires n_arms and horizon")
        return RestlessAR1BanditEnv(
            n_arms=config.n_arms,
            alpha=config.alpha,
            sigma=config.sigma,
            reward_bound=config.reward_bound,
            horizon=config.horizon,
            initial_state=config.initial_state,
            initial_state_mode=config.initial_state_mode,
            noise_model=config.noise_model,
            seed=seed,
        )
    if config.environment_type == "restless_var":
        if config.n_arms is None or config.horizon is None:
            raise ValueError("restless_var requires n_arms and horizon")
        transition_matrix = config.transition_matrix
        if transition_matrix is None:
            transition_matrix = generate_clustered_transition_matrix(
                config.n_arms,
                seed=config.seed,
                **{k: v for k, v in (config.var_generation or {}).items() if k != "noise_scale"},
            ).tolist()
        return RestlessVARBanditEnv(
            n_arms=config.n_arms,
            transition_matrix=transition_matrix,
            var_generation=config.var_generation,
            reward_bound=config.reward_bound,
            horizon=config.horizon,
            initial_state=config.initial_state,
            initial_state_mode=config.initial_state_mode,
            noise_scale=(config.var_generation or {}).get("noise_scale", config.sigma),
            seed=seed,
        )
    if config.environment_type == "stock_returns":
        return StockReturnsBanditEnv(
            tickers=config.tickers,
            tickers_preset=config.tickers_preset,
            calibration_start=config.calibration_start,
            calibration_end=config.calibration_end,
            evaluation_start=config.evaluation_start,
            evaluation_end=config.evaluation_end,
            decision_frequency=config.decision_frequency,
            reward_type=config.reward_type,
            reward_scaling=config.reward_scaling,
            reward_bound=config.reward_bound,
            benchmark_ticker=config.benchmark_ticker,
            price_field=config.price_field,
            cache_dir=config.cache_dir,
            force_download=config.force_download,
            horizon=config.horizon,
            seed=seed,
        )
    raise ValueError(f"Unknown environment_type: {config.environment_type}")


def _environment_horizon(config: ScenarioConfig, env: Any) -> int:
    if config.environment_type == "stock_returns":
        return int(len(env.evaluation_rewards))
    if config.horizon is None:
        raise ValueError(f"{config.environment_type} requires horizon")
    return int(config.horizon)


def _set_var_agent_defaults(
    params: dict[str, Any],
    agent_cfg: dict[str, Any],
    config: ScenarioConfig,
    env: RestlessVARBanditEnv,
    calibration: dict[str, Any] | None,
) -> None:
    agent_type = agent_cfg["type"]
    diagonal_alpha = np.clip(np.diag(env.transition_matrix), 1e-6, 1 - 1e-6)
    if agent_type == "var_oracle_ar2":
        params.setdefault("transition_matrix", env.transition_matrix.tolist())
        params.setdefault("noise_scale", env.noise_scale.tolist())
    elif agent_type == "var_estimated_ar2":
        if calibration is None:
            raise ValueError("var_estimated_ar2 requires calibration")
        params.setdefault("transition_matrix", calibration["transition_matrix_hat"].tolist())
        params.setdefault("noise_scale", env.noise_scale.tolist())
    elif agent_type == "ar2":
        if agent_cfg.get("estimate_alpha", False) or config.estimate_alpha:
            if calibration is None:
                raise ValueError("estimated AR2 in VAR environment requires calibration")
            params.setdefault("alpha", calibration["alpha_hat"].tolist())
        else:
            params.setdefault("alpha", diagonal_alpha.tolist())
        params.setdefault("sigma", env.noise_scale.tolist())
    else:
        params.setdefault("alpha", diagonal_alpha.tolist())
        params.setdefault("sigma", env.noise_scale.tolist())


def _set_stock_agent_defaults(
    params: dict[str, Any],
    agent_cfg: dict[str, Any],
    config: ScenarioConfig,
    env: StockReturnsBanditEnv,
    calibration: dict[str, Any] | None,
) -> None:
    agent_type = agent_cfg["type"]
    if agent_type == "var_estimated_ar2":
        if calibration is None:
            raise ValueError("var_estimated_ar2 requires stock calibration")
        params.setdefault("transition_matrix", calibration["transition_matrix_hat"].tolist())
        params.setdefault("noise_scale", calibration["noise_scale"].tolist())
    elif agent_type == "var_oracle_ar2":
        if config.transition_matrix is None:
            raise ValueError("Stock experiments do not have a true transition_matrix; use var_estimated_ar2")
        params.setdefault("transition_matrix", config.transition_matrix)
        params.setdefault("noise_scale", calibration["noise_scale"].tolist() if calibration else 0.1)
    elif agent_type == "ar2":
        if agent_cfg.get("estimate_alpha", False) or config.estimate_alpha:
            if calibration is None:
                raise ValueError("estimated AR2 in stock environment requires calibration")
            params.setdefault("alpha", calibration["alpha_hat"].tolist())
        else:
            params.setdefault("alpha", config.alpha)
        params.setdefault("sigma", calibration["noise_scale"].tolist() if calibration else config.sigma)
    else:
        if calibration is not None:
            params.setdefault("alpha", calibration["alpha_hat"].tolist())
            params.setdefault("sigma", calibration["noise_scale"].tolist())


def _get_calibration(config: ScenarioConfig, seed: int) -> dict[str, Any] | None:
    if config.environment_type not in {"restless_var", "stock_returns"}:
        return None
    needs_calibration = config.estimate_alpha or config.estimate_transition_matrix
    needs_calibration = needs_calibration or any(
        agent.get("type") == "var_estimated_ar2" or agent.get("estimate_alpha", False)
        for agent in config.agents
    )
    if not needs_calibration:
        return None
    if config.estimator != "least_squares":
        raise ValueError("Only least_squares estimator is currently supported")

    env = _build_environment(config, seed + 10_000)
    if config.environment_type == "stock_returns":
        states_array = env.calibration_rewards.to_numpy(dtype=float)
    else:
        states = [env.state.copy()]
        for _ in range(max(2, config.calibration_rounds)):
            env.step(0)
            states.append(env.state.copy())
        states_array = np.asarray(states)
    target_radius = None
    if config.var_generation:
        target_radius = config.var_generation.get("spectral_radius")
    transition_matrix_hat = estimate_transition_matrix_ls(states_array, spectral_radius=target_radius)
    alpha_hat = estimate_alpha_vector_ls(states_array)
    residuals = _var_residuals(states_array, transition_matrix_hat)
    noise_scale = np.maximum(np.std(residuals, axis=0), 1e-6) if len(residuals) else np.full(states_array.shape[1], 0.1)
    diagnostics = {
        "calibration_rounds": int(states_array.shape[0]),
        "estimator": config.estimator,
        "alpha_hat": alpha_hat.tolist(),
        "transition_matrix_hat": transition_matrix_hat.tolist(),
        "noise_scale": noise_scale.tolist(),
    }
    if config.environment_type == "restless_var":
        true_matrix = env.transition_matrix
        true_alpha = np.clip(np.diag(true_matrix), 1e-6, 1 - 1e-6)
        diagnostics.update(
            {
                "true_diagonal_alpha": true_alpha.tolist(),
                "true_transition_matrix": true_matrix.tolist(),
                "transition_frobenius_error": float(np.linalg.norm(transition_matrix_hat - true_matrix)),
                "alpha_l2_error": float(np.linalg.norm(alpha_hat - true_alpha)),
            }
        )
    else:
        diagnostics.update(
            {
                "tickers": env.tickers,
                "stock_metadata": env.metadata,
                "return_correlation_matrix": env.calibration_rewards.corr().fillna(0.0).to_numpy().tolist(),
            }
        )
    return {
        "alpha_hat": alpha_hat,
        "transition_matrix_hat": transition_matrix_hat,
        "noise_scale": noise_scale,
        "diagnostics": diagnostics,
    }


def _var_residuals(states: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.shape[0] < 2:
        return np.empty((0, states.shape[1]), dtype=float)
    x = states[:-1]
    y = states[1:]
    return y - x @ transition_matrix.T


def _estimate_alpha_for_run(config: ScenarioConfig, seed: int) -> float:
    env = RestlessAR1BanditEnv(
        n_arms=config.n_arms,
        alpha=config.alpha,
        sigma=config.sigma,
        reward_bound=config.reward_bound,
        horizon=config.estimation_rounds,
        initial_state=config.initial_state,
        initial_state_mode=config.initial_state_mode,
        noise_model=config.noise_model,
        seed=seed + 10_000,
    )
    rewards = []
    for _ in range(max(2, config.estimation_rounds)):
        obs = env.step(0)
        rewards.append(obs.reward)
    return LeastSquaresAlphaEstimator().estimate(np.asarray(rewards))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    final_by_run: dict[int, float] = {}
    best_reward_by_run: dict[int, float] = {}
    optimal = []
    rewards = []
    for row in rows:
        final_by_run[row["run"]] = row["cumulative_regret"]
        best_reward_by_run[row["run"]] = best_reward_by_run.get(row["run"], 0.0) + row["best_reward"]
        optimal.append(row["optimal"])
        rewards.append(row["reward"])
    normalized_regrets = []
    for run, regret in final_by_run.items():
        denom = best_reward_by_run.get(run, 0.0)
        if abs(denom) > 1e-12:
            normalized_regrets.append(regret / denom)
    return {
        "mean_final_cumulative_regret": float(np.mean(list(final_by_run.values()))),
        "std_final_cumulative_regret": float(np.std(list(final_by_run.values()))),
        "mean_final_normalized_regret": float(np.mean(normalized_regrets)) if normalized_regrets else float("nan"),
        "std_final_normalized_regret": float(np.std(normalized_regrets)) if normalized_regrets else float("nan"),
        "optimal_pull_ratio": float(np.mean(optimal)),
        "mean_reward": float(np.mean(rewards)),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    agents = sorted({row["agent"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, 6))
    for agent in agents:
        xs, ys = _mean_series(rows, agent, "cumulative_regret")
        ax.plot(xs, ys, label=agent)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative regret")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_regret.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for agent in agents:
        xs, ys = _mean_series(rows, agent, "optimal")
        ax.plot(xs, ys, label=agent)
    ax.set_xlabel("Round")
    ax.set_ylabel("Optimal pull rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optimal_pull_ratio.png")
    plt.close(fig)


def _mean_series(rows: list[dict[str, Any]], agent: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    by_t: dict[int, list[float]] = {}
    for row in rows:
        if row["agent"] == agent:
            by_t.setdefault(row["t"], []).append(float(row[key]))
    xs = np.array(sorted(by_t))
    ys = np.array([np.mean(by_t[t]) for t in xs])
    if key == "optimal":
        ys = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return xs, ys


def _write_experiment_readme(
    output_dir: Path,
    config: ScenarioConfig,
    summary: dict[str, Any],
    has_estimated_parameters: bool,
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    agents = sorted(summary["agents"].items())
    lines = [
        f"# {config.name}",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Environment: `{config.environment_type}`",
        f"- Horizon: `{config.horizon if config.horizon is not None else 'derived from environment'}`",
        f"- Runs: `{config.n_runs}`",
        f"- Arms: `{config.n_arms if config.n_arms is not None else 'derived from environment'}`",
        f"- Seed: `{config.seed}`",
    ]
    if config.description:
        lines.append(f"- Description: {config.description}")
    if config.tags:
        lines.append(f"- Tags: `{', '.join(config.tags)}`")
    if config.environment_type == "restless_var":
        lines.extend(
            [
                f"- Calibration rounds: `{config.calibration_rounds}`",
                f"- Estimator: `{config.estimator}`",
                f"- VAR generation: `{json.dumps(config.var_generation or {}, sort_keys=True)}`",
            ]
        )
    if config.environment_type == "stock_returns":
        lines.extend(
            [
                f"- Ticker preset: `{config.tickers_preset}`",
                f"- Decision frequency: `{config.decision_frequency}`",
                f"- Reward type: `{config.reward_type}`",
                f"- Reward scaling: `{config.reward_scaling}`",
                f"- Calibration period: `{config.calibration_start}` to `{config.calibration_end}`",
                f"- Evaluation period: `{config.evaluation_start}` to `{config.evaluation_end}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `scenario.json`: exact scenario config used for this run.",
            "- `summary.json`: final aggregate metrics by algorithm.",
            "- `metrics.csv`: per-round rewards, actions, regret, and hidden state for evaluator analysis.",
            "- `plots/cumulative_regret.png`: mean cumulative regret by round.",
            "- `plots/optimal_pull_ratio.png`: cumulative optimal-pull ratio by round.",
        ]
    )
    if has_estimated_parameters:
        lines.append("- `estimated_parameters.json`: run-wise calibration estimates and true-vs-estimated diagnostics.")
    if config.environment_type == "stock_returns":
        lines.append("- `stock_metadata.json`: ticker universe, aligned dates, preprocessing, and correlation diagnostics.")
    lines.extend(["", "## Final Summary", ""])
    lines.append("| Agent | Final Regret | Normalized Regret | Optimal Pull Ratio | Mean Reward |")
    lines.append("|---|---:|---:|---:|---:|")
    for agent, metrics in agents:
        lines.append(
            "| "
            f"{agent} | "
            f"{metrics['mean_final_cumulative_regret']:.6g} | "
            f"{metrics['mean_final_normalized_regret']:.6g} | "
            f"{metrics['optimal_pull_ratio']:.6g} | "
            f"{metrics['mean_reward']:.6g} |"
        )
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _update_experiment_index(output_dir: Path, config: ScenarioConfig, summary: dict[str, Any]) -> None:
    if output_dir.parent.name != "outputs":
        return
    index_path = output_dir.parent / "experiment_index.csv"
    agents_summary = {
        agent: metrics["mean_final_cumulative_regret"]
        for agent, metrics in sorted(summary["agents"].items())
    }
    row = {
        "output_dir": output_dir.name,
        "name": config.name,
        "environment_type": config.environment_type,
        "horizon": config.horizon,
        "n_runs": config.n_runs,
        "n_arms": config.n_arms,
        "seed": config.seed,
        "description": config.description,
        "tags": ",".join(config.tags),
        "tickers_preset": config.tickers_preset or "",
        "decision_frequency": config.decision_frequency if config.environment_type == "stock_returns" else "",
        "reward_type": config.reward_type if config.environment_type == "stock_returns" else "",
        "calibration_period": (
            f"{config.calibration_start}:{config.calibration_end}" if config.environment_type == "stock_returns" else ""
        ),
        "evaluation_period": (
            f"{config.evaluation_start}:{config.evaluation_end}" if config.environment_type == "stock_returns" else ""
        ),
        "final_regrets_json": json.dumps(agents_summary, sort_keys=True),
    }
    rows: list[dict[str, Any]] = []
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as f:
            rows = [existing for existing in csv.DictReader(f) if existing["output_dir"] != output_dir.name]
    rows.append(row)
    fieldnames = list(row.keys())
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_stock_metadata(output_dir: Path, config: ScenarioConfig) -> None:
    env = _build_environment(config, config.seed)
    metadata = dict(env.metadata)
    metadata["tickers"] = env.tickers
    metadata["calibration_reward_dates"] = [str(idx.date()) for idx in env.calibration_rewards.index]
    metadata["evaluation_reward_dates"] = [str(idx.date()) for idx in env.evaluation_rewards.index]
    metadata["calibration_reward_shape"] = list(env.calibration_rewards.shape)
    metadata["evaluation_reward_shape"] = list(env.evaluation_rewards.shape)
    with (output_dir / "stock_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
