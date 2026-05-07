# AR2 Replication And Extension Framework

This package implements an AR2-focused experiment framework for restless per-arm AR(1) bandits. It is meant to support two near-term goals:

- replicate and inspect the AR2 algorithm from `Non-Stationary Bandits with Auto-Regressive Temporal Dependency`;
- compare future AR2 extensions against the original AR2 algorithm and classic baselines under the same synthetic runner.

The baseline AR2 environment assumes independent per-arm AR(1) reward states. By default it uses the paper-style additive update

```text
r_i(t+1) = clip(alpha * r_i(t) + epsilon_i(t), -R, R)
```

where `epsilon_i(t)` is sampled independently from `Normal(0, sigma)`. The project also includes correlated synthetic VAR environments and a yfinance-backed stock-return environment for testing whether cross-arm correlation improves AR2-style decision making.

## Quick Start

Run tests:

```bash
pytest -q
```

Run the main AR2 example:

```bash
python -m autoregressive_bandits.experiments.run --config configs/example_ar2.json --output outputs/example_ar2
```

Run one paper-style synthetic AR2 scenario:

```bash
python -m autoregressive_bandits.experiments.run --config configs/ar2_synthetic_large_alpha_k10.json --output outputs/ar2_synthetic_large_alpha_k10
```

Run the medium-persistence 10-arm scenario:

```bash
python -m autoregressive_bandits.experiments.run --config configs/ar2_paper_style_alpha08_k10.json --output outputs/ar2_paper_style_alpha08_k10
```

Run a high-persistence scenario where AR2 should have a clearer advantage over UCB1:

```bash
python -m autoregressive_bandits.experiments.run --config configs/ar2_favorable_high_persistence_k10.json --output outputs/ar2_favorable_high_persistence_k10
```

Run Appendix-A-style synthetic scenarios with heterogeneous per-arm AR parameters, heterogeneous noise, and stationary initialization:

```bash
python -m autoregressive_bandits.experiments.run --config configs/ar2_paper_replica_large_alpha_k10.json --output outputs/ar2_paper_replica_large_alpha_k10
python -m autoregressive_bandits.experiments.run --config configs/ar2_paper_replica_small_alpha_k10.json --output outputs/ar2_paper_replica_small_alpha_k10
```

The medium-persistence setting `alpha = 0.8, sigma = 0.2, k = 10` is intentionally harder for AR2; UCB1 can be competitive or better there because the identity of the best arm changes often enough that AR2's restart and trigger exploration costs are not always recovered. The fixed `alpha = 0.98` config is a stress test for strong temporal persistence, not the paper's representative synthetic setup. The closer paper-style setup uses heterogeneous `alpha_i` values with mean `0.4` or `0.9`.

Run the first correlated-arm VAR prototype:

```bash
python -m autoregressive_bandits.experiments.run --config configs/var_clustered_oracle_k10.json --output outputs/var_clustered_oracle_k10
```

The VAR track uses `environment_type = "restless_var"` with latent dynamics `r(t+1) = clip(A @ r(t) + epsilon(t), -R, R)`. `VAR-Oracle-AR2` receives the transition matrix `A` but still observes only the pulled arm reward. Original AR2, UCB1, epsilon-greedy, and random run on the same hidden VAR paths.

Run the known-vs-estimated parameter comparison:

```bash
python -m autoregressive_bandits.experiments.run --config configs/var_known_unknown_comparison_k10.json --output outputs/var_known_unknown_comparison_k10
```

This compares `AR2-Known`, `AR2-Estimated`, `VAR2-Known`, and `VAR2-Estimated` in the same VAR environment. The estimated variants use a separate full-state calibration trajectory and least-squares estimates; calibration regret is not counted in the online regret metrics.

Run the first stock-data experiment:

```bash
python -m autoregressive_bandits.experiments.run --config configs/stock/stock_tech_daily_2020_2024.json --output outputs/stock_tech_daily_2020_2024
```

The stock track uses `environment_type = "stock_returns"` with yfinance data cached under `data/cache/yfinance/`. It estimates AR2/VAR2 parameters from a past calibration period, then evaluates the algorithms on a future historical reward path. Decision frequency (`1d`, `1wk`, `1mo`), date ranges, ticker presets, and reward definitions are configurable.

Outputs include:

- `scenario.json`
- `metrics.csv`
- `summary.json`
- `plots/cumulative_regret.png`
- `plots/optimal_pull_ratio.png`
- stock runs also include `stock_metadata.json`

The example configs use `c0 = 0.01` for AR2. This is a practical trigger calibration for clipped rewards in `[-1, 1]`; the conservative theoretical trigger constant can still be used by setting `"theoretical_c0": true` in an AR2 agent config.

## Project Layout

- `autoregressive_bandits/algorithms/`: AR2 plus classic baseline algorithms.
- `autoregressive_bandits/environments/`: restless AR(1) and VAR synthetic environments.
- `autoregressive_bandits/estimators/`: alpha estimation utilities for AR2 experiments.
- `autoregressive_bandits/experiments/`: config loading, simulation runner, plotting, and CLI entry points.
- `configs/`: runnable AR2 experiment configs.
- `docs/references/`: AR2 paper and local implementation notes.
- `docs/STOCK_VAR2_RESEARCH_ROADMAP.md`: stock-data research goals and extension roadmap.
- `tests/`: unit and integration tests.

## Algorithms

Registered algorithms:

- `ar2`
- `var_oracle_ar2`
- `var_estimated_ar2`
- `epsilon_greedy`
- `ucb1`
- `random`

Add a new algorithm by subclassing `autoregressive_bandits.algorithms.base.Agent`, registering it in `autoregressive_bandits/algorithms/__init__.py`, and adding it to a scenario config.

## Next Research Direction

The current research track is to compare original AR2 with VAR2-style extensions on synthetic correlated rewards and historical stock-return paths. The roadmap in `docs/STOCK_VAR2_RESEARCH_ROADMAP.md` describes the next steps: richer reward definitions, same-cluster versus mixed-cluster stock universes, ridge/sparse/rolling VAR2, and eventually regret-bound assumptions for cross-arm AR2 extensions.
