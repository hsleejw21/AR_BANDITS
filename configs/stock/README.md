# Stock Bandit Experiment Configs

These configs evaluate AR2/VAR2-style bandit algorithms on historical stock return paths downloaded with `yfinance`.

The default stock comparison uses estimated parameters only:

- `VAR2-Estimated`: estimates a full transition matrix from calibration returns.
- `AR2-Estimated`: estimates per-stock AR(1) persistence from calibration returns.
- `UCB1`, `Epsilon Greedy`, and `Random`.

The default split is past calibration followed by future online evaluation. The online agent observes only the selected ticker's reward at each step; the evaluator uses the full same-period return vector for regret.

Example:

```bash
python -m autoregressive_bandits.experiments.run \
  --config configs/stock/stock_tech_daily_2020_2024.json \
  --output outputs/stock_tech_daily_2020_2024
```

Larger-arm cluster experiments:

- `stock_tech_extended_daily_2020_2024.json`: 10 technology stocks in one broad cluster.
- `stock_two_cluster_tech_finance_daily_2020_2024.json`: 12 stocks across technology and finance.
- `stock_three_cluster_tech_finance_energy_daily_2020_2024.json`: 18 stocks across technology, finance, and energy.
- `stock_sector_clusters_24_daily_2020_2024.json`: 24 stocks across five broad sector groups.
- `stock_broad_market_36_daily_2020_2024.json`: 36 stocks across a broader market universe.
- `stock_sector_clusters_24_weekly_2015_2024.json`: 24 stocks at weekly frequency with a longer calibration period.
- `stock_sector_clusters_24_daily_2010_2024.json`: 24 stocks with 2010-2021 daily calibration and plain least-squares VAR2.
- `stock_sector_clusters_24_daily_2010_2024_ridge.json`: the same 24-stock setting with ridge VAR2.
- `stock_broad_market_36_daily_2010_2024.json`: 36 stocks with long daily calibration and plain least-squares VAR2.
- `stock_broad_market_36_daily_2010_2024_ridge.json`: the same 36-stock setting with ridge VAR2.
- `stock_broad_market_50_weekly_2010_2024_ridge.json`: 50 stocks at weekly frequency with ridge VAR2.

These are useful for testing whether VAR2's estimated cross-arm transition matrix helps more as the number of related arms grows.

For larger universes, prefer the ridge configs first. Plain least squares estimates `n_arms x n_arms` transition coefficients, so it can become unstable when the number of arms grows.
