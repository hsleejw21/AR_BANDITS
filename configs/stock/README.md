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
