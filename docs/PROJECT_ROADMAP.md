# Project Roadmap: Autoregressive Bandits For Stock Experiments

This document is the working plan for the project after the stock-ready migration. It records what the codebase currently supports, what should be checked before pushing to GitHub, and what research directions come next.

## Current State

The codebase now supports three experiment tracks:

- Synthetic AR(1): original AR2-style restless per-arm autoregressive bandits.
- Synthetic VAR: cross-arm reward dynamics through a transition matrix.
- Historical stock returns: yfinance-backed reward paths with calibration and evaluation splits.

Registered agents are:

- `ar2`: original AR2-style agent, with configurable `c1_multiplier` for paper-style or author-code-style triggering.
- `var_oracle_ar2` and `var_estimated_ar2`: VAR2-style AR2 agents for known or calibrated transition matrices.
- `ar_ucb`: stock-picking adaptation of AR-UCB using lagged observed rewards.
- `dynlin_ucb`: stock portfolio-action adaptation of DynLin-UCB over a finite action set.
- `ucb1`, `epsilon_greedy`, and `random`: classic baselines.

The stock-ready migration should be treated as a practical adaptation layer, not a claim that AR-UCB or DynLin-UCB are being run in their native paper environments.

## Push Checklist

Before pushing the next branch or opening a PR:

- Run `pytest -q`.
- Run at least one tiny cached stock integration test through the test suite.
- Run one real or cached stock smoke experiment and confirm `metrics.csv`, `summary.json`, plots, and `stock_metadata.json` are written.
- Inspect `metrics.csv` for vector-action rows from `dynlin_ucb`; actions should be JSON-serialized lists.
- Confirm README commands still match existing config paths.
- Keep generated outputs out of git unless a specific result artifact is intentionally being committed.

## Research Question

The central question is whether an AR2-style bandit algorithm can improve when it models relationships between arms.

Original AR2 treats each arm as a separate autoregressive reward process. That is natural for independent restless arms, but stocks are rarely independent. Stocks can move together through sectors, factors, market regimes, supply chains, and macro shocks. VAR2 is the first prototype for testing whether cross-arm structure can be useful in this setting.

In this project:

- AR2 estimates and exploits each arm's own temporal persistence.
- VAR2 estimates and exploits a transition matrix across arms.
- AR-UCB-Stock estimates per-arm autoregressive reward models from selected-stock reward history.
- DynLin-UCB-Portfolio treats finite stock baskets as vector actions and learns an optimistic steady-reward parameter.
- UCB1, epsilon-greedy, and random provide non-time-series baselines.

The goal is not yet to claim deployable trading performance. The goal is to compare decision rules on the same historical reward paths and understand when cross-arm modeling helps or hurts.

## Current Evidence And Open Validation

Synthetic VAR experiments already show a useful first pattern:

- When cross-arm correlation is weak, VAR2 and AR2 behave similarly.
- When clustered correlation is strong and noise is moderate, VAR2 beats AR2 and UCB1.
- When the transition matrix is estimated from enough calibration data, VAR2-Estimated approaches VAR2-Known.
- When correlation is dense and noisy, VAR2 can still help, but the advantage shrinks.

This motivates a stock-data track where the arms are real tickers and the reward path is historical return data.

The AR-UCB and DynLin-UCB stock adaptations still need empirical validation. The first question is not whether they beat all baselines, but whether their behavior is sensible and explainable under the shared stock runner.

## Stock Environment Design

The stock environment uses yfinance price data and converts it into reward vectors.

At each online step:

1. The hidden state is the vector of rewards across all selected tickers for the current date or period.
2. A scalar-action agent chooses one ticker, or a vector-action agent chooses a finite portfolio weight vector.
3. The agent observes the selected ticker reward or portfolio reward.
4. The evaluator computes regret using the best ticker reward or best candidate portfolio reward available in that same period.

The default split is:

- calibration period: past full return matrix used to estimate parameters;
- evaluation period: future online bandit path where only selected-arm rewards are observed.

This avoids look-ahead bias in parameter estimation.

## Reward Definitions

Different reward definitions answer different research questions.

- `log_return`: standard close-to-close log return.
- `simple_return`: close-to-close percentage return.
- `excess_return`: stock return minus a benchmark return such as SPY.
- `risk_adjusted_return`: return divided by calibration-period volatility.
- `directional_return`: sign-based reward for positive movement exposure.

The default scaling is `standardize_by_calibration`, which uses only calibration-period mean and standard deviation before clipping rewards. This keeps the reward range compatible with the current AR2/VAR2 implementation and avoids using future evaluation statistics.

## Experiment Families

### Stock-Ready Migration Smoke Tests

Use `configs/stock/stock_tech_daily_2020_2024_stock_ready_migration.json` as the first end-to-end comparison.

Expected checks:

- AR2 author preset and VAR2-Estimated run without changing the historical reward path.
- AR-UCB-Stock produces scalar ticker actions.
- DynLin-UCB-Portfolio produces JSON-serialized vector actions in `metrics.csv`.
- The same hidden stock return vectors are recorded for all agents at each time step.

### Same-Cluster Universes

These use tickers from one broad sector, such as tech, finance, energy, or healthcare.

Expected pattern:

- cross-stock relationships should be stronger;
- VAR2 may have an advantage over AR2 if the relationships are persistent enough.

### Multi-Cluster Universes

These mix several sectors or large-cap groups.

Expected pattern:

- VAR2 may help if it can estimate useful block structure;
- performance may depend strongly on calibration length and decision frequency.

### Weak-Relation Universes

These intentionally mix stocks with weaker or less stable relationships.

Expected pattern:

- VAR2's advantage should shrink;
- if VAR2 overfits noisy relationships, it may underperform AR2 or UCB1.

### Frequency Tests

The same ticker universe should be tested at daily, weekly, and monthly frequencies.

Expected pattern:

- daily rewards may be noisier;
- weekly or monthly rewards may reveal more stable relationships;
- too-slow frequency reduces sample size and makes estimation harder.

### Reward-Definition Tests

The same ticker universe and period should be tested with several reward types.

Expected pattern:

- raw returns may mostly reflect market beta;
- excess returns may emphasize stock-specific or sector-specific relationships;
- risk-adjusted returns may reduce dominance by high-volatility tickers.

## Future Algorithmic Extensions

VAR2, AR-UCB-Stock, and DynLin-UCB-Portfolio are first prototypes. Future versions can add richer time-series structure and cleaner financial evaluation.

Promising directions:

- ridge VAR2 for more stable transition estimation;
- sparse VAR2 or Lasso-style transition estimation;
- rolling-window VAR2 to adapt to changing market regimes;
- online transition estimation from bandit feedback;
- richer AR-UCB context definitions, such as per-stock lag windows instead of only selected-reward history;
- richer DynLin-UCB action sets, such as sector baskets or constrained long-only portfolios;
- regime-switching models;
- factor models with market and sector latent factors;
- state-space or Kalman-style models;
- higher-order VAR(p);
- uncertainty models that distinguish estimation uncertainty from process noise.

Longer term, the theoretical goal is to define a regret bound for an AR2-derived algorithm that uses cross-arm structure. The synthetic and stock experiments should help identify the right assumptions before attempting a proof.

## Near-Term Success Criteria

The next stage should count as successful if:

- VAR2-Estimated beats AR2-Estimated in at least some same-cluster or multi-cluster stock settings;
- the advantage is smaller or absent in weak-relation universes;
- performance changes are explainable by estimated correlation or transition diagnostics;
- results are checked across at least two time periods and two decision frequencies;
- failure cases are documented rather than hidden.

## Limitations

The first stock environment intentionally omits several trading details:

- transaction costs;
- slippage;
- shorting and leverage;
- liquidity constraints;
- risk limits;
- survivorship-bias-safe universe construction.

These omissions are acceptable for the first research milestone because the purpose is algorithm comparison on shared historical reward paths, not live trading validation.
