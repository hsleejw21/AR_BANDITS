# Stock VAR2 Research Roadmap

## Research Question

The central question is whether an AR2-style bandit algorithm can improve when it models relationships between arms.

Original AR2 treats each arm as a separate autoregressive reward process. That is natural for independent restless arms, but stocks are rarely independent. Stocks can move together through sectors, factors, market regimes, supply chains, and macro shocks. VAR2 is the first prototype for testing whether cross-arm structure can be useful in this setting.

In this project:

- AR2 estimates and exploits each arm's own temporal persistence.
- VAR2 estimates and exploits a transition matrix across arms.
- UCB1, epsilon-greedy, and random provide non-time-series baselines.

The goal is not yet to claim deployable trading performance. The goal is to compare decision rules on the same historical reward paths and understand when cross-arm modeling helps or hurts.

## Current Evidence

Synthetic VAR experiments already show a useful first pattern:

- When cross-arm correlation is weak, VAR2 and AR2 behave similarly.
- When clustered correlation is strong and noise is moderate, VAR2 beats AR2 and UCB1.
- When the transition matrix is estimated from enough calibration data, VAR2-Estimated approaches VAR2-Known.
- When correlation is dense and noisy, VAR2 can still help, but the advantage shrinks.

This motivates a stock-data track where the arms are real tickers and the reward path is historical return data.

## Stock Environment Design

The stock environment uses yfinance price data and converts it into reward vectors.

At each online step:

1. The hidden state is the vector of rewards across all selected tickers for the current date or period.
2. The agent chooses one ticker.
3. The agent observes only that ticker's reward.
4. The evaluator computes regret using the best ticker reward available in that same period.

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

VAR2 is only the first cross-arm AR2-style prototype. Future versions can add richer time-series structure.

Promising directions:

- ridge VAR2 for more stable transition estimation;
- sparse VAR2 or Lasso-style transition estimation;
- rolling-window VAR2 to adapt to changing market regimes;
- online transition estimation from bandit feedback;
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
- portfolio allocation;
- shorting and leverage;
- liquidity constraints;
- risk limits;
- survivorship-bias-safe universe construction.

These omissions are acceptable for the first research milestone because the purpose is algorithm comparison on shared historical reward paths, not live trading validation.
