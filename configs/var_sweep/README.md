# VAR Known/Unknown Sweep Configs

These configs test AR2 and VAR2 variants on correlated hidden VAR reward paths.

All configs compare:

- `VAR2-Known`: VAR2 using the true transition matrix `A`.
- `VAR2-Estimated`: VAR2 using least-squares `A_hat` from a separate calibration trajectory.
- `AR2-Known`: AR2 using only `diag(A)`.
- `AR2-Estimated`: AR2 using least-squares per-arm `alpha_hat`.
- `UCB1`, `Epsilon Greedy`, and `Random`.

Suggested command pattern:

```bash
python -m autoregressive_bandits.experiments.run \
  --config configs/var_sweep/<config>.json \
  --output outputs/<config>
```

The runner writes `README.md`, `scenario.json`, `summary.json`, `metrics.csv`, plots, and when applicable `estimated_parameters.json` into each output folder. It also updates `outputs/experiment_index.csv`.
