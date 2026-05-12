# Implementation Guide: Bandits with Temporal / Dynamical Structure

This document is a detailed, implementation-oriented walkthrough of three closely related bandit algorithms that all deal with **rewards that carry temporal structure** rather than being i.i.d. across rounds:

| # | Algorithm | Paper | PDF in project | Reference code |
|---|-----------|-------|----------------|----------------|
| 1 | **AR2**       | *Non-Stationary Bandits with Auto-Regressive Temporal Dependency* — Chen, Golrezaei, Bouneffouf (NeurIPS 2023) | `NonStationary_Bandits_with_AutoRegressive_Temporal_Dependency.pdf` | `code.ipynb` in the project folder (the authors' implementation) |
| 2 | **DynLin-UCB** | *Dynamical Linear Bandits* — Mussi, Metelli, Restelli (ICML 2023) | `dynamic_linear_bandits.pdf` | <https://github.com/marcomussi/DLB> |
| 3 | **AR-UCB**    | *Autoregressive Bandits* — Bacchiocchi, Genalti, Maran, Mussi, Restelli, Gatti, Metelli (AISTATS 2024) | `autoregressive_bandits.pdf` | <https://github.com/gianmarcogenalti/autoregressive-bandits> |

The intended consumer of this document is an automated code-generation agent (Codex / Claude Code). The goal is to provide enough notation, intuition, equations, pseudocode, and *deviations between paper and reference code* that the agent can implement each algorithm in a clean, reproducible Python codebase (with a shared environment simulator and a shared benchmarking harness) without going back to the PDFs.

> **Important:** Where the published pseudocode and the authors' released code disagree, this guide flags both and tells the implementer which to prefer. The cleanest reproducible behavior tends to follow the *released code* (since that is what produced the published figures), with the *paper's pseudocode* used as the theoretical reference.

---

## 0. Why these three algorithms belong together

All three works share a single conceptual core: **the reward at time `t` is not an i.i.d. draw — it depends on history**. A naive bandit (UCB1, EXP3, even Lin-UCB) suffers *linear* regret in these settings because it ignores history. The three papers differ in *what kind* of temporal structure they assume:

| Paper | What is non-i.i.d.? | Action set | Per-arm structure |
|---|---|---|---|
| **AR2** | Each arm has its **own** AR(1) chain on its expected reward; the chain is **exogenous** (pulling does not modify it) | Finite `k` arms | `r_i(t+1) = B(α · (r_i(t) + ε_i(t)))`, truncated to `[-L, L]`, `ε_i(t) ~ N(0,σ²)` |
| **DynLin-UCB** | A **hidden state** `x_t` evolves linearly under the chosen action; the reward is a linear functional of state + action | Continuous `u ∈ U ⊂ ℝ^d` | `x_{t+1} = A x_t + B u_t + ε_t`, `y_t = ⟨ω,x_t⟩ + ⟨θ,u_t⟩ + η_t` |
| **AR-UCB** | The reward itself is an **AR(k) chain** whose coefficients are *chosen by* the action — no hidden state | Finite `n` actions | `x_t = γ_0(a_t) + Σ_{i=1}^{k} γ_i(a_t) · x_{t-i} + ξ_t` |

Two axes distinguish them:

- **Who controls the dynamics?** In **AR2** nobody does (the action only chooses *which* AR chain we observe). In **AR-UCB** and **DynLin-UCB** the action chooses the dynamics themselves.
- **Is there a hidden state?** Only in **DynLin-UCB**.

The three algorithms then share UCB-style ideas but apply them very differently. The unifying principle: **construct a confidence set over the right "right quantity" for each setting** — the *latent reward* `r_i(t)` in AR2, the *cumulative Markov parameter* `h` in DLB, and the *per-action AR coefficient vector* `γ(a)` in ARB — and act optimistically with respect to that confidence set.

---

## 1. Common scaffolding (build this first)

Before implementing any algorithm, create a shared scaffolding so that all three live in the same codebase, share a simulator, and can be compared on the same plots.

### 1.1 Recommended file layout

```
bandits_temporal/
├── envs/
│   ├── __init__.py
│   ├── base.py              # abstract Environment
│   ├── ar1_env.py           # AR2 environment: independent AR(1) per arm with truncation
│   ├── arp_env.py           # AR(p)-with-trend environment (for the AR2-p case study)
│   ├── arb_env.py           # ARB environment: AR(k) with action-dependent coefficients
│   └── dlb_env.py           # DLB environment: LTI system with hidden state
├── agents/
│   ├── __init__.py
│   ├── base.py              # abstract Agent
│   ├── ar2.py               # AR2 and AR2-p
│   ├── dynlin_ucb.py        # DynLin-UCB
│   ├── ar_ucb.py            # AR-UCB
│   ├── baselines.py         # UCB1, EXP3, B-EXP3, Lin-UCB, D-Lin-UCB, ε-greedy,
│   │                        # SW-UCB, SW-TS, RExp3, mod-UCB, ETC, predictive sampling
│   └── ho_kalman.py         # Ho-Kalman identification (DLB real-world experiment only)
├── runners/
│   ├── run_ar2.py
│   ├── run_dynlin.py
│   ├── run_arucb.py
│   └── plotting.py
└── tests/
    ├── test_envs.py
    └── test_agents.py
```

### 1.2 The common `Environment` interface

Every environment exposes the same loop:

```python
class Environment:
    def reset(self, seed: int | None = None) -> None: ...
    # pull(action) returns the observed reward at the current round t.
    # The environment internally advances its state.
    def pull(self, action) -> float: ...
    # The expected reward of the optimal *dynamic* benchmark at the current round.
    # See each section for the exact definition (it differs across settings).
    def optimal_expected_reward(self) -> float: ...
    # Expected reward of a given action conditional on current internal state, used
    # to compute regret on the *expected* (not noisy) reward.
    def expected_reward(self, action) -> float: ...
    @property
    def t(self) -> int: ...   # current round, 1-based
```

**Crucial convention:** the environment is allowed to remember *every arm's* trajectory internally, so that regret can be measured against the dynamic benchmark, but the agent only sees the realized reward of the arm it pulled. For AR2 in particular, every arm's AR(1) chain advances every round regardless of whether it was pulled. The cleanest implementation pre-generates all `k × T` reward sequences at `reset()` time (as the authors do in `code.ipynb`); see §2.10.

### 1.3 The common `Agent` interface

```python
class Agent:
    def reset(self, seed: int | None = None) -> None: ...
    def select_action(self, t: int): ...
    def update(self, t: int, action, reward: float) -> None: ...
```

The training loop is then:

```python
env.reset(seed); agent.reset(seed)
inst_regret = np.zeros(T)
for t in range(1, T + 1):
    a       = agent.select_action(t)
    r_star  = env.optimal_expected_reward()      # dynamic benchmark
    r_exp   = env.expected_reward(a)             # E[r_t | F_{t-1}, a]
    r_noisy = env.pull(a)                        # the noisy observation given to the agent
    agent.update(t, a, r_noisy)
    inst_regret[t-1] = r_star - r_exp
cum_regret = np.cumsum(inst_regret)
```

Always compute regret on **expected** reward (`r_star - r_exp`), not the noisy realization (`r_star - r_noisy`). This makes curves smooth and reflects what the *algorithm* did rather than what the noise did.

### 1.4 Seeding and reproduction

- All randomness must go through a `numpy.random.Generator` instance (`rng = np.random.default_rng(seed)`) stored on both the environment and the agent. **Never call `np.random` globally.** The legacy notebook (`code.ipynb`) does, which makes its results non-reproducible across cells; do not copy that convention.
- Run each experiment with at least 50–100 independent seeds and report mean ± std (the paper conventions: AR2 uses 100, DLB and ARB use 50–100 depending on the experiment).
- For each `(algorithm, seed)` pair, use a **fresh** environment seeded with the *same* seed used for the algorithm so that two algorithms see identical reward streams (paired-comparison setup; reduces variance dramatically).

---

## 2. Algorithm 1 — AR2 (Alternating and Restarting AR-Bandit)

> **Read in detail:** Sections 2 (setup), 4 (algorithm), 5 (regret bound), Appendix B (AR-p extension and implementation notes), Appendix C (mod-UCB baseline) of `NonStationary_Bandits_with_AutoRegressive_Temporal_Dependency.pdf`. **Also read the AR2 function in `code.ipynb` — there are non-trivial differences from the paper's pseudocode** (see §2.6 and §2.9).

### 2.1 Problem setup

There are `k` arms. Each arm `i ∈ [k]` carries its own *latent* expected reward `r_i(t)` that evolves **independently of any agent action** as an AR(1) chain with a symmetric truncating boundary:

```
r_i(t+1) = B( α · ( r_i(t) + ε_i(t) ) ),     ε_i(t) ~ N(0, σ²),  i.i.d. across i and t
B(y)     = min(max(y, -L), +L)
```

(The paper uses `ℛ` for the boundary; the reference code uses `L`. We use `L` throughout for consistency with the code.)

The agent observes only `R_{I_t}(t) = r_{I_t}(t) + ε_{I_t}(t)`, the *noisy* realization of the pulled arm. The other arms' chains keep evolving silently.

**Hyperparameters of the environment:**
- `α ∈ (0, 1)`: AR coefficient (proxy for temporal correlation). Larger ⇒ slower change. In the heterogeneous setting (Appendix A/B.1, which is what `code.ipynb` actually runs), each arm has its own `α_i` and `σ_i`.
- `σ ∈ (0, 1)`: stochastic rate of change (per-step innovation s.d.).
- `L > 0`: truncating boundary, typically `L = 1`.

**Assumptions the analysis relies on:**
1. The agent **knows `α`** (or pre-estimates it via MLE on offline data; see Section 8 of the paper).
2. Innovations are Gaussian; truncation is symmetric.

**Performance metric — per-round steady-state dynamic regret** (Definition 2.1):

```
REG = lim sup_{T → ∞}  (1/T) · E[ Σ_{t=1}^T (r*(t) − r_{I_t}(t)) ],     r*(t) := max_i r_i(t)
```

For implementation, just compute the cumulative pseudo-regret `Σ (r*(t) − r_{I_t}(t))` and divide by the total optimal reward to get a **normalized regret** in `[0, 1]` — this is what Table 2 reports:

```
normalized_regret = Σ_t (r*(t) − r_{I_t}(t)) / Σ_t r*(t)        # see compute_regret_percentage in code.ipynb
```

### 2.2 Key insight (why AR2 is built the way it is)

In standard MABs, *over-exploration is fine asymptotically* — you eventually identify the best arm and exploit. In an AR(1) world, the "best" arm changes constantly. Two coupled tensions emerge:

1. **Exploration vs. exploitation.** Unpulled arms keep evolving, so any stale estimate decays geometrically (factor `α` per step). A standard UCB bonus makes any arm not pulled for ~`O(1/√(1−α²))` rounds look attractive ⇒ over-exploration ⇒ no exploitation gets done.
2. **Remembering vs. forgetting.** Old observations become useless quickly (information about `r_i(t')` is worth only `α^{t−t'}` for predicting `r_i(t)`). The algorithm must throw old information away at some point.

AR2 resolves both with two structural devices:

1. **Alternation.** *Force* exploitation at least every other round (even rounds always exploit). Exploration only happens on **odd** rounds, and only if at least one *triggered* arm exists. This caps the exploration rate at 50%, regardless of how wide the confidence intervals get.
2. **Restart in epochs.** Every `Δ_ep` rounds, wipe memory and re-pull each arm once. `Δ_ep = ⌈k · α⁻³ · σ⁻³⌉` is the schedule from Theorem 5.2.

### 2.3 The estimator (geometric decay of stale samples)

When arm `i` was last pulled at round `τ_i` and observed `R_i(τ_i)`, AR2's running estimate of `E[r_i(t) | history]` at any later round `t ≥ τ_i + 1` is

```
r̂_i(t) = α^{t − τ_i − 1} · B( α · R_i(τ_i) )
```

**Intuition:** `E[r_i(τ_i+1) | r_i(τ_i)] ≈ α · r_i(τ_i)` (ignoring truncation), and each unobserved step shrinks the expectation by another factor of `α`. The `B(·)` is applied once at the one-step prediction.

**Maintenance is purely recursive** (this is what you implement):

```
When arm i is pulled at t:        r̂_i(t+1) := B(α · R_i(t))     # fresh anchor from observation
When arm i is NOT pulled at t:    r̂_i(t+1) := α · r̂_i(t)        # geometric decay
```

In the reference code, the per-arm estimates are stored in a single vector `gamma[i]` (one scalar per arm); every round, **all** `gamma[i]` are multiplied by `α_i`, and the pulled arm's entry is overwritten with `B(α_i · time_series[i_t, t])`. This is much cleaner than tracking `(τ_i, R_i(τ_i))` pairs.

### 2.4 The triggering criterion (confidence interval)

Define the confidence width for arm `i` at round `t`:

```
w_i(t) := σ_i · √( (α_i² − α_i^{2(t − τ_i)}) / (1 − α_i²) )           (if α_i < 1)
        := σ_i · √( t − τ_i − 1 )                                      (if α_i = 1, degenerate)
```

The paper writes the exponent as `2(t − τ_i + 1)` whereas the code uses `2(t − τ_i)`. These differ by a factor of `α²` inside the square root — practically negligible; **prefer the code's `2(t − τ_i)`** to match published experimental results.

By Hoeffding's inequality on the weighted Gaussian sum, with probability at least `1 − δ`:

```
| r_i(t) − r̂_i(t) | ≤ √(2 log(2/δ)) · w_i(t)
```

AR2 does **not** use this raw UCB to choose arms. Instead it uses the same width to decide whether an arm is *worth triggering*. Let `i_sup(t)` be the current "superior" arm and `r̂_sup(t) := r̂_{i_sup(t)}(t)` its estimate. For each arm `i ∉ T ∪ {i_sup(t)}`, **trigger** arm `i` if

```
r̂_sup(t) − r̂_i(t)  ≤  c_1 · w_i(t)                                   (*)
```

If triggered, `T ← T ∪ {i}` and trigger time `τ_i^{trig} := t`.

The constant `c_1 = c_const · c_0` ties theory and code: see §2.6.

### 2.5 The "superior arm"

The paper defines `i_sup(t)` as the better-estimated of the **two most recently pulled** arms:

```
i_sup(t) = argmax( r̂_{I_{t-1}}(t), r̂_{I_{t-2}}(t) )
```

The paper itself notes (and the reference code adopts) that taking `argmax_i r̂_i(t)` **over all arms** gives the same theoretical guarantee and performs better empirically. **Implement both as a flag, default to "argmax over all arms".**

### 2.6 Constants `c_0` and `c_1`: paper vs. code

This is the single most important discrepancy to be aware of:

| | `c_0` | `c_1` |
|---|---|---|
| Paper (Theorem 5.2) | `√( 4 log(1 / (α σ)) + 4 log Δ_ep + 2 log(4k) )` | `24 · c_0` |
| Reference code (`code.ipynb`, `AR2(...)`) | passed in as a hyperparameter, tuned (e.g. `0.001`) | `8 · c_0` |

**Why:** Theoretical constants in self-normalized concentration bounds are notoriously loose. In experiments the authors treat `c_1` as a tuning knob. Implement it as a hyperparameter `c_const` (default `8`) so the agent can search over it. The theoretical formula for `c_0` is what you'd use if you wanted to certify the regret bound; in practice tune `c_0` over a small grid.

### 2.7 Alternation rule

After the triggering pass:

- If `t` is **odd** and `T ≠ ∅`: explore. Pick a triggered arm and remove it from `T`.
  - **Paper rule:** earliest trigger time, `I_t = argmin_{j ∈ T} τ_j^{trig}`.
  - **Code rule (when `criteria="UCB"`):** pick `argmax_{j ∈ T} UCB_j(t)` where `UCB_j(t) = r̂_j(t) + c_1 · w_j(t)`, *but only* if `UCB_j(t) > UCB_{i_sup}(t)`; otherwise fall back to `i_sup`. This is the variant that produced Table 2.
- Otherwise (`t` even, or `T = ∅`): exploit. `I_t = i_sup`.

Implement both rules as a flag, default to `"UCB"` (the code's default for the published Table 2).

### 2.8 Full pseudocode (faithful synthesis of paper + reference code)

```
Inputs:
  α = (α_1, …, α_k)       # AR parameters per arm (assumed known or pre-estimated)
  σ = (σ_1, …, σ_k)       # innovation sd per arm
  Δ_ep                     # epoch length; default ⌈k · ᾱ⁻³ · σ̄⁻³⌉ with averages,
                           # or treat as a tunable; the code's e2e runs use full T (1 epoch)
  L                        # truncation boundary, default 1.0
  c_0                      # base confidence constant; default 0.001 (tuned in code) or
                           # theoretical √(4 log(1/(α σ)) + 4 log Δ_ep + 2 log(4 k))
  c_const                  # multiplier; default 8 (code) or 24 (paper)
  superior_rule ∈ {"all", "last_two"}    # default "all"
  trigger_pick ∈ {"earliest", "UCB"}     # default "UCB"

  c_1 ← c_const · c_0
  S   ← ⌈T / Δ_ep⌉

For s = 1, …, S:
    t0 ← (s − 1) · Δ_ep
    T  ← ∅                                                # triggered set
    # ----- Initialization phase: one pull per arm
    For i = 1, …, k:
        Pull arm i at round t0 + i, observe R_i(t0 + i)
        τ_i ← t0 + i
    # The published Algorithm 1 sets, at round t0+k+1:
    #     r̂_i(t0+k+1) = α^{k − i} · B(α · R_i(t0 + i))
    # The reference code uses (with t = k, 0-indexed):
    #     gamma[i] = α_i^{k − i} · time_series[i, i]            (note: NO B, NO α multiplier)
    # The simplest correct convention is the paper's; the code variant works because
    # the truncation is applied later. Use the paper's formula for correctness.

    # ----- Main inner loop
    For t = t0 + k + 1, …, min(t0 + Δ_ep, T):
        # 1) Superior arm
        if superior_rule == "all":
            i_sup ← argmax_{i ∈ [k]} r̂_i(t)
        else:  # "last_two"
            i_sup ← argmax( r̂_{I_{t-1}}(t), r̂_{I_{t-2}}(t) )
        r̂_sup ← r̂_{i_sup}(t)

        # 2) Trigger pass
        For each arm i ∉ T ∪ {i_sup}:
            w_i ← σ_i · sqrt( (α_i² − α_i^{2(t − τ_i)}) / (1 − α_i²) )    # see §2.4
            if r̂_sup − r̂_i(t) ≤ c_1 · w_i:
                T ← T ∪ {i}
                τ_i^{trig} ← t

        # 3) Action selection (alternation)
        if (t is odd) and (T ≠ ∅):
            if trigger_pick == "earliest":
                I_t ← argmin_{j ∈ T} τ_j^{trig}
                T ← T \ {I_t}
            else:  # "UCB"
                j_best ← argmax_{j ∈ T} ( r̂_j(t) + c_1 · σ_j · sqrt((α_j² − α_j^{2(t-τ_j)}) / (1 − α_j²)) )
                ucb_isup ← r̂_sup + c_1 · σ_{i_sup} · sqrt((α_{i_sup}² − α_{i_sup}^{2(t-τ_{i_sup})}) / (1 − α_{i_sup}²))
                if (r̂_{j_best} + c_1 * w_{j_best}) > ucb_isup:
                    I_t ← j_best
                    T ← T \ {I_t}
                else:
                    I_t ← i_sup
        else:
            I_t ← i_sup
            if i_sup ∈ T: T ← T \ {i_sup}      # safety; an arm chosen for exploit leaves T

        # 4) Pull, observe, update
        Pull arm I_t, observe R_{I_t}(t)
        τ_{I_t} ← t

        # Update estimates: decay everyone, overwrite the pulled arm
        For each arm i:
            r̂_i(t+1) ← α_i · r̂_i(t)
        r̂_{I_t}(t+1) ← B( α_{I_t} · R_{I_t}(t) )
```

### 2.9 Notable subtleties from `code.ipynb`

1. **The "two-step" alpha factor.** The very first time we look at an arm after the init pull, its estimate is `α^{k−i} · R_i(t0+i)` (no inner `α`!) in the code, whereas the paper writes `α^{k−i} · B(α · R_i(t0+i))`. The mathematical difference is small and the code's choice is fine in practice. Use the paper's form for analytical compatibility.
2. **One global epoch in practice.** Although the algorithm supports `S = ⌈T/Δ_ep⌉` epochs, all the experiments in `code.ipynb` run with `Δ_ep = T` (i.e., no restart). The restart is mostly a *theoretical* device to control the worst-case regret; empirically the unrestarted algorithm wins as long as `α < 1`. Expose `Δ_ep` as a hyperparameter and default to `T` (single epoch).
3. **"Activation set" vs. "triggered set".** The code uses the variable name `V`; the paper uses `T`. Same thing.
4. **`alpha[i] == 1` branch.** Both the code and any faithful implementation need to special-case `α_i = 1` (the formula `(α² − α^{2(t-τ)})/(1−α²)` is `0/0`). Use the limit `t − τ − 1` in that case (see §2.4).
5. **Edge case at the first two iterations.** For "last_two" mode at `t = t0 + k + 1` and `t0 + k + 2`, `I_{t-1}` and `I_{t-2}` reference rounds inside the init phase, so set `I_{t0+k} = k` and `I_{t0+k-1} = k-1`. This is implicit in the code.

### 2.10 The AR2 environment (translation of `generate_time_series` from `code.ipynb`)

```python
class AR1Env(Environment):
    """Independent AR(1) chain per arm with symmetric truncation.

    State `mu[i]` is the latent expected reward of arm i at the current round.
    Observed reward is mu[i] + N(0, sigma_i).
    """
    def __init__(self, T, k, alpha, sigma, L=1.0):
        self.T, self.k = T, k
        self.alpha = np.asarray(alpha, float)         # shape (k,)
        self.sigma = np.asarray(sigma, float)         # shape (k,)
        self.L = L

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        T, k, alpha, sigma, L = self.T, self.k, self.alpha, self.sigma, self.L
        # Pre-generate all latent (mu) and observed (X) trajectories.
        mu  = np.zeros((k, T + 1))
        X   = np.zeros((k, T + 1))
        mu[:, 0]  = rng.uniform(-L, L, k)
        X[:, 0]   = mu[:, 0] + rng.normal(0, sigma)
        for t in range(1, T + 1):
            mu[:, t] = np.clip(alpha * X[:, t-1], -L, L)
            X[:, t]  = mu[:, t] + rng.normal(0, sigma)
        self.mu = mu
        self.X  = X
        self._t = 0

    def pull(self, a):
        # Observed reward at the current round, then advance.
        r = self.X[a, self._t]
        self._t += 1
        return r

    def expected_reward(self, a):
        return self.mu[a, self._t]

    def optimal_expected_reward(self):
        return self.mu[:, self._t].max()
```

This mirrors `generate_time_series` from `code.ipynb` exactly, except (a) all arms' trajectories are pre-generated (deterministic given seed; cleaner), and (b) we go through a `numpy.random.Generator` instead of `np.random.seed`.

### 2.11 AR(p) extension — `AR2-p` (Algorithm 2, Appendix B.2)

For each arm `i`, `r_i(t) = α_{i,0} + Σ_{j=1}^{p} α_{i,j} · R_i(t − j)`. Maintain **two** parallel recursions, one for the reward estimate and one for an error-bound estimate:

```
r̂_i(t) = α_{i,0} + Σ_{j=1}^{p} α_{i,j} · r̂_i(t − j)
Ê_i(t) = Σ_{j=1}^{p} α_{i,j} · ( Ê_i(t − j) + 1 )
```

Differences from AR(1):
- **Init:** `r̂_i(t) = 0`, `Ê_i(t) = ∞` for `t ≤ 0`. Init phase pulls each arm consecutively for `p` rounds (not 1) so the AR(p) recursion has p observations to seed.
- **Triggering:** `r̂_sup(t) − r̂_i(t) ≤ c · σ_i · √Ê_i(t)`.
- **Exploration pick:** `I_t = argmax_{i ∈ T} ( r̂_i(t) + c · σ_i · √Ê_i(t) )` — UCB-style, not earliest-triggered.
- **On pull:** *reset* the pulled arm's estimate to the observation and the error to zero:
  `r̂_{I_t}(t) ← R_{I_t}(t)`, `Ê_{I_t}(t) ← 0`.

Only implement AR2-p if you reproduce the tourism-demand case study in Section 7 of the paper.

### 2.12 Other baselines you'll need (all in `code.ipynb` already)

- **ETC** (explore-then-commit): pull each arm `m` times, then commit to the empirical winner. Default `m = 100`.
- **UCB1** (Auer et al. 2002): `UCB_i = mean_i + √(2 log(t+1) / n_i)`. Note: in non-stationary, "mean" is the running mean over all pulls; the wrong baseline because it never forgets.
- **ε-greedy**: act greedily on `α^{d_i} · last_obs_i` with probability `1 − ε`, else random. The "greedy" estimate uses the AR-aware decay.
- **mod-UCB** (Appendix C of the paper, Algorithm 3 there): same as UCB1 but with the AR-aware confidence bound `√(2 log(1/δ)) · σ_i · √((α_i² − α_i^{2(t−τ_i)})/(1−α_i²))`. **Always argmax over all arms** (no alternation). This is AR2's nearest competitor.
- **SW-UCB** (sliding-window UCB): use only observations in the last `τ` rounds; bonus `B √(ξ log τ / N_t)`. Default `ξ = 0.8`, `τ = ⌈2L √(T log T / num_period)⌉`.
- **SW-TS** (sliding-window Thompson Sampling): same windowing, Beta posterior on the rescaled-to-[0,1] reward.
- **RExp3** (Besbes et al. 2014): EXP3 run in batches of size `⌈(K log K)^{1/3} (T/V)^{2/3}⌉` with `V = 0.05 T` (a "variation budget").

All of these are short (10–30 lines each); see `code.ipynb` cells 4–10 for verbatim implementations.

### 2.13 Estimating `α` from data (Section 8, only if `α` is unknown)

Pull arm `i` consecutively for `T_est = O(T^β)` rounds (β ≈ 1/2 is fine), then maximize the *truncated AR(1)* log-likelihood over `α ∈ (0, 1)`. Use `scipy.optimize.minimize_scalar(method='bounded')`. **Do NOT use OLS** on the raw `R(t+1) ≈ α · R(t)` recursion — truncation biases OLS. The MLE handles truncation correctly.

### 2.14 Sanity checks

- **`α → 0`:** the AR chains become almost i.i.d.; AR2 should still match (≈) a standard MAB algorithm. The paper shows AR2 still beats benchmarks in this regime; reproduce this.
- **`α → 1`:** chains become slow random walks; per-round regret should scale like `α² σ² k` (Theorem 5.2).
- **Reproduce Table 2:** for `k ∈ {2, 10, 20}`, `E[α_i] ∈ {0.4, 0.9}`, AR2 should beat all baselines on normalized regret.

---

## 3. Algorithm 2 — DynLin-UCB (Dynamical Linear Bandits)

> **Read in detail:** Sections 2 (setup), 3 (algorithm) of `dynamic_linear_bandits.pdf`. Reference code at <https://github.com/marcomussi/DLB> (`dynlinucb/` folder; configs in `config/`; entry point `scripts/main.py`).

### 3.1 Problem setup — a hidden-state LTI bandit

```
x_{t+1} = A x_t + B u_t + ε_t          # x_t ∈ ℝⁿ is HIDDEN (not observed by the agent)
y_t     = ⟨ω, x_t⟩ + ⟨θ, u_t⟩ + η_t     # y_t ∈ ℝ is the (noisy) reward
u_t     ∈ U ⊂ ℝᵈ                        # action; vector-valued, continuous
```

with `ε_t, η_t` zero-mean σ²-subgaussian and `A, B, ω, θ` unknown. The action set `U` is an arbitrary bounded subset of `ℝᵈ`. The synthetic experiment uses a budget simplex `{u ∈ [0,1]³ : Σ u_i ≤ 1.5}`.

**Two structural assumptions:**
- **Stability (2.1):** `ρ(A) < 1` and `Φ(A) := sup_{τ ≥ 0} ‖A^τ‖₂ / ρ(A)^τ < ∞`. The agent gets to know `ρ̄ ≥ ρ(A)` and `Φ̄ ≥ Φ(A)` — these *cannot* be avoided (Theorem 2.2).
- **Boundedness (2.2):** `‖θ‖₂ ≤ Θ`, `‖ω‖₂ ≤ Ω`, `‖B‖₂ ≤ B̄`, `sup_{u ∈ U}‖u‖₂ ≤ Ū`, `sup_{x}‖x‖₂ ≤ X̄`, `sup_u |J(u)| ≤ 1`.

### 3.2 The key quantity: cumulative Markov parameter `h`

For a **constant** action `u`, the steady-state expected reward is

```
J(u) := lim_{H→∞} (1/H) Σ_{t=1}^H E[y_t]  =  ⟨h, u⟩,    where
   h := θ + Bᵀ (Iₙ − A)⁻ᵀ ω    ∈ ℝᵈ
```

(`h` is well-defined because of stability.) Theorem 2.1 then says:

> Under stability + boundedness, the optimal infinite-horizon policy is the **constant** policy `u* = argmax_{u ∈ U} ⟨h, u⟩`.

Two important consequences:

1. **The DLB optimal policy is open-loop.** No need to condition on history. You just need to estimate `h`.
2. **A myopic agent that maximizes `⟨θ, u⟩` is wrong.** The myopic objective ignores how state propagates; Lin-UCB and D-Lin-UCB both fall into this trap (the synthetic experiment in §5.1 shows this clearly: `θ = (0, 0.5, 0.1)` but `h = (0.56, 0.5, 0.11)`, and the optimal `u*` invests in coordinate 1, not coordinate 2).

### 3.3 Key insight (why DynLin-UCB persists actions)

You cannot use a single `y_t` as a clean label for `⟨h, u_t⟩` — `y_t` is corrupted by the transient `ωᵀ A^t x_0 + ωᵀ Σ A^{s−1} ε_{t−s}`, which only decays at rate `ρ(A) < 1`. So the algorithm **persists** a chosen action long enough for the system to (approximately) reach steady state, **then** records one (almost-unbiased) sample.

The persistence length grows **logarithmically** in the epoch index. In epoch `m`, the agent keeps the chosen action constant for

```
H_m = ⌊ log(m) / log(1/ρ̄) ⌋
```

extra rounds (so the epoch has `1 + H_m` rounds in total). Early epochs are short (waste-minimizing when estimates are bad); later epochs are long (low-bias when estimates are good). Crucially this schedule only needs `ρ̄`, not `T`.

The algorithm then runs Lin-UCB *at the epoch level*, using only the **last** observation of each epoch as a Ridge regression sample with feature `u` (the action taken throughout the epoch) and target `y_last`.

### 3.4 Exploration coefficient `β_t`

From the self-normalized concentration in Theorem 3.1:

```
β_t  :=  c_1 / √λ · log(e (t + 1))
       + c_2 √λ
       + √( 2 σ̃² · ( log(1/δ) + (d/2) · log(1 + t Ū² / (d λ)) ) )

c_1  :=  Ū · Ω · Φ̄ · ( Ū · B̄ / (1 − ρ̄)  +  X̄ )
c_2  :=  Θ + Ω · B̄ · Φ̄ / (1 − ρ̄)
σ̃²  :=  σ² · ( 1 + Ω² Φ̄² / (1 − ρ̄²) )
δ    :=  1 / T
```

Use `λ = log T` as default (paper's empirical choice — see §5).

### 3.5 Pseudocode (Algorithm 1 of the paper, expanded with implementation details)

```
Inputs:
  λ                                 # Ridge regularization; default log T
  ρ̄ ∈ [0, 1)                       # upper bound on spectral radius
  Φ̄, B̄, Ū, X̄, Θ, Ω                # bounds (Assumption 2.2)
  σ                                  # noise sd
  T                                  # horizon
  U_action ⊂ ℝᵈ                     # action set (typically discretized; see §3.6)

Derived:
  σ̃²  = σ² (1 + Ω² Φ̄² / (1 − ρ̄²))
  c_1 = Ū Ω Φ̄ (Ū B̄ / (1 − ρ̄) + X̄)
  c_2 = Θ + Ω B̄ Φ̄ / (1 − ρ̄)

Initialize:
  V    ← λ · I_d         # d × d Gram matrix
  b    ← 0_d             # d-vector
  ĥ    ← 0_d
  t    ← 1
  m    ← 1

Loop (until total rounds elapsed ≥ T):
    # ----- Beginning of epoch m -----
    # 1) Compute β at the START of the epoch (using current t)
    β ← c_1/√λ · log(e(t+1)) + c_2 √λ
         + √( 2 σ̃² · (log(1/δ) + (d/2) log(1 + t Ū² / (d λ))) )

    # 2) Optimistic action
    u_m ← argmax_{u ∈ U_action}  ⟨ĥ, u⟩ + β · √(uᵀ V⁻¹ u)

    # 3) Persist for 1 + H_m rounds
    H_m ← ⌊ log(m) / log(1/ρ̄) ⌋
    y_first ← play(u_m); t ← t + 1            # round 1 of epoch; observation DISCARDED
    For j = 2, …, H_m + 1:
        y_last ← play(u_m); t ← t + 1         # observations 2..H_m discarded;
                                              # only the last one is used.
    # (When H_m = 0, the single round IS the last round.)

    # 4) Ridge update using ONLY (u_m, y_last)
    V ← V + u_m · u_mᵀ
    b ← b + u_m · y_last
    ĥ ← V⁻¹ b
    m ← m + 1
```

**The single most common bug** is updating `V, b` on every round inside the epoch. *Only the last round of each epoch produces a Ridge sample.*

### 3.6 The argmax over `U_action`

If `U_action` is a finite discrete set, scan it directly.

If `U_action` is a continuous, polyhedral set (e.g. a budget simplex), the inner UCB problem
`maximize ⟨ĥ, u⟩ + β · √(uᵀ V⁻¹ u)  s.t.  u ∈ U_action`
is the maximization of a **convex** function (the UCB term is a Euclidean-style norm) over a convex feasible set — generally NP-hard. Standard practice:

1. **Vertex enumeration + uniform grid.** For polyhedral `U_action` with `d ≤ 5`, enumerate the vertices of the feasible polytope plus a moderately fine internal grid (e.g. 50–200 points per dimension after restricting to the affine hull of `U_action`).
2. **Random search** (cheap fallback). Sample `M ~ 10⁴` random points uniformly from `U_action` and take the argmax. Acceptable for `d ≤ 10`.

The paper's reference code uses approach (1) (the synthetic experiment has `d = 3`).

### 3.7 Hyperparameter / implementation notes

- **`λ = log T`** is recommended; `λ = 1` is also reported. Avoid `λ = 0`.
- **`ρ̄` matters a lot.** Setting `ρ̄ = 0` reduces DynLin-UCB to Lin-UCB and gives linear regret in the experiments. Setting `ρ̄ ≫ ρ(A)` only slows convergence; setting `ρ̄ ≪ ρ(A)` can prevent learning. **When in doubt, overestimate `ρ̄`.** The empirical study (Figure 2) shows DynLin-UCB stays sublinear even with `ρ̄ = 2 ρ(A)`.
- **First epoch.** `H_1 = ⌊log 1 / log(1/ρ̄)⌋ = 0`, so epoch 1 is a single round and the algorithm degenerates to Lin-UCB at the very start. This is intended.
- **Epoch count.** The algorithm is *anytime*: you don't precompute `M`, just keep cranking out epochs until `t > T` and truncate the last (incomplete) epoch.
- **Sherman–Morrison for `V⁻¹`.** Maintain `V⁻¹` directly: `V⁻¹ ← V⁻¹ − (V⁻¹ u uᵀ V⁻¹) / (1 + uᵀ V⁻¹ u)`. Saves `O(d³)` per epoch.

### 3.8 The DLB environment simulator

```python
class DLBEnv(Environment):
    """
    x_{t+1} = A x_t + B u_t + eps_t
    y_t     = omega^T x_t + theta^T u_t + eta_t

    NOTE on timing: the convention used in the paper (and code) is that the reward
    y_t is observed BEFORE the state advances (so y_t depends on x_t and u_t).
    """
    def __init__(self, A, B, omega, theta, sigma, x0=None):
        self.A, self.B = A, B
        self.omega, self.theta = omega, theta
        self.sigma = sigma
        self.n, self.d = B.shape
        self.x0 = np.zeros(self.n) if x0 is None else x0.copy()
        I = np.eye(self.n)
        # Cumulative Markov parameter (Eq. 4 in the paper)
        self.h = theta + B.T @ np.linalg.solve((I - A).T, omega)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.x = self.x0.copy()
        self._t = 0

    def pull(self, u):
        eta = self.rng.normal(0, self.sigma)
        y   = self.omega @ self.x + self.theta @ u + eta
        eps = self.rng.normal(0, self.sigma, size=self.n)
        self.x = self.A @ self.x + self.B @ u + eps
        self._t += 1
        return y

    def expected_reward(self, u):
        # Steady-state expected reward of playing the *constant* action u
        # (used for regret in the constant-policy benchmark)
        return float(self.h @ u)

    def optimal_expected_reward(self, U_action):
        return max(self.expected_reward(u) for u in U_action)
```

**Regret convention.** Because the optimal policy is constant and equal to `argmax_u ⟨h, u⟩` with value `J* := max_u ⟨h, u⟩`, define cumulative regret as `T · J* − Σ y_t` (Equation 3). The expected version is `T · J* − Σ E[y_t]`; you can compute `E[y_t]` deterministically once you track `E[x_t]` (drop noises from the recursion).

### 3.9 Synthetic experiment (Section 5.1) — reproduce this

```
A = diag(0.2, 0, 0.1)
B = diag(0.25, 0, 0.1)
θ = (0, 0.5, 0.1)
ω = (1, 0, 0.1)
σ = 0.01
U = {u ∈ [0,1]³ : u₁ + u₂ + u₃ ≤ 1.5}
ρ(A) = 0.2; Φ(A) = 1; h = (0.56, 0.5, 0.11)
optimal action u* = (1, 0.5, 0); J* = 0.81
T = 10⁶, 50 seeds
```

The reference plot has DynLin-UCB sublinear and all other algorithms linear.

### 3.10 Optional: Ho-Kalman identification (Appendix D)

Only relevant if you want to learn `(A, B, ω, θ)` from raw advertising data. Workflow:

1. Build the lifted regressor `φ_t = [y_{t-1}, …, y_{t-H}, u_t, …, u_{t-H}]ᵀ`.
2. Solve regularized least squares for the Markov-parameter matrix `Ĝ_y` (Eq. 27 in Appendix D).
3. Remove the `D` block to recover `Ĝ_{ỹ}`, then apply Ho-Kalman realization to recover `(Â, B̂, ω̂, θ̂)`.

This is **not** part of the bandit algorithm itself — it produces the simulator for the real-world experiment.

---

## 4. Algorithm 3 — AR-UCB (Autoregressive Bandits)

> **Read in detail:** Sections 2 (setup), 3 (AR-UCB algorithm), 4 (regret analysis), 5 (experiments), Appendix D (misspecification of `k` and `m`) of `autoregressive_bandits.pdf`. Reference code at <https://github.com/gianmarcogenalti/autoregressive-bandits>.

### 4.1 Problem setup

There are `n` discrete actions `a ∈ [n]`. The reward is a **single scalar AR(k) chain** whose coefficients are *chosen by the agent's action at the current round*:

```
x_t  =  γ_0(a_t) + Σ_{i=1}^{k} γ_i(a_t) · x_{t-i} + ξ_t,    ξ_t  is σ²-subgaussian
     =  ⟨γ(a_t), z_{t-1}⟩ + ξ_t

where
   z_{t-1} := (1, x_{t-1}, x_{t-2}, …, x_{t-k})ᵀ ∈ ℝ^{k+1}     # "state": 1 + last k rewards
   γ(a)    := (γ_0(a), γ_1(a), …, γ_k(a))ᵀ ∈ ℝ^{k+1}          # per-action AR coefficients
```

**Contrast with the other settings:** in ARB there is **one shared reward chain**, not one per arm; the agent's action **modulates how that chain evolves**.

**Assumption 1** (parameter conditions):
- **(a) Non-negative coefficients:** `γ_i(a) ≥ 0` for all `i, a`. This rules out sign-flipping AR processes and — crucially — makes the **myopic** policy optimal (see §4.2).
- **(b) Stability:** `Γ := max_a Σ_{i=1}^{k} γ_i(a) < 1`. **`Γ` need NOT be known to the algorithm.**
- **(c) Boundedness:** `m := max_a γ_0(a) < ∞`. The algorithm only needs an upper bound `m̄ ≥ m`.

### 4.2 Key insight (Theorem 1): the optimal policy is myopic

Because `γ_i(a) ≥ 0`, the action that maximizes the *expected immediate reward* `E[x_t | H_{t-1}] = ⟨γ(a), z_{t-1}⟩` also maximizes the expected cumulative reward. So

```
π*(H_{t-1}) ∈ argmax_a ⟨γ(a), z_{t-1}⟩
```

This is *non-Markovian with memory `k`* (you need the last `k` rewards) but *stationary* in the augmented state `z_{t-1}`. So the algorithm just needs to (i) estimate `γ(a)` per action, (ii) act greedily-with-optimism w.r.t. the current `z_{t-1}`.

### 4.3 Per-action Ridge regression

For each action `a`, maintain its own Gram matrix `V_t(a) ∈ ℝ^{(k+1)×(k+1)}` and target vector `b_t(a) ∈ ℝ^{k+1}`. Only the action played at round `t` gets updated:

```
If a_t = a:
    V_t(a)  ← V_{t-1}(a) + z_{t-1} z_{t-1}ᵀ
    b_t(a)  ← b_{t-1}(a) + z_{t-1} · x_t
    γ̂_t(a) ← V_t(a)⁻¹ b_t(a)
Else:
    V_t(a), b_t(a), γ̂_t(a)  unchanged
```

Initialization: `V_0(a) = λ · I_{k+1}`, `b_0(a) = 0`, `γ̂_0(a) = 0`. And `z_0 = (1, 0, …, 0)ᵀ` (assumes the AR process starts "at zero"; if not, play any action for the first `k` rounds to bootstrap `z_k`).

### 4.4 Exploration coefficient (Lemma 2)

```
β_t(a) :=  √( λ · (m̄² + 1) )
         + σ · √( 2 log(n/δ) + log( det V_t(a) / λ^{k+1} ) )
```

**Notes:**
- `m̄ ≥ m` is supplied to the agent. Section 5.3 of the paper shows AR-UCB is robust to choosing `m̄` of the same order of magnitude as `m`.
- `det V_t(a)` is **per-action**, which makes the bonus action-specific — unlike Lin-UCB, where the bonus only depends on the shared Gram matrix.
- `Γ` is **not** required.

### 4.5 Pseudocode (Algorithm 1 of the paper)

```
Inputs:
  λ > 0                 # Ridge regularization; default 1
  k                     # AR order (or upper bound k̄ ≥ k)
  m̄ ≥ m                # upper bound on max_a γ_0(a)
  σ                     # subgaussian parameter of ξ_t
  n, T
  δ                     # confidence; default 1/T

For each a ∈ [n]:
    V_0(a) ← λ · I_{k+1}
    b_0(a) ← 0
    γ̂_0(a) ← 0
z_0 ← (1, 0, …, 0)ᵀ

For t = 1, …, T:
    For each a ∈ [n]:
        β_t(a) ← √(λ (m̄² + 1)) + σ · √( 2 log(n/δ) + log( det V_{t-1}(a) / λ^{k+1} ) )
        UCB_t(a) ← ⟨γ̂_{t-1}(a), z_{t-1}⟩ + β_t(a) · √( z_{t-1}ᵀ V_{t-1}(a)⁻¹ z_{t-1} )

    a_t ← argmax_a UCB_t(a)
    Play a_t; observe x_t = ⟨γ(a_t), z_{t-1}⟩ + ξ_t

    # Per-action Ridge update
    V_t(a_t) ← V_{t-1}(a_t) + z_{t-1} z_{t-1}ᵀ
    b_t(a_t) ← b_{t-1}(a_t) + z_{t-1} · x_t
    γ̂_t(a_t) ← V_t(a_t)⁻¹ b_t(a_t)

    # Slide the window
    z_t ← (1, x_t, x_{t-1}, …, x_{t-k+1})ᵀ
```

### 4.6 Implementation tips

- **Inverse maintenance (Sherman–Morrison).** Update `V_t(a)⁻¹` in `O((k+1)²)` instead of recomputing:
  ```
  q ← V⁻¹ · z_{t-1}
  V_t(a)⁻¹ ← V⁻¹ − (q · qᵀ) / (1 + z_{t-1}ᵀ · q)
  ```
  Store `V⁻¹(a)` and `b(a)` per action; recompute `γ̂(a) = V⁻¹(a) · b(a)` only when `a = a_t`.
- **Determinant maintenance (matrix determinant lemma).** `det V_t(a) ← det V_{t-1}(a) · (1 + z_{t-1}ᵀ V_{t-1}(a)⁻¹ z_{t-1})`. Avoids re-computing `det`.
- **`z_t` is a sliding window.** Use `collections.deque(maxlen=k)` or a small ring buffer. Position 0 stays as `1`; positions 1..k slide.
- **Quadratic form `‖z‖²_{V⁻¹}`.** Compute `zᵀ V⁻¹ z` once per `(a, t)`; reuse for both the bonus and the determinant update.
- **Initial `k` rounds.** If you don't want to assume `z_0 = (1, 0, …, 0)`, play any action for the first `k` rounds and only start the UCB loop at `t = k + 1`. This costs at most an additive constant in regret.

### 4.7 Misspecification of `k` and `m`

The paper provides robustness studies (Sections 5.3, 5.4; Appendix D.2).

- **Overestimating `k` (using `k̄ > k`).** Set `γ_i(a) = 0` for `i > k`. The regret rate stays sublinear but the constant worsens: `(k + 1)^{3/2}` → `(k̄ + 1) · √(k̄ + 1)`. Severe overestimation (`k̄ = 16` vs. true `k = 0`) still gives sublinear regret in experiments.
- **Overestimating `m`.** Using `m̄ ≥ m` keeps sublinear regret with an extra `√(m̄² + 1)` factor in the bonus. Severe underestimation (`m̄ ≪ m`) causes linear regret.
- **Practical rule of thumb:** pick `m̄` of the same order of magnitude as the expected `m`; pick `k̄ = k` if known, else a small constant like 2 or 4.

### 4.8 The ARB environment simulator

```python
class ARBEnv(Environment):
    """
    x_t = gamma_0(a) + sum_{i=1}^k gamma_i(a) * x_{t-i} + xi_t,  xi_t ~ N(0, sigma^2)

    gamma : (n, k+1) array; gamma[a, 0] is the intercept, gamma[a, 1..k] are the AR coefs.
    """
    def __init__(self, gamma, k, sigma, x_init=None):
        self.gamma = np.asarray(gamma, float)
        self.n, kp1 = self.gamma.shape
        assert kp1 == k + 1
        self.k = k
        self.sigma = sigma
        self.x_init = (np.zeros(k) if x_init is None
                       else np.asarray(x_init, float))

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        # past[0] is the most recent reward, past[1] one step earlier, etc.
        self.past = self.x_init.copy()
        self._t = 0
        self._last_a = None

    @property
    def z(self):
        return np.concatenate([[1.0], self.past])      # ℝ^{k+1}

    def expected_reward(self, a):
        return float(self.gamma[a] @ self.z)

    def pull(self, a):
        mu = self.expected_reward(a)
        x_t = mu + self.rng.normal(0, self.sigma)
        # Slide the window: drop the oldest, prepend the newest
        self.past = np.concatenate([[x_t], self.past[:-1]])
        self._t += 1
        return x_t

    def optimal_expected_reward(self):
        # The optimal myopic action given the current z (Theorem 1)
        vals = self.gamma @ self.z              # length n
        return float(vals.max())
```

### 4.9 Experimental settings from Section 5.1 of the paper

Three benchmark scenarios (Table 1 of the paper):

| Setting | k | n | m | σ |
|---|---|---|---|---|
| A | 2 | 2 | 1   | 0.75 |
| B | 4 | 7 | 20  | 1.5 |
| C | 4 | 7 | 920 | 10 |

`γ_i(a)` are sampled from uniform distributions per action and setting, with `Σ_i γ_i(a) < 1` to ensure stability. Hyperparameters of AR-UCB: `λ = 1`, `m̄ ∈ {10, 100, 1000}` (same order of magnitude as true `m`). Baselines: UCB1, EXP3, B-EXP3, AR2. AR-UCB should dominate all three settings.

### 4.10 Sanity checks

- **`k = 0` (degenerate to stochastic MAB).** `z_{t-1} = (1)`, `γ(a) = (γ_0(a))`, so the algorithm reduces to UCB1 with per-action confidence bounds. Verify by comparing to UCB1 on a stationary MAB.
- **`γ_i(a) = 0` for `i > k_true` but `k̄ > k_true`.** AR-UCB should still be sublinear (Appendix D.2 plots).

---

## 5. Cross-algorithm comparison: putting all three on the same plot

Because the three settings are *different*, you cannot in general run an "apples to apples" benchmark. But you can do the following compatibility checks, all of which the papers themselves perform:

| Run on… | AR2 | DynLin-UCB | AR-UCB | Comment |
|---|---|---|---|---|
| AR(1) env (AR2's home turf) | **wins** | linear regret (paper Figs.) | sublinear (mod-UCB-like) | AR-UCB with `k=1, n=k_arms` is sensible but ignores that the chain is per-arm |
| ARB env (AR-UCB's home turf) | linear regret (paper Fig. 8) | linear regret | **wins** | AR2 is not applicable in spirit because the per-arm chain assumption fails |
| DLB env (DynLin-UCB's home turf) | linear regret | **wins** | linear regret | AR-UCB cannot exploit the action's vector structure |

So if you want to "compare three algorithms across three settings", you produce a 3×3 grid of plots where the diagonal shows each algorithm winning on its home turf and the off-diagonals demonstrate the cost of model misspecification. This is the natural Option A (Review & Application) write-up structure for the course's final project.

---

## 6. Reproduction checklist (what to run, what to plot)

1. **AR2 reproduction.**
   - Run AR2 + 7 baselines (ETC, UCB1, ε-greedy, EXP3, RExp3, mod-UCB, SW-UCB, SW-TS, PS) on the heterogeneous AR(1) setting from Appendix A, `k ∈ {2, 10, 20}`, `E[α_i] ∈ {0.4, 0.9}`, 100 seeds.
   - Reproduce Table 2 (normalized regret).
   - Verify AR2 wins on (almost) every cell.

2. **DynLin-UCB reproduction.**
   - Synthetic DLB from §5.1 (matrices above), `T = 10⁶`, 50 seeds.
   - Reproduce Figure 1 (cumulative regret vs. rounds, log-log axes optional).
   - Reproduce Figure 2 (sensitivity to `ρ̄` — show that even `ρ̄ = 2 ρ(A)` is sublinear).
   - Reproduce Figure 4 (sensitivity to noise level `σ`).

3. **AR-UCB reproduction.**
   - Settings A, B, C from Section 5.1, `T = 10⁴`, 100 seeds.
   - Reproduce Figure 1 (cumulative regret in three subplots; AR-UCB wins all).
   - Reproduce Figure 2 (sensitivity to `m̄`) and Figure 3 (sensitivity to `k̄`).

For each plot, save raw arrays (`numpy.savez`) so you can re-plot later without rerunning.

---

## 7. Common pitfalls (read this twice before you start)

1. **Forgetting to advance the unpulled arms in AR2.** Every arm's latent chain advances at every round; the agent only fails to *observe* the unpulled ones. The pre-generated `time_series` matrix from `code.ipynb` handles this automatically.
2. **Updating Ridge inside the persistence loop in DynLin-UCB.** Don't. Only the **last** observation of each epoch is unbiased.
3. **Using OLS on AR(1) data when α is unknown.** Truncation biases OLS. Use MLE on the *truncated* likelihood.
4. **Sliding window bugs in AR-UCB.** `z_t` must include the constant `1` in position 0; the AR window is positions `1..k`. Off-by-one errors here silently produce a different algorithm.
5. **Forgetting to drop `i_sup` from `T` in AR2.** When an arm becomes the superior arm, it should leave the triggered set (or the alternation logic will keep picking it as a "triggered" arm).
6. **Mismatching `c_1` in AR2 paper vs. code.** Default to `c_1 = 8 c_0` (the code) to match published experimental numbers.
7. **Using `det V_t(a)` directly in AR-UCB instead of `det V_t(a) / λ^{k+1}`.** The log inside the bonus is over the **ratio** `det V / λ^{k+1}`, which equals zero at initialization (when `V = λ I`). Forgetting the denominator makes the bonus grow without bound at `t = 0`.
8. **Different timing convention in DLB.** Reward `y_t` is observed *before* state advances. Don't accidentally swap the order.
9. **Discretization of `U_action` in DynLin-UCB.** If your grid is too coarse, the agent literally cannot reach `u*` and converges to a suboptimal grid point. Refine until you can verify the optimal action is in the grid.

---

## 8. Required external packages

```
numpy>=1.24
scipy>=1.10        # truncated MLE for α in AR2
matplotlib>=3.7
seaborn>=0.13      # only for nicer plots
tqdm>=4.65         # progress bars
pytest>=7.4        # unit tests
```

No deep-learning frameworks, no JAX, no PyTorch. Everything should run in ≤ a few minutes per (algorithm, seed, T = 10⁴) pair on a laptop CPU.

---

## 9. Suggested deliverable order

1. **Week 1:** scaffolding (`Environment`, `Agent`, `AR1Env`, `ARBEnv`, `DLBEnv`); UCB1 + EXP3 baselines; tests verify each env is deterministic given seed.
2. **Week 2:** AR2 + AR2-p; reproduce Table 2 of the AR2 paper.
3. **Week 3:** AR-UCB; reproduce Figure 1 (Settings A/B/C) and Figures 2/3 (`m̄`, `k̄` sensitivity).
4. **Week 4:** DynLin-UCB; reproduce Figures 1, 2, 4 of the DLB paper on the synthetic setting.
5. **Week 5 (optional, Option B):** cross-algorithm 3×3 grid; ablations; novel contributions (e.g. hybrid algorithms, alternative confidence bounds, AR(k)-version of AR2 vs. AR-UCB head-to-head).

---

## 10. Reference: variable-name cheat sheet across the three papers

Different papers reuse the same letters for different things. Keep this table next to you while reading.

| Quantity | AR2 paper | DynLin-UCB paper | AR-UCB paper | Suggested code name |
|---|---|---|---|---|
| Number of arms / actions | `k` | n/a (continuous) | `n` | `n_arms` |
| Horizon | `T` | `T` | `T` | `T` |
| AR order / state dim | n/a | `n` (state dim) | `k` | `k_ar` / `n_state` |
| Action dim | n/a | `d` | n/a | `d_action` |
| AR coefficient(s) | `α` (scalar or vector) | n/a | `γ(a) ∈ ℝ^{k+1}` | `alpha` / `gamma_a` |
| Bonus constant | `c_0`, `c_1` | `β_t` | `β_t(a)` | `beta_t` |
| Gram matrix | n/a | `V_t` | `V_t(a)` | `V` (per-action dict if ARB) |
| Truncation boundary | `ℛ` / `L` | (none; uses Ω, X) | n/a | `L` |
| Innovation sd | `σ` | `σ` | `σ` | `sigma` |
| Stability index | n/a | `ρ̄` (spectral radius UB) | `Γ` | `rho_bar` / `Gamma` |

---

## Appendix A. Quick-reference: the three papers' key theorems

- **AR2 (Theorem 5.2):** `REG ≤ O(c_0² α² σ² k³ · log(c_0 α σ √k))` when `α ∈ [0.5, 1)` and `k ≤ K(α) := ⌊½ (log(1/8)/log α + 1)⌋`. Matches the lower bound `Ω(k α² σ²)` up to log factors as `α → 1`.
- **DynLin-UCB (Theorem 3.2):** `E[R(T)] ≤ Õ(d σ √T (log T)^{3/2} / (1 − ρ̄) + √(d T) (log T)² / (1 − ρ̄)^{3/2} + 1/(1 − ρ(A))²)`. Lower bound `Ω(d √T / √(1 − ρ(A)))`.
- **AR-UCB (Theorem 5):** `E[R_T] ≤ Õ((k + 1)^{3/2} √(n T) / (1 − Γ)²)`. Matches `Õ(√T)` standard MAB rate when `k = 0, Γ = 0`.

---

## Appendix B. File `code.ipynb` — what's actually in it

For convenience, the cells of `code.ipynb` and what each contains:

| Cell | Content |
|---|---|
| 0 | Imports (`numpy`, `matplotlib`, `seaborn`, `scipy.stats.truncnorm`) |
| 1 | `generate_time_series(T, k, alpha, sigma, L, seed)` — AR(1) env (per-arm, truncated) |
| 2 | `compute_regret_percentage(...)` — normalized regret formula |
| 3 | **`AR2(time_series, alpha, sigma, c_0, criteria=None, L=5.0)`** — main algorithm |
| 4 | `ETC(time_series, m=50)` — explore-then-commit |
| 5 | `RExp3(time_series, V)` — restarting EXP3 (Besbes et al. 2014) |
| 6 | `epsilon_greedy(time_series, alpha, epsilon=0.1)` |
| 7 | `UCB_simple(time_series)` — UCB1 |
| 8 | `UCB_missing_data(time_series, alpha, sigma, delta=0.1)` — **mod-UCB** (Algorithm 3 in the paper) |
| 9 | `sliding_window_UCB(time_series, B, ksi=0.8)` — SW-UCB |
| 10 | `SW_TS(time_series, B)` — Sliding-window Thompson |
| 11+ | Experiment driver — sweeps seeds, collects normalized regret, makes box plots |

Treat cells 3 and 8 as the canonical authority on AR2's actual numerical behavior. Disagreements with the published Algorithm 1 are documented in §2.6 and §2.9 above.

---

*End of guide. Once your three implementations replicate the headline plots of each paper, you have a strong foundation for either the Option A (review-and-application) or Option B (novel extension) deliverable described in `final_project_guidelines1.pdf`.*