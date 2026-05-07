# AR2 Algorithm Implementation Guide: Non-stationary Auto-Regressive Bandits

## 1. Project Overview
This project implements the **AR2 (Alternating and Restarting)** algorithm designed for Multi-Armed Bandit (MAB) problems in non-stationary environments where the reward of each arm follows an **Auto-Regressive (AR)** process. 

The core challenge is that rewards are "restless" (they change even when not pulled) and the temporal correlation of past observations decays exponentially. AR2 solves this by balancing continuous exploitation of a "superior arm" with carefully timed, mathematically triggered exploration of dormant arms.

---

## 2. Environment Specifications (The AR-1 Model)

The simulation environment must adhere to the following rules:

### 2.1 State Dynamics
For each arm $i \in \{1, 2, \dots, k\}$, the reward $r_i(t)$ at time step $t$ follows a truncated AR-1 process:
$$r_i(t+1) = \max \left( \min( \alpha r_i(t) + \epsilon_i(t), \mathfrak{R} ), -\mathfrak{R} \right)$$

*   **$\alpha$ (AR parameter):** $0 < \alpha < 1$. Controls the correlation with the past.
*   **$\epsilon_i(t)$ (Noise):** Sampled independently from a zero-mean distribution (e.g., Gaussian $\mathcal{N}(0, \sigma^2)$).
*   **$\mathfrak{R}$ (Truncation Boundary):** A predefined maximum absolute bound for rewards (e.g., $\mathfrak{R} = 1.0$).

### 2.2 Observation Model
*   The decision-maker only observes the reward of the pulled arm $I_t$: Observe $R_{I_t}(t) = r_{I_t}(t)$.
*   Rewards of unpulled arms evolve silently in the background according to the state dynamics above.

---

## 3. The AR2 Algorithm Core Logic

The algorithm operates by tracking specific variables for each arm and using a "triggering mechanism" to decide when to explore.

### 3.1 Tracked Variables per Arm $i$
Initialize the following for each arm $i$:
1.  **$\hat{r}_i(t)$ (Estimated Expected Reward):** 
    *   Initialize: $\hat{r}_i(0) = 0$
    *   Update if arm $i$ is pulled at $t$: $\hat{r}_i(t) = R_i(t)$
    *   Update if arm $i$ is NOT pulled at $t$: $\hat{r}_i(t) = \alpha \cdot \hat{r}_i(t-1)$
2.  **$\hat{E}_i(t)$ (Estimated Tracking Error Bound):**
    *   Initialize: $\hat{E}_i(0) = \mathfrak{R}$ (Maximum uncertainty)
    *   Update if arm $i$ is pulled at $t$: $\hat{E}_i(t) = 0$
    *   Update if arm $i$ is NOT pulled at $t$: $\hat{E}_i(t) = \alpha \cdot \hat{E}_i(t-1) + \tilde{\sigma}$ 
        *(Note: $\tilde{\sigma}$ is a confidence parameter related to the noise variance, e.g., $\tilde{\sigma} \approx 2\sigma$)*
3.  **$\tau_i(t)$ (Time since last pull):**
    *   Initialize: $\tau_i(0) = \infty$ (or a large number)
    *   Update if arm $i$ is pulled at $t$: $\tau_i(t) = 0$
    *   Update if arm $i$ is NOT pulled at $t$: $\tau_i(t) = \tau_i(t-1) + 1$

### 3.2 The Triggering Condition
Let $i^*(t)$ be the index of the "Superior Arm" (the arm currently considered the best).
A dormant arm $j \neq i^*(t)$ is **triggered** for exploration if its potential upper bound exceeds the estimated value of the superior arm:
$$\text{Trigger if: } \hat{r}_j(t) + \hat{E}_j(t) > \hat{r}_{i^*}(t)$$

### 3.3 Main Loop (Phase II: Exploitation & Alternating)
*Assume $\alpha$ is known for now (Section 4 handles unknown $\alpha$).*
*Initialize by pulling all arms once to set initial $\hat{r}_i$ and find an initial superior arm $i^*$.*

**For $t = 1$ to $T$:**
1.  **Check Triggers:** For all arms $j \neq i^*$, check the trigger condition: $\hat{r}_j(t) + \hat{E}_j(t) > \hat{r}_{i^*}(t)$.
2.  **Select Arm:**
    *   **If triggers occurred:** Let $S$ be the set of triggered arms. Pull the arm $j \in S$ that has the highest upper confidence bound: $I_t = \arg\max_{j \in S} (\hat{r}_j(t) + \hat{E}_j(t))$. 
    *   **If no triggers:** Pull the superior arm: $I_t = i^*$.
3.  **Observe and Update:**
    *   Observe actual reward $R_{I_t}(t)$.
    *   Update $\hat{r}_{I_t}(t)$, $\hat{E}_{I_t}(t)$, and $\tau_{I_t}(t)$ for the pulled arm.
    *   Update $\hat{r}_j(t)$, $\hat{E}_j(t)$, and $\tau_j(t)$ for all unpulled arms using the decay formulas (Section 3.1).
4.  **Superior Arm Restart (Crucial Step):**
    *   If a triggered arm was pulled ($I_t \neq i^*$), it becomes the new candidate. 
    *   Compare the newly observed value of $I_t$ with the updated estimate of the old superior arm.
    *   If $R_{I_t}(t) > \hat{r}_{i^*}(t)$, update the superior arm: $i^* = I_t$. 
    *   *(Note: The logic here is "Alternating and Restarting". Once an arm is triggered, it gets a chance to become the new $i^*$.)*

---

## 4. Extension: Unknown AR Parameter (MLE Phase)

If $\alpha$ is unknown, the algorithm must implement a two-phase approach (Phase I for estimation, Phase II for AR2 execution).

### 4.1 Phase I: Estimation (Rounds $t=1$ to $T_{est}$)
1.  Set $T_{est} = \lceil T^{\beta} \rceil$ where $\beta \in (0,1)$ (e.g., $T_{est} = 100$).
2.  Select a fixed arm (e.g., Arm 1).
3.  Pull this arm consecutively for $T_{est}$ rounds. Store the sequence of rewards: $R(1), R(2), \dots, R(T_{est})$.

### 4.2 Maximum Likelihood Estimation (MLE)
Calculate $\hat{\alpha}$ by minimizing the negative log-likelihood function (handling the truncation boundaries). 
*Implementation note for Claude: A simple Least Squares estimator might be biased due to truncation. Implement an MLE function that accounts for the Gaussian PDF and CDF at boundaries $-\mathfrak{R}$ and $\mathfrak{R}$. If a simplified version is needed for a prototype, standard Least Squares ($\hat{\alpha} = \frac{\sum r(t)r(t+1)}{\sum r(t)^2}$) can be used as a baseline, but MLE is required for full paper fidelity.*

### 4.3 Phase II: Proceed with AR2
Use the estimated $\hat{\alpha}$ as the true $\alpha$ in the AR2 Main Loop described in Section 3.3 for the remaining $T - T_{est}$ rounds.

---

## 5. Metrics to Track (For Evaluation)
The code should track and output the following to compare performance against benchmarks (like $\epsilon$-greedy or standard UCB):
*   **Cumulative Regret:** $\sum_{t=1}^T (r^*(t) - r_{I_t}(t))$, where $r^*(t) = \max_i r_i(t)$. (Requires the simulation to track true states of all arms in the background).
*   **Optimal Pull Ratio:** The percentage of rounds where $I_t == \arg\max_i r_i(t)$.

## 6. Development Checklist for AI Agent
- [ ] Set up the AR-1 Environment Class (handling background restless updates and truncation).
- [ ] Implement the MLE estimator for $\alpha$ (or baseline Least Squares).
- [ ] Implement the AR2 Agent Class (managing $\hat{r}$, $\hat{E}$, and triggers).
- [ ] Implement baseline agents ($\epsilon$-greedy, Mod-UCB) for comparison.
- [ ] Create a simulation runner that logs Cumulative Regret and outputs comparison plots.