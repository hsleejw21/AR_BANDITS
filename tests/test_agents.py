import numpy as np

from autoregressive_bandits.algorithms.ar2 import AR2Agent
from autoregressive_bandits.algorithms.base import ObservationContext
from autoregressive_bandits.algorithms.var_oracle_ar2 import VAROracleAR2Agent


def _ctx(agent):
    return ObservationContext(t=agent.t, n_arms=agent.n_arms, horizon=agent.horizon)


def test_ar2_epoch_initialization_pulls_every_arm_once():
    agent = AR2Agent().reset(3, horizon=20, seed=1, alpha=0.9, sigma=0.1, epoch_size=10)

    actions = []
    for reward in [0.1, 0.2, 0.3]:
        action = agent.select_action(_ctx(agent))
        actions.append(action)
        agent.update(action, reward, {})

    assert actions == [0, 1, 2]


def test_ar2_default_trigger_constant_is_practical_but_theoretical_available():
    practical = AR2Agent().reset(5, horizon=100, seed=1, alpha=0.9, sigma=0.1)
    theoretical = AR2Agent().reset(5, horizon=100, seed=1, alpha=0.9, sigma=0.1, theoretical_c0=True)

    assert practical.c0 == 0.01
    assert theoretical.c0 > practical.c0


def test_ar2_estimate_update_matches_recurrence():
    agent = AR2Agent().reset(2, horizon=10, seed=1, alpha=0.5, sigma=0.1, reward_bound=1.0, epoch_size=10)

    a0 = agent.select_action(_ctx(agent))
    agent.update(a0, 0.8, {})
    np.testing.assert_allclose(agent.est_rewards, [0.4, 0.0])

    a1 = agent.select_action(_ctx(agent))
    agent.update(a1, -0.2, {})
    np.testing.assert_allclose(agent.est_rewards, [0.2, -0.1])


def test_ar2_triggers_dormant_arm_and_selects_earliest_on_odd_round():
    agent = AR2Agent().reset(3, horizon=20, seed=1, alpha=0.9, sigma=0.1, epoch_size=20, c0=1.0)
    for reward in [0.1, 0.2, 0.3]:
        action = agent.select_action(_ctx(agent))
        agent.update(action, reward, {})

    action = agent.select_action(_ctx(agent))
    assert action == 2
    assert agent.triggered
    agent.update(action, 0.3, {})
    action = agent.select_action(_ctx(agent))

    assert action in {0, 1}
    assert agent.triggered


def test_var_oracle_ar2_initialization_pulls_every_arm_once():
    agent = VAROracleAR2Agent().reset(
        3,
        horizon=20,
        seed=1,
        transition_matrix=np.eye(3) * 0.8,
        noise_scale=0.1,
        epoch_size=10,
    )

    actions = []
    for reward in [0.1, 0.2, 0.3]:
        action = agent.select_action(_ctx(agent))
        actions.append(action)
        agent.update(action, reward, {})

    assert actions == [0, 1, 2]


def test_var_oracle_ar2_prediction_uses_transition_matrix_without_hidden_state():
    transition = np.array([[0.5, 0.25], [0.1, 0.8]])
    agent = VAROracleAR2Agent().reset(
        2,
        horizon=10,
        seed=1,
        transition_matrix=transition,
        noise_scale=0.0,
        epoch_size=10,
    )

    action = agent.select_action(_ctx(agent))
    agent.update(action, 0.8, {"observation": {"state": [99.0, 99.0]}})

    np.testing.assert_allclose(agent.est_rewards, transition @ np.array([0.8, 0.0]))
    assert np.all(np.isfinite(agent.est_cov))
    np.testing.assert_allclose(agent.est_cov, agent.est_cov.T)
