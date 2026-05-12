import numpy as np

from autoregressive_bandits.algorithms.ar2 import AR2Agent
from autoregressive_bandits.algorithms.ar_ucb import ARUCBAgent
from autoregressive_bandits.algorithms.base import ObservationContext
from autoregressive_bandits.algorithms.dynlin_ucb import DynLinUCBAgent
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


def test_ar2_author_preset_uses_notebook_trigger_multiplier_and_ucb_selection():
    agent = AR2Agent().reset(
        3,
        horizon=20,
        seed=1,
        alpha=0.9,
        sigma=0.1,
        epoch_size=20,
        c0=0.01,
        c1_multiplier=8,
        superior_mode="max_estimate",
        triggered_selection="ucb",
    )

    assert agent.c1 == 8 * agent.c0
    assert agent.superior_mode == "max_estimate"
    assert agent.triggered_selection == "ucb"


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


def test_ar_ucb_updates_only_selected_arm_and_slides_history():
    agent = ARUCBAgent().reset(
        3,
        horizon=20,
        seed=1,
        ar_order=2,
        m_bound=1.0,
        sigma=0.1,
        lambda_=1.0,
        bootstrap_each_arm=False,
    )

    action = agent.select_action(_ctx(agent))
    assert 0 <= action < 3
    v_before = agent.v_inv.copy()
    agent.update(1, 0.4, {})

    assert not np.allclose(agent.v_inv[1], v_before[1])
    np.testing.assert_allclose(agent.v_inv[0], v_before[0])
    np.testing.assert_allclose(agent.v_inv[2], v_before[2])
    np.testing.assert_allclose(agent.history, [0.4, 0.0])
    assert np.all(np.isfinite([agent._beta(a) for a in range(3)]))


def test_dynlin_ucb_persists_epoch_action_and_updates_on_final_reward_only():
    agent = DynLinUCBAgent().reset(
        3,
        horizon=20,
        seed=1,
        rho_bar=0.5,
        lambda_=1.0,
        sigma=0.1,
        action_set="one_hot",
    )

    first = agent.select_action(_ctx(agent))
    agent.update(first, 0.1, {})
    h_after_epoch_one = agent.h_hat.copy()

    second = agent.select_action(_ctx(agent))
    agent.update(second, 0.2, {})
    np.testing.assert_allclose(agent.h_hat, h_after_epoch_one)
    second_again = agent.select_action(_ctx(agent))
    np.testing.assert_allclose(second_again, second)
    agent.update(second_again, 0.3, {})

    assert not np.allclose(agent.h_hat, h_after_epoch_one)
