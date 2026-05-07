import numpy as np

from autoregressive_bandits.environments import (
    RestlessAR1BanditEnv,
    RestlessVARBanditEnv,
    generate_clustered_transition_matrix,
)


def test_environment_updates_all_arms_and_clips():
    env = RestlessAR1BanditEnv(
        n_arms=3,
        alpha=0.9,
        sigma=0.0,
        reward_bound=1.0,
        initial_state=[2.0, -2.0, 0.5],
        seed=1,
    )

    obs = env.step(2)

    assert obs.reward == 0.5
    assert obs.best_arm == 0
    np.testing.assert_allclose(obs.state, [1.0, -1.0, 0.5])
    np.testing.assert_allclose(obs.next_state, [0.9, -0.9, 0.45])


def test_environment_same_seed_same_hidden_path_independent_of_agent_name():
    env_a = RestlessAR1BanditEnv(3, alpha=0.8, sigma=0.05, initial_state=[0.1, 0.2, 0.3], seed=11)
    env_b = RestlessAR1BanditEnv(3, alpha=0.8, sigma=0.05, initial_state=[0.1, 0.2, 0.3], seed=11)

    states_a = [env_a.step(0).next_state for _ in range(5)]
    states_b = [env_b.step(2).next_state for _ in range(5)]

    for a, b in zip(states_a, states_b):
        np.testing.assert_allclose(a, b)


def test_environment_supports_legacy_scaled_noise_model():
    additive = RestlessAR1BanditEnv(1, alpha=0.5, sigma=0.2, initial_state=[0.4], seed=7)
    scaled = RestlessAR1BanditEnv(
        1,
        alpha=0.5,
        sigma=0.2,
        initial_state=[0.4],
        noise_model="scaled_noise",
        seed=7,
    )

    additive_next = additive.step(0).next_state[0]
    scaled_next = scaled.step(0).next_state[0]

    assert additive_next != scaled_next


def test_var_environment_deterministic_matrix_update_and_clipping():
    env = RestlessVARBanditEnv(
        n_arms=2,
        transition_matrix=[[0.5, 0.25], [0.0, 1.5]],
        noise_scale=0.0,
        reward_bound=1.0,
        initial_state=[0.8, 0.9],
        seed=3,
    )

    obs = env.step(0)

    assert obs.reward == 0.8
    np.testing.assert_allclose(obs.next_state, [0.625, 1.0])


def test_clustered_transition_matrix_is_stable_and_reproducible():
    a = generate_clustered_transition_matrix(6, seed=4, n_clusters=2, spectral_radius=0.85)
    b = generate_clustered_transition_matrix(6, seed=4, n_clusters=2, spectral_radius=0.85)

    assert a.shape == (6, 6)
    np.testing.assert_allclose(a, b)
    assert max(abs(np.linalg.eigvals(a))) <= 0.85 + 1e-9


def test_var_environment_same_seed_same_hidden_path_independent_of_action():
    kwargs = dict(
        n_arms=2,
        transition_matrix=[[0.8, 0.1], [0.1, 0.8]],
        noise_scale=0.05,
        initial_state=[0.1, 0.2],
        seed=8,
    )
    env_a = RestlessVARBanditEnv(**kwargs)
    env_b = RestlessVARBanditEnv(**kwargs)

    states_a = [env_a.step(0).next_state for _ in range(5)]
    states_b = [env_b.step(1).next_state for _ in range(5)]

    for a, b in zip(states_a, states_b):
        np.testing.assert_allclose(a, b)
