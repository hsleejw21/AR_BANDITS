import numpy as np

from autoregressive_bandits.estimators import (
    estimate_alpha_ls,
    estimate_alpha_vector_ls,
    estimate_transition_matrix_ls,
    estimate_transition_matrix_ridge,
    rescale_spectral_radius,
)


def test_least_squares_alpha_estimator_recovers_alpha_on_clean_sequence():
    alpha = 0.82
    rewards = np.array([alpha**t for t in range(20)], dtype=float)

    assert abs(estimate_alpha_ls(rewards) - alpha) < 1e-9


def test_alpha_vector_estimator_recovers_diagonal_coefficients():
    alpha = np.array([0.3, 0.7])
    states = np.array([[alpha[0] ** t, 2 * alpha[1] ** t] for t in range(10)], dtype=float)

    np.testing.assert_allclose(estimate_alpha_vector_ls(states), alpha)


def test_var_transition_estimator_recovers_matrix_on_clean_sequence():
    transition = np.array([[0.6, 0.2], [0.1, 0.7]])
    states = [np.array([1.0, -0.4])]
    for _ in range(20):
        states.append(transition @ states[-1])

    estimated = estimate_transition_matrix_ls(np.asarray(states))

    np.testing.assert_allclose(estimated, transition, atol=1e-8)


def test_rescale_spectral_radius_caps_unstable_matrix():
    matrix = np.array([[1.2, 0.0], [0.0, 0.5]])
    scaled = rescale_spectral_radius(matrix, 0.9)

    assert max(abs(np.linalg.eigvals(scaled))) <= 0.9 + 1e-12


def test_ridge_transition_estimator_shrinks_coefficients():
    transition = np.array([[0.6, 0.2], [0.1, 0.7]])
    states = [np.array([1.0, -0.4])]
    for _ in range(20):
        states.append(transition @ states[-1])
    states = np.asarray(states)

    ls = estimate_transition_matrix_ls(states)
    ridge = estimate_transition_matrix_ridge(states, ridge_lambda=10.0)

    assert np.linalg.norm(ridge) < np.linalg.norm(ls)


def test_ridge_transition_estimator_respects_spectral_radius_cap():
    states = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]])

    estimated = estimate_transition_matrix_ridge(states, ridge_lambda=0.0, spectral_radius=0.8)

    assert max(abs(np.linalg.eigvals(estimated))) <= 0.8 + 1e-12
