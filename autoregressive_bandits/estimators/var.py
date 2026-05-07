from __future__ import annotations

import numpy as np


def estimate_alpha_vector_ls(states: np.ndarray) -> np.ndarray:
    """Estimate one AR coefficient per coordinate from full-state sequences."""

    states = _validate_states(states)
    x = states[:-1]
    y = states[1:]
    denom = np.sum(x * x, axis=0)
    numer = np.sum(x * y, axis=0)
    alpha = np.divide(numer, denom, out=np.full(x.shape[1], 0.5), where=denom > 1e-12)
    return np.clip(alpha, 1e-6, 1 - 1e-6)


def estimate_transition_matrix_ls(
    states: np.ndarray,
    spectral_radius: float | None = None,
) -> np.ndarray:
    """Estimate A in r(t+1) = A @ r(t) with least squares."""

    states = _validate_states(states)
    x = states[:-1]
    y = states[1:]
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    matrix = coef.T
    if spectral_radius is not None:
        matrix = rescale_spectral_radius(matrix, spectral_radius)
    return matrix


def rescale_spectral_radius(matrix: np.ndarray, target_radius: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if target_radius <= 0:
        raise ValueError("target_radius must be positive")
    radius = max(abs(np.linalg.eigvals(matrix)))
    if radius > target_radius and radius > 0:
        matrix = matrix * (target_radius / float(radius))
    return matrix


def _validate_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must have shape (time, n_arms) with at least two rows")
    return states
