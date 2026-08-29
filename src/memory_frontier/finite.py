from __future__ import annotations

import numpy as np

from .core import (
    ControllerOptimum,
    UnifilarSource,
    entropy_from_joint_symbol_mass,
    enumerate_deterministic_controllers,
    product_transition_matrix,
    source_stationary_distribution,
)


def _loss_from_occupancy(source: UnifilarSource, occupancy: np.ndarray) -> float:
    k = occupancy.shape[1]
    mass = np.zeros((k, source.alphabet_size), dtype=float)
    for s in range(source.n_states):
        for m in range(k):
            mass[m] += occupancy[s, m] * source.emissions[s]
    return entropy_from_joint_symbol_mass(mass)


def controller_average_occupancy(
    source: UnifilarSource,
    f: np.ndarray,
    initial_memory: int,
    horizon: int,
) -> np.ndarray:
    """Average p(S_t,M_t) over t=0,...,horizon-1 for reset sequences."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    f = np.asarray(f, dtype=int)
    k = f.shape[0]
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory state")

    pi_source = source_stationary_distribution(source)
    dist = np.zeros(source.n_states * k, dtype=float)
    for s in range(source.n_states):
        dist[s * k + initial_memory] = pi_source[s]

    p = product_transition_matrix(source, f)
    average = np.zeros_like(dist)
    for _ in range(horizon):
        average += dist
        dist = dist @ p
    return (average / horizon).reshape(source.n_states, k)


def controller_finite_horizon_log_loss(
    source: UnifilarSource,
    f: np.ndarray,
    initial_memory: int,
    horizon: int,
) -> float:
    """Exact expected reset-sequence NLL with one shared readout per memory state."""
    occupancy = controller_average_occupancy(source, f, initial_memory, horizon)
    return _loss_from_occupancy(source, occupancy)


def best_finite_horizon_deterministic_controller(
    source: UnifilarSource,
    k: int,
    horizon: int,
) -> ControllerOptimum:
    """Exhaustive deterministic K-state oracle for a fixed reset horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    best: ControllerOptimum | None = None
    for f in enumerate_deterministic_controllers(k, source.alphabet_size):
        for m0 in range(k):
            occupancy = controller_average_occupancy(source, f, m0, horizon)
            loss = _loss_from_occupancy(source, occupancy)
            if best is None or loss < best.loss - 1e-15:
                best = ControllerOptimum(loss, f.copy(), m0, occupancy.copy())
    assert best is not None
    return best
