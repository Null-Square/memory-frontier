from __future__ import annotations

from math import log

import numpy as np

from .core import UnifilarSource, source_stationary_distribution


def smooth_controller_finite_horizon_log_loss(
    source: UnifilarSource,
    transition_probabilities: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
) -> float:
    """Exact expected loss for a stochastic finite-memory transition controller."""
    transition = np.asarray(transition_probabilities, dtype=float)
    if transition.ndim != 3:
        raise ValueError("transition_probabilities must have shape (k, alphabet, k)")
    k, alphabet_size, k2 = transition.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition probability shape mismatch")
    if np.any(transition < 0.0) or not np.allclose(
        transition.sum(axis=-1), 1.0, atol=1e-12
    ):
        raise ValueError("transition rows must be probability distributions")
    decoder = np.asarray(readout, dtype=float)
    if decoder.shape != (k, alphabet_size):
        raise ValueError("readout shape mismatch")
    if np.any(decoder <= 0.0) or not np.allclose(
        decoder.sum(axis=-1), 1.0, atol=1e-12
    ):
        raise ValueError("readout rows must be strictly positive distributions")
    T = int(horizon)
    if T <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    distribution = np.zeros((source.n_states, k), dtype=float)
    distribution[:, initial_memory] = source_stationary_distribution(source)
    log_readout = np.log(decoder)
    total = 0.0
    for _ in range(T):
        total -= float(
            np.einsum(
                "sm,sx,mx->",
                distribution,
                source.emissions,
                log_readout,
            )
        )
        next_distribution = np.zeros_like(distribution)
        for s in range(source.n_states):
            for m in range(k):
                weight = distribution[s, m]
                if weight == 0.0:
                    continue
                for x in range(alphabet_size):
                    probability = source.emissions[s, x]
                    if probability == 0.0:
                        continue
                    successor = source.transitions[s, x]
                    next_distribution[successor] += (
                        weight * probability * transition[m, x]
                    )
        distribution = next_distribution
    return total / T


def binary_soft_chain_transition(
    delay: int,
    link_probabilities: np.ndarray | list[float] | tuple[float, ...] | float,
) -> np.ndarray:
    """Collapsed binary memory with a soft token-0-triggered chain of length R.

    Memory 0 is the baseline state. Token 0 can enter state 1 with probability
    ``u_1``. From intermediate state ``j``, either token advances to ``j+1`` with
    probability ``u_{j+1}``; failure collapses to state 0. Final state R returns
    to state 0 after the next observation.
    """
    R = int(delay)
    if R < 1:
        raise ValueError("delay must be positive")
    if np.isscalar(link_probabilities):
        links = np.full(R, float(link_probabilities), dtype=float)
    else:
        links = np.asarray(link_probabilities, dtype=float)
    if links.shape != (R,):
        raise ValueError("link_probabilities must have length delay")
    if np.any((links < 0.0) | (links > 1.0)):
        raise ValueError("link probabilities must lie in [0,1]")

    k = R + 1
    transition = np.zeros((k, 2, k), dtype=float)
    transition[:, :, 0] = 1.0
    transition[0, 0, 0] = 1.0 - links[0]
    transition[0, 0, 1] = links[0]
    for state in range(1, R):
        transition[state, :, 0] = 1.0 - links[state]
        transition[state, :, state + 1] = links[state]
    transition[R, :, 0] = 1.0
    return transition


def binary_chain_readout(delay: int, logit_half_gap: float) -> np.ndarray:
    """Uniform decoder rows except a token-0-specialized final chain state."""
    R = int(delay)
    if R < 1:
        raise ValueError("delay must be positive")
    gap = float(logit_half_gap)
    readout = np.full((R + 1, 2), 0.5, dtype=float)
    logits = np.array([gap, -gap], dtype=float)
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    readout[R] = probabilities / probabilities.sum()
    return readout


def binary_delay_chain_leading_gain_coefficient(
    delay: int,
    switch_probability: float,
    horizon: int,
    logit_half_gap: float,
) -> float:
    """Coefficient C_R in log(2)-L(epsilon)=C_R epsilon^R+o(epsilon^R)."""
    R = int(delay)
    rho = float(switch_probability)
    T = int(horizon)
    if R < 1:
        raise ValueError("delay must be positive")
    if not 0.0 < rho < 1.0:
        raise ValueError("switch_probability must lie strictly between 0 and 1")
    if T <= R:
        return 0.0
    gap = float(logit_half_gap)
    q0 = 1.0 / (1.0 + np.exp(-2.0 * gap))
    q1 = 1.0 - q0
    conditional_gain = (
        log(2.0)
        + (1.0 - rho) * log(q0)
        + rho * log(q1)
    )
    return (T - R) / T * 0.5 * conditional_gain
