from __future__ import annotations

import numpy as np

from .counterfactual import transition_pair_jacobian_scale


def dormant_chain_controller(
    chain_depth: int,
    alphabet_size: int = 2,
) -> np.ndarray:
    """Forward-collapsed controller with an unreachable chain of given depth.

    Memory 0 is absorbing from reset. Counterfactually, target 1 flows through
    1 -> 2 -> ... -> chain_depth -> 0 independent of the observed symbols.
    All nonzero memory states are therefore unreachable in the hard forward pass.
    """
    depth = int(chain_depth)
    alphabet = int(alphabet_size)
    if depth < 1:
        raise ValueError("chain_depth must be positive")
    if alphabet < 2:
        raise ValueError("alphabet_size must be at least 2")
    table = np.zeros((depth + 1, alphabet), dtype=int)
    for memory in range(1, depth):
        table[memory, :] = memory + 1
    table[depth, :] = 0
    return table


def binary_dormant_chain_readout_logits(
    chain_depth: int,
    logit_half_gap: float,
) -> np.ndarray:
    """Uniform dormant-chain decoders except the final +/- contrast row."""
    depth = int(chain_depth)
    if depth < 1:
        raise ValueError("chain_depth must be positive")
    u = float(logit_half_gap)
    logits = np.zeros((depth + 1, 2), dtype=float)
    logits[depth] = np.array([u, -u])
    return logits


def binary_delay_matched_first_order_pressure_coefficient(
    delay: int,
    switch_probability: float,
    horizon: int,
    margin: float,
    temperature: float,
) -> float:
    """Per-unit decoder contrast pressure for a delay-matched dormant chain.

    For the binary delayed-repeat source with delay ``R``, a dormant chain of
    depth ``R`` and final decoder logits ``(+eps,-eps)`` gives, to first order,

        pressure(token 0) = +coefficient * eps
        pressure(token 1) = -coefficient * eps.

    The formula assumes the standard equal-margin K-way transition-logit
    embedding with K=R+1. It is zero when the horizon is too short to reach the
    final dormant decoder.
    """
    delay = int(delay)
    rho = float(switch_probability)
    horizon = int(horizon)
    if delay < 2:
        raise ValueError("delay must be at least 2")
    if not 0.0 < rho < 1.0:
        raise ValueError("switch_probability must lie strictly between 0 and 1")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon <= delay:
        return 0.0
    predictive_correlation = 1.0 - 2.0 * rho
    jacobian = transition_pair_jacobian_scale(
        delay + 1, margin, temperature
    )
    return (
        jacobian
        * 0.5
        * (horizon - delay)
        / horizon
        * predictive_correlation
    )
