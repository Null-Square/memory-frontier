from __future__ import annotations

from math import exp, log

import numpy as np

from .core import UnifilarSource, source_stationary_distribution


def _log_stationary_exp(probabilities: np.ndarray, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values))
    return peak + log(float(np.sum(probabilities * np.exp(values - peak))))


def transition_pair_jacobian_scale(
    memory_states: int,
    margin: float,
    temperature: float,
) -> float:
    """ST pressure scale for selected target 0 versus alternative target 1."""
    k = int(memory_states)
    a = float(margin)
    tau = float(temperature)
    if k < 2:
        raise ValueError("memory_states must be at least 2")
    if a <= 0:
        raise ValueError("margin must be positive")
    if tau <= 0:
        raise ValueError("temperature must be positive")
    weight = exp(a / tau)
    denom = weight + k - 1
    selected = weight / denom
    alternative = 1.0 / denom
    return alternative * (selected + 1.0 - alternative) / tau


def stationary_token_lag_conditionals(
    source: UnifilarSource,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact stationary token conditional operators Q_l for lags 1..max_lag.

    Returns ``(mu, operators)`` where

        operators[l-1, x, y] = P(X_{t+l}=y | X_t=x).
    """
    L = int(max_lag)
    if L < 1:
        raise ValueError("max_lag must be positive")
    pi = source_stationary_distribution(source)
    mu = pi @ source.emissions
    if np.any(mu <= 1e-15):
        raise ValueError("stationary token marginal must have full support")

    state_transition = source.state_transition_matrix()
    posterior_successor = np.zeros(
        (source.alphabet_size, source.n_states), dtype=float
    )
    for s in range(source.n_states):
        for x in range(source.alphabet_size):
            mass = pi[s] * source.emissions[s, x]
            successor = source.transitions[s, x]
            posterior_successor[x, successor] += mass / mu[x]

    operators = np.empty(
        (L, source.alphabet_size, source.alphabet_size), dtype=float
    )
    state_given_token = posterior_successor
    for lag in range(L):
        operators[lag] = state_given_token @ source.emissions
        state_given_token = state_given_token @ state_transition
    return mu, operators


def predict_self_loop_counterfactual_pressure(
    source: UnifilarSource,
    memory_states: int,
    unused_decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact pressure when an unreachable memory state is a hard self-loop.

    Forward behavior starts in memory 0 and has ``F(0,x)=0`` for every token, so
    memory state 1 is never visited. Counterfactually, however, let
    ``F(1,x)=1`` for every token. A derivative-induced perturbation from memory 0
    into memory 1 then persists through all later hard transitions.

    Decoder 0 predicts the stationary token marginal ``mu``; decoder 1 has logits
    ``log(mu)+d``. The exact row-0 pressure toward target 1 is

        p = J * diag(mu) * sum_l (T-l)/T * (Q_l d - psi_mu(d) 1),

    where Q_l is the lag-l token conditional operator. Other unreachable memory
    states, if present, are irrelevant to this pairwise pressure.
    """
    T = int(horizon)
    d = np.asarray(unused_decoder_logit_contrast, dtype=float)
    if T <= 0:
        raise ValueError("horizon must be positive")
    if d.shape != (source.alphabet_size,):
        raise ValueError("unused_decoder_logit_contrast has wrong shape")
    if T == 1:
        return np.zeros(source.alphabet_size, dtype=float)

    mu, operators = stationary_token_lag_conditionals(source, T - 1)
    penalty = _log_stationary_exp(mu, d)
    accumulated = np.zeros(source.alphabet_size, dtype=float)
    for lag in range(1, T):
        weight = (T - lag) / T
        accumulated += weight * (operators[lag - 1] @ d - penalty)

    jacobian = transition_pair_jacobian_scale(
        memory_states, margin, temperature
    )
    return jacobian * mu * accumulated


def predict_self_loop_centered_rescaled_pressure(
    source: UnifilarSource,
    memory_states: int,
    decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Centered multi-lag predictive operator seen by the self-loop scaffold."""
    T = int(horizon)
    d = np.asarray(decoder_logit_contrast, dtype=float)
    if T <= 0:
        raise ValueError("horizon must be positive")
    if d.shape != (source.alphabet_size,):
        raise ValueError("decoder_logit_contrast has wrong shape")
    if T == 1:
        return np.zeros(source.alphabet_size, dtype=float)

    mu, operators = stationary_token_lag_conditionals(source, T - 1)
    centered = d - float(mu @ d)
    accumulated = np.zeros(source.alphabet_size, dtype=float)
    for lag in range(1, T):
        accumulated += (T - lag) / T * (operators[lag - 1] @ centered)
    jacobian = transition_pair_jacobian_scale(
        memory_states, margin, temperature
    )
    return jacobian * accumulated
