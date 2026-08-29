from __future__ import annotations

from math import log

import numpy as np

from .core import UnifilarSource, source_stationary_distribution
from .spectral import collapsed_transition_pressure_scale


def _log_stationary_exp(probabilities: np.ndarray, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values))
    return peak + log(float(np.sum(probabilities * np.exp(values - peak))))


def stationary_token_markovization(
    source: UnifilarSource,
) -> tuple[np.ndarray, np.ndarray]:
    """Stationary token marginal and one-step token conditional matrix.

    Returns ``(mu, Q)`` with

        mu[x] = P(X_t=x),
        Q[x,y] = P(X_{t+1}=y | X_t=x)

    under the stationary source process. For a higher-order/unifilar source this
    ``Q`` is only the observable one-token Markovization; it need not contain the
    source's full predictive state.
    """
    pi = source_stationary_distribution(source)
    mu = pi @ source.emissions
    if np.any(mu <= 1e-15):
        raise ValueError("stationary token marginal must have full support")

    joint = np.zeros((source.alphabet_size, source.alphabet_size), dtype=float)
    for s in range(source.n_states):
        for x in range(source.alphabet_size):
            mass = pi[s] * source.emissions[s, x]
            if mass <= 0:
                continue
            successor = source.transitions[s, x]
            joint[x] += mass * source.emissions[successor]

    q = joint / mu[:, None]
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("failed to construct token conditional matrix")
    if not np.allclose(mu @ q, mu, atol=1e-12):
        raise RuntimeError("token Markovization does not preserve stationary marginal")
    return mu, q


def predict_unifilar_raw_collapsed_pressure(
    source: UnifilarSource,
    memory_states: int,
    unused_decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact local STE pressure at a fully collapsed hard memory controller.

    The collapsed decoder predicts the stationary token marginal ``mu``. The
    unused decoder has logits ``log(mu) + d``. All hard memory transitions target
    state 0 and canonical transition logits use a common margin/temperature.

    Even when ``source`` has higher-order predictive state, the exact local
    pressure depends only on its one-token Markovization ``Q``:

        p = K * diag(mu) * (Q d - log(E_mu exp(d)) * 1).

    Positive ``p[x]`` favors routing observed token ``x`` into memory state 1.
    """
    d = np.asarray(unused_decoder_logit_contrast, dtype=float)
    mu, q = stationary_token_markovization(source)
    if d.shape != (source.alphabet_size,):
        raise ValueError("unused_decoder_logit_contrast has wrong shape")
    scale = collapsed_transition_pressure_scale(
        memory_states, horizon, margin, temperature
    )
    penalty = _log_stationary_exp(mu, d)
    return scale * mu * (q @ d - penalty)


def predict_unifilar_centered_rescaled_pressure(
    source: UnifilarSource,
    memory_states: int,
    decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact centered pressure operator at full memory collapse.

    After dividing raw pressure coordinate ``x`` by stationary token mass
    ``mu[x]`` and removing the ``mu``-weighted constant mode,

        centered_rescaled = K * Q @ (d - <d>_mu * 1).

    Hence the first local memory split sees only the observable one-token
    predictive operator ``Q``, even if the full source is higher-order.
    """
    d = np.asarray(decoder_logit_contrast, dtype=float)
    mu, q = stationary_token_markovization(source)
    if d.shape != (source.alphabet_size,):
        raise ValueError("decoder_logit_contrast has wrong shape")
    scale = collapsed_transition_pressure_scale(
        memory_states, horizon, margin, temperature
    )
    centered = d - float(mu @ d)
    return scale * (q @ centered)


def unifilar_collapsed_pressure_accessibility_margin(
    source: UnifilarSource,
    unused_decoder_logit_contrast: np.ndarray,
) -> float:
    """Scale-free local accessibility margin at a collapsed memory state."""
    d = np.asarray(unused_decoder_logit_contrast, dtype=float)
    mu, q = stationary_token_markovization(source)
    if d.shape != (source.alphabet_size,):
        raise ValueError("unused_decoder_logit_contrast has wrong shape")
    return float(np.max(q @ d) - _log_stationary_exp(mu, d))
