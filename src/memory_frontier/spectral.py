from __future__ import annotations

from math import exp, log

import numpy as np

from .core import UnifilarSource, source_stationary_distribution


def observable_markov_source(transition_matrix: np.ndarray) -> UnifilarSource:
    """Observable Markov source whose state is the previous emitted symbol."""
    p = np.asarray(transition_matrix, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("transition_matrix must be square")
    if np.any(p < 0):
        raise ValueError("transition probabilities must be non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("each transition row must sum to 1")
    q = p.shape[0]
    transitions = np.tile(np.arange(q, dtype=int), (q, 1))
    return UnifilarSource(p, transitions)


def is_doubly_stochastic(matrix: np.ndarray, *, atol: float = 1e-12) -> bool:
    p = np.asarray(matrix, dtype=float)
    return (
        p.ndim == 2
        and p.shape[0] == p.shape[1]
        and np.all(p >= -atol)
        and np.allclose(p.sum(axis=1), 1.0, atol=atol)
        and np.allclose(p.sum(axis=0), 1.0, atol=atol)
    )


def _stationary_law(transition_matrix: np.ndarray) -> np.ndarray:
    source = observable_markov_source(transition_matrix)
    pi = source_stationary_distribution(source)
    if np.any(pi <= 1e-15):
        raise ValueError("stationary token law must have full support")
    return pi


def _log_mean_exp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values))
    return peak + log(float(np.mean(np.exp(values - peak))))


def _log_stationary_exp(pi: np.ndarray, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values))
    return peak + log(float(np.sum(pi * np.exp(values - peak))))


def collapsed_transition_pressure_scale(
    cardinality: int,
    horizon: int,
    margin: float,
    temperature: float,
) -> float:
    """Common positive STE scale before multiplying token stationary mass."""
    q = int(cardinality)
    T = int(horizon)
    a = float(margin)
    tau = float(temperature)
    if q < 2:
        raise ValueError("cardinality must be at least 2")
    if T <= 0:
        raise ValueError("horizon must be positive")
    if a <= 0:
        raise ValueError("margin must be positive")
    if tau <= 0:
        raise ValueError("temperature must be positive")

    weight = exp(a / tau)
    denom = weight + q - 1
    selected = weight / denom
    alternative = 1.0 / denom
    return (
        (T - 1)
        / T
        * alternative
        * (selected + 1.0 - alternative)
        / tau
    )


def collapsed_one_contrast_pressure_scale(
    cardinality: int,
    horizon: int,
    margin: float,
    temperature: float,
) -> float:
    """Uniform-token specialization of ``collapsed_transition_pressure_scale``."""
    q = int(cardinality)
    return collapsed_transition_pressure_scale(
        q, horizon, margin, temperature
    ) / q


def predict_stationary_raw_collapsed_pressure(
    transition_matrix: np.ndarray,
    unused_decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact raw pressure for an arbitrary stationary observable Markov source.

    The collapsed decoder row predicts the stationary token law ``pi``. The
    unused decoder row has logits ``log(pi) + d``, where ``d`` is
    ``unused_decoder_logit_contrast``. All hard transitions target memory 0.

        p = K * diag(pi) * (P d - log(E_pi exp(d)) * 1)

    Positive ``p[x]`` favors routing observed token ``x`` into the unused state.
    """
    p = np.asarray(transition_matrix, dtype=float)
    d = np.asarray(unused_decoder_logit_contrast, dtype=float)
    q = p.shape[0]
    if p.ndim != 2 or p.shape != (q, q):
        raise ValueError("transition_matrix must be square")
    if d.shape != (q,):
        raise ValueError("unused_decoder_logit_contrast must have shape (q,)")
    pi = _stationary_law(p)
    scale = collapsed_transition_pressure_scale(
        q, horizon, margin, temperature
    )
    penalty = _log_stationary_exp(pi, d)
    return scale * pi * (p @ d - penalty)


def predict_stationary_centered_rescaled_pressure(
    transition_matrix: np.ndarray,
    decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact stationary-weighted predictive operator law.

    Let raw pressure be ``p_raw`` and stationary token law be ``pi``. After
    dividing coordinate ``x`` by ``pi[x]`` and removing the pi-weighted constant
    mode, the exact pressure is

        centered_rescaled = K * P @ (d - 1 * <d>_pi).
    """
    p = np.asarray(transition_matrix, dtype=float)
    d = np.asarray(decoder_logit_contrast, dtype=float)
    q = p.shape[0]
    if p.ndim != 2 or p.shape != (q, q):
        raise ValueError("transition_matrix must be square")
    if d.shape != (q,):
        raise ValueError("decoder_logit_contrast must have shape (q,)")
    pi = _stationary_law(p)
    centered = d - float(pi @ d)
    scale = collapsed_transition_pressure_scale(
        q, horizon, margin, temperature
    )
    return scale * (p @ centered)


def stationary_collapsed_pressure_accessibility_margin(
    transition_matrix: np.ndarray,
    unused_decoder_logit_contrast: np.ndarray,
) -> float:
    """Scale-free accessibility margin under stationary-marginal decoder prior."""
    p = np.asarray(transition_matrix, dtype=float)
    d = np.asarray(unused_decoder_logit_contrast, dtype=float)
    q = p.shape[0]
    if p.ndim != 2 or p.shape != (q, q):
        raise ValueError("transition_matrix must be square")
    if d.shape != (q,):
        raise ValueError("unused_decoder_logit_contrast must have shape (q,)")
    pi = _stationary_law(p)
    return float(np.max(p @ d) - _log_stationary_exp(pi, d))


def predict_centered_collapsed_pressure(
    transition_matrix: np.ndarray,
    decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact centered pressure for the doubly-stochastic/uniform special case."""
    p = np.asarray(transition_matrix, dtype=float)
    d = np.asarray(decoder_logit_contrast, dtype=float)
    if not is_doubly_stochastic(p):
        raise ValueError("transition_matrix must be doubly stochastic")
    q = p.shape[0]
    if d.shape != (q,):
        raise ValueError("decoder_logit_contrast must have shape (q,)")
    centered = d - d.mean()
    scale = collapsed_one_contrast_pressure_scale(
        q, horizon, margin, temperature
    )
    return scale * (p @ centered)


def predict_raw_collapsed_pressure(
    transition_matrix: np.ndarray,
    unused_decoder_logits: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact raw pressure for the doubly-stochastic/uniform special case."""
    p = np.asarray(transition_matrix, dtype=float)
    d = np.asarray(unused_decoder_logits, dtype=float)
    if not is_doubly_stochastic(p):
        raise ValueError("transition_matrix must be doubly stochastic")
    q = p.shape[0]
    if d.shape != (q,):
        raise ValueError("unused_decoder_logits must have shape (q,)")
    scale = collapsed_one_contrast_pressure_scale(
        q, horizon, margin, temperature
    )
    penalty = _log_mean_exp(d)
    return scale * (p @ d - penalty)


def collapsed_pressure_accessibility_margin(
    transition_matrix: np.ndarray,
    unused_decoder_logits: np.ndarray,
) -> float:
    """Uniform special case of stationary collapsed accessibility margin."""
    p = np.asarray(transition_matrix, dtype=float)
    d = np.asarray(unused_decoder_logits, dtype=float)
    if not is_doubly_stochastic(p):
        raise ValueError("transition_matrix must be doubly stochastic")
    q = p.shape[0]
    if d.shape != (q,):
        raise ValueError("unused_decoder_logits must have shape (q,)")
    return float(np.max(p @ d) - _log_mean_exp(d))


def two_level_accessibility_cutoff(logit_half_gap: float) -> float:
    """Predictive-eigenvalue cutoff for an equal +/- decoder contrast."""
    u = abs(float(logit_half_gap))
    if u == 0.0:
        return 0.0
    log_cosh = float(np.logaddexp(u, -u) - log(2.0))
    return log_cosh / u
