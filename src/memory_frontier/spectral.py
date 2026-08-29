from __future__ import annotations

from math import exp, log

import numpy as np

from .core import UnifilarSource


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


def _log_mean_exp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values))
    return peak + log(float(np.mean(np.exp(values - peak))))


def collapsed_one_contrast_pressure_scale(
    cardinality: int,
    horizon: int,
    margin: float,
    temperature: float,
) -> float:
    """Positive scalar in the exact collapsed-memory pressure law."""
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
        * (1.0 / q)
        * alternative
        * (selected + 1.0 - alternative)
        / tau
    )


def predict_centered_collapsed_pressure(
    transition_matrix: np.ndarray,
    decoder_logit_contrast: np.ndarray,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    """Exact centered STE pressure for one unused decoder contrast.

    Assumptions:
      * source is ``observable_markov_source(P)`` with doubly stochastic P,
      * memory cardinality equals alphabet size q,
      * all hard transitions target memory state 0,
      * decoder rows 0,2,...,q-1 are identical,
      * row 1 differs from row 0 by ``decoder_logit_contrast``, and
      * transition logits use the canonical common margin/temperature geometry.

    If ``pressure[x] = grad[z(0,x,0)] - grad[z(0,x,1)]``, then

        centered_pressure = C * P @ centered(decoder_logit_contrast)

    exactly, not just to first order in the decoder contrast.
    """
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
    """Exact uncentered pressure when the collapsed decoder is uniform.

    Decoder row 0 and every row except row 1 have zero logits. Row 1 has logits
    ``unused_decoder_logits``. Positive pressure favors routing that observed
    token into memory state 1.
    """
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
    """Scale-free margin for whether any token is pushed toward unused memory."""
    p = np.asarray(transition_matrix, dtype=float)
    d = np.asarray(unused_decoder_logits, dtype=float)
    if not is_doubly_stochastic(p):
        raise ValueError("transition_matrix must be doubly stochastic")
    q = p.shape[0]
    if d.shape != (q,):
        raise ValueError("unused_decoder_logits must have shape (q,)")
    return float(np.max(p @ d) - _log_mean_exp(d))


def two_level_accessibility_cutoff(logit_half_gap: float) -> float:
    """Predictive-eigenvalue cutoff for an equal +/- decoder contrast.

    If a centered predictive eigenmode has equal numbers of decoder logits
    ``+u`` and ``-u``, it has some positive raw routing pressure iff

        abs(lambda) > log(cosh(u)) / u.

    The continuous limit at ``u=0`` is zero. This cutoff rises monotonically
    toward one as the decoder contrast becomes extreme.
    """
    u = abs(float(logit_half_gap))
    if u == 0.0:
        return 0.0
    log_cosh = float(np.logaddexp(u, -u) - log(2.0))
    return log_cosh / u
