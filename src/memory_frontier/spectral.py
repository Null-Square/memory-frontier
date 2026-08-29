from __future__ import annotations

from math import exp

import numpy as np

from .core import UnifilarSource


def observable_markov_source(transition_matrix: np.ndarray) -> UnifilarSource:
    """Observable Markov source whose state is the previous emitted symbol.

    ``transition_matrix[s, x]`` is both the probability of emitting symbol ``x``
    from state ``s`` and, after that emission, the next source state is exactly
    ``x``. This is the simplest source class for connecting predictive dynamics
    directly to an observable Markov operator.
    """
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


def collapsed_one_contrast_pressure_scale(
    cardinality: int,
    horizon: int,
    margin: float,
    temperature: float,
) -> float:
    """Positive scalar in the exact collapsed-memory pressure law.

    The hard controller has ``q=cardinality`` memory states and alphabet symbols,
    every hard transition targets memory 0, and canonical transition logits give
    target 0 margin ``margin`` over every alternative. The backward pass uses a
    softmax with ``temperature``.
    """
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
      * the source is ``observable_markov_source(P)`` with doubly stochastic P,
      * memory cardinality equals the alphabet size q,
      * all hard transitions target memory state 0,
      * decoder rows 0,2,...,q-1 are identical,
      * row 1 differs from row 0 by ``decoder_logit_contrast``, and
      * transition logits use the canonical common margin/temperature geometry.

    If ``pressure[x] = grad[z(0,x,0)] - grad[z(0,x,1)]``, then after removing
    the uniform token mode,

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
