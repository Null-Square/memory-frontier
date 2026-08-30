from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .construction_time import (
    monomial_gradient_flow_velocity,
    symmetric_square_free_completion_time,
)


def _positive_exponents(exponents: Sequence[int]) -> np.ndarray:
    raw = np.asarray(tuple(exponents), dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("exponents must be a non-empty vector")
    if (
        not np.all(np.isfinite(raw))
        or np.any(raw <= 0.0)
        or np.any(raw != np.floor(raw))
    ):
        raise ValueError("exponents must be positive integers")
    return raw.astype(int)


def _positive_metric_weights(
    metric_weights: Sequence[float],
    shape: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(tuple(metric_weights), dtype=float)
    if values.shape != shape:
        raise ValueError("metric_weights must match exponent vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("metric_weights must be finite and strictly positive")
    return values


def diagonal_preconditioned_monomial_velocity(
    coefficient: float,
    exponents: Sequence[int],
    metric_weights: Sequence[float],
    point: Sequence[float],
) -> np.ndarray:
    """Exact diagonal-preconditioned gradient flow for one monomial.

    For

        L = c * prod_i x_i**alpha_i,   c < 0,

    and positive diagonal preconditioner ``M=diag(m_i)``, this returns

        dx/dt = -M grad L.

    The Euclidean monomial velocity is multiplied coordinatewise by ``m_i``.
    """
    alpha = _positive_exponents(exponents)
    weights = _positive_metric_weights(metric_weights, alpha.shape)
    velocity = monomial_gradient_flow_velocity(coefficient, alpha, point)
    return weights * velocity


def preconditioned_normalized_squared_coordinates(
    exponents: Sequence[int],
    metric_weights: Sequence[float],
    point: Sequence[float],
) -> np.ndarray:
    """Return metric-balanced coordinates ``x_i**2 / (m_i * alpha_i)``."""
    alpha = _positive_exponents(exponents)
    weights = _positive_metric_weights(metric_weights, alpha.shape)
    x = np.asarray(tuple(point), dtype=float)
    if x.shape != alpha.shape:
        raise ValueError("point must match exponent vector")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("point coordinates must be finite and non-negative")
    return x * x / (weights * alpha)


def preconditioned_balanced_exponent_vector_completion_time(
    coefficient: float,
    exponents: Sequence[int],
    metric_weights: Sequence[float],
    initial_normalized_scale: float,
    threshold_normalized_scale: float,
) -> float:
    """Exact completion time on the diagonal-metric balanced manifold.

    With ``M=diag(m_i)`` and exponent vector ``alpha``, the quantities

        x_i**2 / (m_i * alpha_i) - x_j**2 / (m_j * alpha_j)

    are conserved. On the balanced manifold

        x_i = sqrt(m_i * alpha_i) * s,

    the common scale obeys

        ds/dt = C * A * s**(d-1),

    where ``C=-coefficient``, ``d=sum(alpha)``, and

        A = prod_i (m_i * alpha_i)**(alpha_i/2).

    Thus a strictly positive diagonal metric changes the prefactor and the
    balanced geometry, but not the degree-controlled completion-time class.
    """
    c = float(coefficient)
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    alpha = _positive_exponents(exponents)
    weights = _positive_metric_weights(metric_weights, alpha.shape)
    combined = weights * alpha.astype(float)
    prefactor = float(np.prod(combined ** (alpha.astype(float) / 2.0)))
    return symmetric_square_free_completion_time(
        c * prefactor,
        int(alpha.sum()),
        float(initial_normalized_scale),
        float(threshold_normalized_scale),
    )
