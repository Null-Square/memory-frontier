from __future__ import annotations

from math import inf, log
from collections.abc import Sequence

import numpy as np


def symmetric_square_free_completion_time(
    coefficient: float,
    degree: int,
    initial_scale: float,
    threshold: float,
) -> float:
    """Exact gradient-flow completion time for a symmetric square-free monomial.

    Consider

        L = c * prod_{j=1}^d epsilon_j,

    with ``c < 0`` and all ``d`` independently trainable coordinates initialized
    to the same positive value. Symmetry is preserved by gradient flow, so every
    coordinate follows

        dx/ds = C * x**(d-1),   C = -c.

    The returned time is when every coordinate first reaches ``threshold``.
    """
    c = float(coefficient)
    d = int(degree)
    initial = float(initial_scale)
    target = float(threshold)
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if d < 1:
        raise ValueError("degree must be positive")
    if initial < 0.0:
        raise ValueError("initial_scale must be non-negative")
    if target <= 0.0:
        raise ValueError("threshold must be positive")
    if initial >= target:
        return 0.0

    strength = -c
    if d == 1:
        return (target - initial) / strength
    if initial == 0.0:
        return inf
    if d == 2:
        return log(target / initial) / strength
    return (
        initial ** (2 - d) - target ** (2 - d)
    ) / (strength * (d - 2))


def tied_repeated_parameter_completion_time(
    coefficient: float,
    multiplicity: int,
    initial_scale: float,
    threshold: float,
) -> float:
    """Exact gradient-flow completion time for ``L = c * epsilon**d``.

    When one scalar parameter is reused ``d`` times in a contributing monomial,
    the chain rule gives

        dx/ds = C * d * x**(d-1).

    Thus tying all coordinates of a degree-``d`` square-free route to one scalar
    leaves the scalar loss curve ``c*x**d`` unchanged along the diagonal but
    changes Euclidean gradient-flow time by a factor ``1/d``.
    """
    c = float(coefficient)
    d = int(multiplicity)
    initial = float(initial_scale)
    target = float(threshold)
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if d < 1:
        raise ValueError("multiplicity must be positive")
    if initial < 0.0:
        raise ValueError("initial_scale must be non-negative")
    if target <= 0.0:
        raise ValueError("threshold must be positive")
    if initial >= target:
        return 0.0

    strength = -c * d
    if d == 1:
        return (target - initial) / strength
    if initial == 0.0:
        return inf
    if d == 2:
        return log(target / initial) / strength
    return (
        initial ** (2 - d) - target ** (2 - d)
    ) / (strength * (d - 2))


def _positive_exponent_vector(exponents: Sequence[int]) -> np.ndarray:
    values = np.asarray(tuple(exponents), dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("exponents must be a non-empty vector")
    if np.any(values <= 0):
        raise ValueError("active monomial exponents must be positive")
    return values


def monomial_gradient_flow_velocity(
    coefficient: float,
    exponents: Sequence[int],
    point: Sequence[float],
) -> np.ndarray:
    """Exact gradient-flow velocity for one beneficial monomial.

    For

        L = c * prod_i x_i**alpha_i,   c < 0,

    this returns ``-grad L``. Exponents may exceed one, so the function applies
    to repeated-parameter / repeated-edge computational constructions as well as
    square-free routes.
    """
    c = float(coefficient)
    alpha = _positive_exponent_vector(exponents)
    x = np.asarray(tuple(point), dtype=float)
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if x.shape != alpha.shape:
        raise ValueError("point must match exponent vector")
    if np.any(x < 0.0):
        raise ValueError("point coordinates must be non-negative")

    strength = -c
    velocity = np.empty_like(x)
    for parameter in range(len(alpha)):
        term = strength * alpha[parameter]
        for index, multiplicity in enumerate(alpha):
            power = multiplicity - (1 if index == parameter else 0)
            if power:
                term *= x[index] ** power
        velocity[parameter] = term
    return velocity


def normalized_squared_coordinates(
    exponents: Sequence[int],
    point: Sequence[float],
) -> np.ndarray:
    """Return the weighted-balanced coordinates ``x_i**2 / alpha_i``."""
    alpha = _positive_exponent_vector(exponents)
    x = np.asarray(tuple(point), dtype=float)
    if x.shape != alpha.shape:
        raise ValueError("point must match exponent vector")
    if np.any(x < 0.0):
        raise ValueError("point coordinates must be non-negative")
    return x * x / alpha


def balanced_exponent_vector_completion_time(
    coefficient: float,
    exponents: Sequence[int],
    initial_normalized_scale: float,
    threshold_normalized_scale: float,
) -> float:
    """Exact completion time on the weighted-balanced monomial manifold.

    For exponent vector ``alpha``, weighted balance means

        x_i = sqrt(alpha_i) * s

    for one common normalized scale ``s``. Gradient flow preserves this manifold.
    If ``d=sum(alpha)`` and

        A = prod_i alpha_i**(alpha_i/2),

    then ``s`` obeys

        ds/dt = C * A * s**(d-1),   C=-coefficient.

    This function returns the exact time for ``s`` to move from the initial to
    threshold normalized scale.
    """
    c = float(coefficient)
    alpha = _positive_exponent_vector(exponents)
    initial = float(initial_normalized_scale)
    target = float(threshold_normalized_scale)
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if initial < 0.0:
        raise ValueError("initial_normalized_scale must be non-negative")
    if target <= 0.0:
        raise ValueError("threshold_normalized_scale must be positive")

    total_degree = int(alpha.sum())
    multiplicity_prefactor = float(
        np.prod(alpha.astype(float) ** (alpha.astype(float) / 2.0))
    )
    return symmetric_square_free_completion_time(
        c * multiplicity_prefactor,
        total_degree,
        initial,
        target,
    )


def construction_time_divergence_power(degree: int) -> int | None:
    """Power of small initialization in leading completion-time divergence.

    For symmetric square-free routes:

    - degree 1 has a finite nonzero limit as initialization -> 0;
    - degree 2 diverges only logarithmically;
    - degree d >= 3 diverges as initial_scale**(-(d-2)).

    Returns ``None`` for the finite/logarithmic cases and ``d-2`` otherwise.
    """
    d = int(degree)
    if d < 1:
        raise ValueError("degree must be positive")
    return d - 2 if d >= 3 else None
