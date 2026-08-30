from __future__ import annotations

from math import inf, log


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
