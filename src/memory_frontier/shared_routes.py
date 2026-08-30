from __future__ import annotations

from math import asinh, cosh, inf, sinh
from collections.abc import Sequence

import numpy as np


def _route_strength_vector(route_strengths: Sequence[float]) -> np.ndarray:
    strengths = np.asarray(tuple(route_strengths), dtype=float)
    if strengths.ndim != 1 or strengths.size == 0:
        raise ValueError("route_strengths must be a non-empty vector")
    if np.any(strengths < 0.0):
        raise ValueError("route_strengths must be non-negative")
    if not np.any(strengths > 0.0):
        raise ValueError("at least one route strength must be positive")
    return strengths


def shared_entrance_mode_coordinates(
    route_strengths: Sequence[float],
    exits: Sequence[float],
) -> tuple[float, np.ndarray]:
    """Decompose exit coordinates into coefficient-aligned and orthogonal modes.

    For the beneficial quadratic shared-route loss

        L = -x * sum_j a_j y_j,

    this returns ``q=u dot y`` and ``y_perp=y-q*u``, where
    ``u=a/||a||``. The leading gradient flow changes only ``q``; ``y_perp`` is
    exactly constant.
    """
    strengths = _route_strength_vector(route_strengths)
    y = np.asarray(tuple(exits), dtype=float)
    if y.shape != strengths.shape:
        raise ValueError("exits must match route_strengths")
    norm = float(np.linalg.norm(strengths))
    direction = strengths / norm
    aligned = float(np.dot(direction, y))
    orthogonal = y - aligned * direction
    return aligned, orthogonal


def shared_entrance_quadratic_flow(
    route_strengths: Sequence[float],
    initial_entrance: float,
    initial_exits: Sequence[float],
    time: float,
) -> tuple[float, np.ndarray]:
    """Exact gradient-flow solution for several routes sharing one entrance.

    For

        L = -x * sum_j a_j y_j,    a_j >= 0,

    write ``c=||a||``, ``u=a/c``, ``q=u dot y``. Gradient flow reduces to

        dx/dt = c*q,
        dq/dt = c*x,
        d(y_perp)/dt = 0.

    The returned entrance and exit vector are the exact solution at ``time``.
    """
    strengths = _route_strength_vector(route_strengths)
    entrance = float(initial_entrance)
    exits = np.asarray(tuple(initial_exits), dtype=float)
    if exits.shape != strengths.shape:
        raise ValueError("initial_exits must match route_strengths")
    t = float(time)
    if t < 0.0:
        raise ValueError("time must be non-negative")

    norm = float(np.linalg.norm(strengths))
    direction = strengths / norm
    aligned, orthogonal = shared_entrance_mode_coordinates(strengths, exits)
    argument = norm * t
    evolved_entrance = entrance * cosh(argument) + aligned * sinh(argument)
    evolved_aligned = aligned * cosh(argument) + entrance * sinh(argument)
    evolved_exits = orthogonal + evolved_aligned * direction
    return float(evolved_entrance), evolved_exits


def shared_entrance_zero_exit_crossing_times(
    route_strengths: Sequence[float],
    initial_entrance: float,
    threshold: float,
) -> np.ndarray:
    """Exact exit-threshold times when all shared-route exits start at zero.

    With ``y(0)=0`` and positive entrance ``x0``, each exit obeys

        y_j(t) = (a_j/||a||) * x0 * sinh(||a|| t).

    Thus stronger routes cross any common positive threshold earlier. A route
    with zero strength never moves and receives crossing time ``inf``.
    """
    strengths = _route_strength_vector(route_strengths)
    entrance = float(initial_entrance)
    target = float(threshold)
    if entrance < 0.0:
        raise ValueError("initial_entrance must be non-negative")
    if target <= 0.0:
        raise ValueError("threshold must be positive")

    times = np.full(strengths.shape, inf, dtype=float)
    if entrance == 0.0:
        return times

    norm = float(np.linalg.norm(strengths))
    for index, strength in enumerate(strengths):
        if strength > 0.0:
            times[index] = asinh(target * norm / (strength * entrance)) / norm
    return times
