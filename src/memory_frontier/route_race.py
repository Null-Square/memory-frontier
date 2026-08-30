from __future__ import annotations

from math import asinh, inf, log, sqrt

import numpy as np


Polynomial = dict[tuple[int, ...], float]


def _parameter_count(coefficients: Polynomial) -> int:
    if not coefficients:
        raise ValueError("coefficients must be non-empty")
    lengths = {len(exponent) for exponent in coefficients}
    if len(lengths) != 1:
        raise ValueError("all exponent tuples must have the same length")
    return next(iter(lengths))


def multivariate_polynomial_gradient_coefficients(
    coefficients: Polynomial,
) -> tuple[Polynomial, ...]:
    """Differentiate a sparse multivariate polynomial exactly.

    The input uses exponent tuples as keys. The returned tuple contains one
    sparse coefficient dictionary per parameter.
    """
    n_parameters = _parameter_count(coefficients)
    gradients: list[Polynomial] = [dict() for _ in range(n_parameters)]
    for exponent, coefficient in coefficients.items():
        if any(power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative")
        value = float(coefficient)
        for parameter, multiplicity in enumerate(exponent):
            if multiplicity == 0:
                continue
            derivative_exponent = list(exponent)
            derivative_exponent[parameter] -= 1
            key = tuple(derivative_exponent)
            gradients[parameter][key] = (
                gradients[parameter].get(key, 0.0) + value * multiplicity
            )
    return tuple(gradients)


def multivariate_polynomial_hessian_coefficients(
    coefficients: Polynomial,
) -> tuple[tuple[Polynomial, ...], ...]:
    """Return the exact sparse Hessian of a multivariate polynomial."""
    n_parameters = _parameter_count(coefficients)
    hessian: list[list[Polynomial]] = [
        [dict() for _ in range(n_parameters)] for _ in range(n_parameters)
    ]
    for exponent, coefficient in coefficients.items():
        if any(power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative")
        value = float(coefficient)
        for first in range(n_parameters):
            if exponent[first] == 0:
                continue
            for second in range(n_parameters):
                if first == second:
                    multiplicity = exponent[first] * (exponent[first] - 1)
                else:
                    multiplicity = exponent[first] * exponent[second]
                if multiplicity == 0:
                    continue
                derivative_exponent = list(exponent)
                derivative_exponent[first] -= 1
                derivative_exponent[second] -= 1
                key = tuple(derivative_exponent)
                hessian[first][second][key] = (
                    hessian[first][second].get(key, 0.0)
                    + value * multiplicity
                )
    return tuple(tuple(row) for row in hessian)


def evaluate_multivariate_polynomial(
    coefficients: Polynomial,
    point: np.ndarray | list[float] | tuple[float, ...],
) -> float:
    """Evaluate a sparse multivariate polynomial at one point."""
    n_parameters = _parameter_count(coefficients)
    values = np.asarray(point, dtype=float)
    if values.shape != (n_parameters,):
        raise ValueError("point dimension does not match polynomial")

    total = 0.0
    for exponent, coefficient in coefficients.items():
        term = float(coefficient)
        for value, power in zip(values, exponent):
            if power:
                term *= value**power
        total += term
    return total


def evaluate_multivariate_polynomial_gradient(
    coefficients: Polynomial,
    point: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Evaluate the exact gradient of a sparse multivariate polynomial."""
    gradient_coefficients = multivariate_polynomial_gradient_coefficients(
        coefficients
    )
    return np.asarray(
        [
            evaluate_multivariate_polynomial(component, point)
            if component
            else 0.0
            for component in gradient_coefficients
        ],
        dtype=float,
    )


def evaluate_multivariate_polynomial_hessian(
    coefficients: Polynomial,
    point: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Evaluate the exact Hessian of a sparse multivariate polynomial."""
    hessian_coefficients = multivariate_polynomial_hessian_coefficients(
        coefficients
    )
    n_parameters = len(hessian_coefficients)
    result = np.zeros((n_parameters, n_parameters), dtype=float)
    for first, row in enumerate(hessian_coefficients):
        for second, component in enumerate(row):
            if component:
                result[first, second] = evaluate_multivariate_polynomial(
                    component, point
                )
    return result


def gradient_flow_velocity_acceleration(
    coefficients: Polynomial,
    point: np.ndarray | list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Local velocity and acceleration for gradient flow ``dot(eps)=-grad L``."""
    gradient = evaluate_multivariate_polynomial_gradient(coefficients, point)
    hessian = evaluate_multivariate_polynomial_hessian(coefficients, point)
    velocity = -gradient
    acceleration = hessian @ gradient
    return velocity, acceleration


def two_link_leading_completion_time(
    coefficient: float,
    initial_first: float,
    initial_second: float,
    threshold: float,
) -> float:
    """Exact gradient-flow completion time for ``L = c*x*y`` with ``c < 0``.

    Gradient flow obeys ``dx/ds = C*y`` and ``dy/ds = C*x`` where ``C=-c``.
    The invariant ``x^2-y^2`` reduces route completion to a closed form. A route
    is complete when both positive link coordinates reach ``threshold``.
    """
    c = float(coefficient)
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    first = float(initial_first)
    second = float(initial_second)
    target = float(threshold)
    if first < 0.0 or second < 0.0:
        raise ValueError("initial link values must be non-negative")
    if target <= 0.0:
        raise ValueError("threshold must be positive")

    smaller = min(first, second)
    larger = max(first, second)
    if smaller >= target:
        return 0.0
    if larger == 0.0:
        return inf

    strength = -c
    invariant = larger * larger - smaller * smaller
    if invariant == 0.0:
        if smaller == 0.0:
            return inf
        return log(target / smaller) / strength

    root = sqrt(invariant)
    return (
        asinh(target / root) - asinh(smaller / root)
    ) / strength


def quadratic_crossing_time(
    position: float,
    velocity: float,
    acceleration: float,
    threshold: float,
    *,
    acceleration_atol: float = 1e-15,
) -> float:
    """Positive crossing time under a local constant-acceleration expansion."""
    x = float(position)
    v = float(velocity)
    a = float(acceleration)
    target = float(threshold)
    if target <= 0.0:
        raise ValueError("threshold must be positive")
    if acceleration_atol < 0.0:
        raise ValueError("acceleration_atol must be non-negative")
    if x >= target:
        return 0.0

    distance = target - x
    if abs(a) <= acceleration_atol:
        return distance / v if v > 0.0 else inf

    discriminant = v * v + 2.0 * a * distance
    if discriminant < 0.0:
        return inf
    root = sqrt(discriminant)
    candidates = [(-v + root) / a, (-v - root) / a]
    positive = [time for time in candidates if time >= 0.0]
    return min(positive) if positive else inf


def leading_two_link_route_times(
    coefficients: Polynomial,
    point: np.ndarray | list[float] | tuple[float, ...],
    threshold: float,
    *,
    coefficient_atol: float = 1e-12,
) -> list[tuple[tuple[int, int], float, float]]:
    """Completion times for beneficial square-free degree-two leading routes.

    Returns ``[(parameter_pair, coefficient, time), ...]`` sorted by predicted
    completion time. Only negative monomials of the form ``c*eps_i*eps_j`` are
    included; repeated-parameter terms and higher-degree constructions require a
    different flow reduction.
    """
    n_parameters = _parameter_count(coefficients)
    values = np.asarray(point, dtype=float)
    if values.shape != (n_parameters,):
        raise ValueError("point dimension does not match polynomial")
    if coefficient_atol < 0.0:
        raise ValueError("coefficient_atol must be non-negative")

    routes: list[tuple[tuple[int, int], float, float]] = []
    for exponent, coefficient in coefficients.items():
        if sum(exponent) != 2 or coefficient >= -coefficient_atol:
            continue
        active = [index for index, power in enumerate(exponent) if power]
        if len(active) != 2 or any(exponent[index] != 1 for index in active):
            continue
        first, second = active
        time = two_link_leading_completion_time(
            coefficient,
            values[first],
            values[second],
            threshold,
        )
        routes.append(((first, second), float(coefficient), time))
    routes.sort(key=lambda route: (route[2], route[0]))
    return routes


def quadratic_two_link_route_times(
    coefficients: Polynomial,
    point: np.ndarray | list[float] | tuple[float, ...],
    threshold: float,
    *,
    coefficient_atol: float = 1e-12,
) -> list[tuple[tuple[int, int], float, float]]:
    """Curvature-corrected local completion times for leading two-link routes.

    Route identities come from the beneficial square-free degree-two support,
    while crossing times use the full polynomial's local gradient and Hessian.
    This is a second-order trajectory predictor, not an exact integration of the
    full gradient flow.
    """
    values = np.asarray(point, dtype=float)
    routes = leading_two_link_route_times(
        coefficients,
        values,
        threshold,
        coefficient_atol=coefficient_atol,
    )
    velocity, acceleration = gradient_flow_velocity_acceleration(
        coefficients, values
    )

    corrected: list[tuple[tuple[int, int], float, float]] = []
    for pair, coefficient, _ in routes:
        crossing_times = [
            quadratic_crossing_time(
                values[parameter],
                velocity[parameter],
                acceleration[parameter],
                threshold,
            )
            for parameter in pair
        ]
        corrected.append((pair, coefficient, max(crossing_times)))
    corrected.sort(key=lambda route: (route[2], route[0]))
    return corrected
