"""Deterministic audit of construction-order escape-time scaling.

This experiment stays outside CI. For the delay-5 functional-equivalence
spectrum, it compares the exact leading-monomial gradient-flow completion time
against RK4 integration of the full exact finite-horizon loss polynomial.
"""

from __future__ import annotations

import numpy as np

from memory_frontier import delayed_repeat_source
from memory_frontier.construction_time import symmetric_square_free_completion_time
from memory_frontier.order_barrier import binary_chain_readout
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import multivariate_polynomial_gradient_coefficients


def partial_scaffold_family(delay: int, missing_prefix: int):
    k = delay + 1
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    for link in range(missing_prefix + 1, delay + 1):
        state = link - 1
        base[state, :, :] = 0.0
        base[state, :, state + 1] = 1.0

    directions = []
    first = np.zeros_like(base)
    first[0, 0, 0] = -1.0
    first[0, 0, 1] = 1.0
    directions.append(first)
    for link in range(2, missing_prefix + 1):
        state = link - 1
        direction = np.zeros_like(base)
        direction[state, :, 0] = -1.0
        direction[state, :, state + 1] = 1.0
        directions.append(direction)
    return base, np.asarray(directions)


def compile_polynomial(coefficients, n_parameters):
    if not coefficients:
        return np.zeros((0, n_parameters), dtype=int), np.zeros(0, dtype=float)
    exponents = np.asarray(list(coefficients), dtype=int)
    values = np.asarray([coefficients[key] for key in coefficients], dtype=float)
    return exponents, values


def compile_gradient(coefficients, n_parameters):
    gradient = multivariate_polynomial_gradient_coefficients(coefficients)
    return [
        compile_polynomial(component, n_parameters)
        for component in gradient
    ]


def evaluate_compiled_gradient(compiled, point):
    result = np.zeros(len(compiled), dtype=float)
    for parameter, (exponents, coefficients) in enumerate(compiled):
        if len(coefficients) == 0:
            continue
        monomials = np.prod(point[None, :] ** exponents, axis=1)
        result[parameter] = float(coefficients @ monomials)
    return result


def rk4_completion_time(
    coefficients,
    degree,
    initial_scale,
    threshold,
    leading_time,
    *,
    steps=3_000,
    horizon_factor=1.5,
    coefficient_atol=1e-12,
):
    filtered = {
        exponent: float(coefficient)
        for exponent, coefficient in coefficients.items()
        if abs(float(coefficient)) > coefficient_atol
    }
    compiled = compile_gradient(filtered, degree)
    point = np.full(degree, initial_scale, dtype=float)
    dt = horizon_factor * leading_time / steps

    def vector_field(values):
        return -evaluate_compiled_gradient(compiled, values)

    for step in range(steps + 1):
        if np.min(point) >= threshold:
            return step * dt
        if step == steps:
            break
        k1 = vector_field(point)
        k2 = vector_field(point + 0.5 * dt * k1)
        k3 = vector_field(point + 0.5 * dt * k2)
        k4 = vector_field(point + dt * k3)
        point += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return float("nan")


def main():
    delay = 5
    horizon = 13
    source = delayed_repeat_source(2, delay, 0.1)
    readout = binary_chain_readout(delay, 0.4)
    initial_scale = 0.02
    threshold = 0.1

    print("degree leading_time full_time full/leading")
    for degree in range(1, delay + 1):
        base, directions = partial_scaffold_family(delay, degree)
        coefficients = multivariate_controller_loss_coefficients(
            source, base, directions, readout, horizon
        )
        leading_key = (1,) * degree
        leading_coefficient = float(coefficients[leading_key])
        leading_time = symmetric_square_free_completion_time(
            leading_coefficient,
            degree,
            initial_scale,
            threshold,
        )
        full_time = rk4_completion_time(
            coefficients,
            degree,
            initial_scale,
            threshold,
            leading_time,
        )
        print(
            f"{degree:>6d} "
            f"{leading_time:>14.6f} "
            f"{full_time:>14.6f} "
            f"{full_time / leading_time:>12.6f}"
        )


if __name__ == "__main__":
    main()
