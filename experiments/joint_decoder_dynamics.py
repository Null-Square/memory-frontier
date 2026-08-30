"""Exact smooth joint decoder/transition gradient-flow audit.

This experiment stays outside CI. On the frozen non-delayed suffix-01 witness,
the finite-horizon loss factorizes exactly as

    L(a, eps) = log(2) + f(a) * R(eps),

where ``a`` is decoder logit contrast, ``R`` is an exact transition occupancy
polynomial, and ``f(a)=log(cosh(a))-(3/5)a``. We integrate the resulting exact
joint vector field with fixed-step RK4 and compare against the leading monomial
construction-time oracle.
"""

from __future__ import annotations

import math

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.construction_time import symmetric_square_free_completion_time
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import multivariate_polynomial_gradient_coefficients


def second_order_pattern_source():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    emissions = np.column_stack([1.0 - p1, p1])
    transitions = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int)
    return UnifilarSource(emissions, transitions)


def pattern_controller():
    k = 3
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    first = np.zeros_like(base)
    first[0, 0, 0] = -1.0
    first[0, 0, 1] = 1.0

    second = np.zeros_like(base)
    second[1, 1, 0] = -1.0
    second[1, 1, 2] = 1.0
    return base, first, second


def readout_from_gap(gap: float):
    q1 = 1.0 / (1.0 + math.exp(-2.0 * gap))
    readout = np.full((3, 2), 0.5, dtype=float)
    readout[2] = np.array([1.0 - q1, q1])
    return readout


def decoder_excess_cost(gap: float) -> float:
    return math.log(math.cosh(gap)) - (3.0 / 5.0) * gap


def decoder_excess_gradient(gap: float) -> float:
    return math.tanh(gap) - (3.0 / 5.0)


def compile_polynomial(coefficients, n_parameters):
    if not coefficients:
        return np.zeros((0, n_parameters), dtype=int), np.zeros(0, dtype=float)
    exponents = np.asarray(list(coefficients), dtype=int)
    values = np.asarray([coefficients[key] for key in coefficients], dtype=float)
    return exponents, values


def evaluate_compiled(compiled, point):
    exponents, coefficients = compiled
    if len(coefficients) == 0:
        return 0.0
    monomials = np.prod(point[None, :] ** exponents, axis=1)
    return float(coefficients @ monomials)


def exact_occupancy_polynomial(prewired: bool):
    source = second_order_pattern_source()
    base, first, second = pattern_controller()
    transition_base = base + second if prewired else base
    directions = np.asarray([first]) if prewired else np.asarray([first, second])

    reference_gap = 0.4
    reference_cost = decoder_excess_cost(reference_gap)
    loss_coefficients = multivariate_controller_loss_coefficients(
        source,
        transition_base,
        directions,
        readout_from_gap(reference_gap),
        horizon=12,
    )
    occupancy = {
        exponent: float(coefficient) / reference_cost
        for exponent, coefficient in loss_coefficients.items()
        if sum(exponent) > 0 and abs(float(coefficient)) > 1e-12
    }
    return occupancy


def compile_joint_vector_field(prewired: bool):
    occupancy = exact_occupancy_polynomial(prewired)
    n_parameters = len(next(iter(occupancy)))
    compiled_occupancy = compile_polynomial(occupancy, n_parameters)
    gradient_coefficients = multivariate_polynomial_gradient_coefficients(occupancy)
    compiled_gradient = [
        compile_polynomial(component, n_parameters)
        for component in gradient_coefficients
    ]

    def vector_field(point):
        gap = float(point[0])
        transition = np.asarray(point[1:], dtype=float)
        occupancy_value = evaluate_compiled(compiled_occupancy, transition)
        occupancy_gradient = np.asarray(
            [evaluate_compiled(component, transition) for component in compiled_gradient],
            dtype=float,
        )
        return np.concatenate(
            [
                np.array(
                    [-decoder_excess_gradient(gap) * occupancy_value],
                    dtype=float,
                ),
                -decoder_excess_cost(gap) * occupancy_gradient,
            ]
        )

    return vector_field


def rk4_completion_time(
    prewired: bool,
    initial_scale: float,
    threshold: float,
    leading_time: float,
    *,
    steps: int = 3_000,
    horizon_factor: float = 1.4,
):
    vector_field = compile_joint_vector_field(prewired)
    degree = 2 if prewired else 3
    point = np.full(degree, initial_scale, dtype=float)
    dt = horizon_factor * leading_time / steps

    for step in range(steps + 1):
        if np.min(point) >= threshold:
            return step * dt, point
        if step == steps:
            break
        k1 = vector_field(point)
        k2 = vector_field(point + 0.5 * dt * k1)
        k3 = vector_field(point + 0.5 * dt * k2)
        k4 = vector_field(point + dt * k3)
        point += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return float("nan"), point


def main():
    threshold = 0.1
    leading_coefficient = -1.0 / 14.0
    initial_scales = (0.005, 0.01, 0.02, 0.03)

    for prewired in (True, False):
        degree = 2 if prewired else 3
        label = "prewired" if prewired else "unscaffolded"
        print(f"\n{label}: joint leading degree {degree}")
        print("initial leading_time full_time full/leading")
        for initial in initial_scales:
            leading_time = symmetric_square_free_completion_time(
                leading_coefficient,
                degree,
                initial,
                threshold,
            )
            full_time, _ = rk4_completion_time(
                prewired,
                initial,
                threshold,
                leading_time,
            )
            print(
                f"{initial:>7.4f} "
                f"{leading_time:>12.6f} "
                f"{full_time:>12.6f} "
                f"{full_time / leading_time:>12.6f}"
            )


if __name__ == "__main__":
    main()
