import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient
from memory_frontier.support_invariants import (
    evaluate_quadratic_invariant,
    quadratic_invariant_derivative,
    quadratic_invariant_discretization_defect_coefficients,
    quadratic_invariant_discretization_degree,
    quadratic_invariant_gradient_descent_step_change,
)


def _second_order_pattern_source():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    emissions = np.column_stack([1.0 - p1, p1])
    transitions = np.array(
        [[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int
    )
    return UnifilarSource(emissions, transitions)


def _shared_route_polynomial():
    source = _second_order_pattern_source()
    k = 4
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = np.zeros((3, k, 2, k), dtype=float)
    directions[0, 0, 0, 0] = -1.0
    directions[0, 0, 0, 1] = 1.0
    directions[1, 1, 0, 0] = -1.0
    directions[1, 1, 0, 2] = 1.0
    directions[2, 1, 1, 0] = -1.0
    directions[2, 1, 1, 3] = 1.0
    readout = np.full((k, 2), 0.5, dtype=float)
    readout[2] = np.array([0.9, 0.1])
    readout[3] = np.array([0.2, 0.8])
    return multivariate_controller_loss_coefficients(
        source, base, directions, readout, horizon=12
    )


def _leading_polynomial(coefficients, atol=1e-12):
    active = {
        exponent: coefficient
        for exponent, coefficient in coefficients.items()
        if sum(exponent) > 0 and abs(coefficient) > atol
    }
    degree = min(sum(exponent) for exponent in active)
    return {
        exponent: coefficient
        for exponent, coefficient in active.items()
        if sum(exponent) == degree
    }


def _rk4(coefficients, point, time=1.0, steps=50):
    point = np.asarray(point, dtype=float).copy()
    h = float(time) / int(steps)

    def velocity(value):
        return -evaluate_multivariate_polynomial_gradient(coefficients, value)

    for _ in range(int(steps)):
        k1 = velocity(point)
        k2 = velocity(point + 0.5 * h * k1)
        k3 = velocity(point + 0.5 * h * k2)
        k4 = velocity(point + h * k3)
        point += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return point


def _gradient_descent(coefficients, point, step_size, time=1.0):
    point = np.asarray(point, dtype=float).copy()
    steps = int(round(float(time) / float(step_size)))
    assert np.isclose(steps * step_size, time, atol=1e-14)
    for _ in range(steps):
        point -= step_size * evaluate_multivariate_polynomial_gradient(
            coefficients, point
        )
    return point


def test_one_step_quadratic_identity_matches_direct_gradient_descent():
    coefficients = {
        (2, 1): 0.3,
        (0, 2): -0.4,
        (1, 0): 0.2,
        (0, 0): 1.7,
    }
    point = np.array([0.31, 0.47])
    weights = np.array([1.2, -0.7])
    eta = 0.08

    gradient = evaluate_multivariate_polynomial_gradient(coefficients, point)
    updated = point - eta * gradient
    direct = (
        evaluate_quadratic_invariant(updated, weights)
        - evaluate_quadratic_invariant(point, weights)
    )
    oracle = quadratic_invariant_gradient_descent_step_change(
        coefficients, point, weights, eta
    )
    assert np.isclose(oracle, direct, rtol=1e-13, atol=1e-14)


def test_exact_continuous_balance_contracts_under_finite_steps_for_xy_route():
    strength = 0.37
    coefficients = {(1, 1): -strength}
    weights = np.array([1.0, -1.0])
    point = np.array([0.42, 0.19])
    eta = 0.11

    assert np.isclose(
        quadratic_invariant_derivative(coefficients, point, weights),
        0.0,
        atol=1e-15,
    )
    defect = quadratic_invariant_discretization_defect_coefficients(
        coefficients, weights
    )
    assert np.isclose(defect[(0, 2)], strength**2, atol=1e-15)
    assert np.isclose(defect[(2, 0)], -strength**2, atol=1e-15)

    gradient = evaluate_multivariate_polynomial_gradient(coefficients, point)
    updated = point - eta * gradient
    initial_q = evaluate_quadratic_invariant(point, weights)
    updated_q = evaluate_quadratic_invariant(updated, weights)
    assert np.isclose(
        updated_q,
        (1.0 - eta**2 * strength**2) * initial_q,
        rtol=1e-14,
        atol=1e-15,
    )


def test_shared_leading_routes_have_degree_two_discretization_defect():
    full = _shared_route_polynomial()
    leading = _leading_polynomial(full)
    weights = np.array([1.0, -1.0, -1.0])

    # Leading flow conserves x^2-y^2-z^2 exactly, but vanilla gradient descent
    # breaks it at second order in step size through a degree-two defect.
    for point in ((0.2, 0.3, 0.4), (0.5, 0.7, 0.6)):
        assert np.isclose(
            quadratic_invariant_derivative(leading, point, weights),
            0.0,
            atol=1e-14,
        )
    assert quadratic_invariant_discretization_degree(leading, weights) == 2


def test_full_shared_route_discrete_minus_flow_drift_is_linear_in_step_size():
    coefficients = _shared_route_polynomial()
    weights = np.array([1.0, -1.0, -1.0])
    delta = 1e-4
    initial = delta * np.array([1.0, 0.7, 0.5])
    continuous = _rk4(coefficients, initial)

    step_sizes = np.array([0.1, 0.05, 0.025, 0.0125])
    discrepancies = []
    for eta in step_sizes:
        discrete = _gradient_descent(coefficients, initial, eta)
        discrepancies.append(
            abs(
                evaluate_quadratic_invariant(discrete, weights)
                - evaluate_quadratic_invariant(continuous, weights)
            )
            / delta**2
        )

    slope = np.polyfit(np.log(step_sizes), np.log(discrepancies), 1)[0]
    # For leading degree d=2 and discretization-defect degree r=2, accumulation
    # over a fixed physical-time window gives normalized error O(eta).
    assert np.isclose(slope, 1.0, atol=0.02)
