import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient
from memory_frontier.support_invariants import (
    evaluate_quadratic_invariant,
    quadratic_invariant_breaking_degree,
    quadratic_invariant_derivative_coefficients,
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

    # Shared entrance x after symbol 0.
    directions[0, 0, 0, 0] = -1.0
    directions[0, 0, 0, 1] = 1.0
    # Exit y recognizes suffix 00.
    directions[1, 1, 0, 0] = -1.0
    directions[1, 1, 0, 2] = 1.0
    # Exit z recognizes suffix 01.
    directions[2, 1, 1, 0] = -1.0
    directions[2, 1, 1, 3] = 1.0

    readout = np.full((k, 2), 0.5, dtype=float)
    readout[2] = np.array([0.9, 0.1])
    readout[3] = np.array([0.2, 0.8])
    return multivariate_controller_loss_coefficients(
        source, base, directions, readout, horizon=12
    )


def _rk4_full_polynomial(coefficients, point, time=1.0, steps=50):
    point = np.asarray(point, dtype=float).copy()
    step = float(time) / int(steps)

    def velocity(value):
        return -evaluate_multivariate_polynomial_gradient(coefficients, value)

    for _ in range(int(steps)):
        k1 = velocity(point)
        k2 = velocity(point + 0.5 * step * k1)
        k3 = velocity(point + 0.5 * step * k2)
        k4 = velocity(point + step * k3)
        point += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return point


def test_derivative_polynomial_exposes_first_symmetry_breaking_degree():
    # The quadratic routes preserve Q=x^2-y^2+z^2. A cubic xyz term is the
    # first support that violates this balance law.
    coefficients = {
        (1, 1, 0): -0.4,
        (0, 1, 1): -0.7,
        (1, 1, 1): 0.3,
    }
    weights = np.array([1.0, -1.0, 1.0])
    derivative = quadratic_invariant_derivative_coefficients(
        coefficients, weights
    )

    assert derivative == {(1, 1, 1): -0.6}
    assert quadratic_invariant_breaking_degree(coefficients, weights) == 3


def test_exact_shared_route_polynomial_first_breaks_leading_balance_at_degree_three():
    coefficients = _shared_route_polynomial()
    weights = np.array([1.0, -1.0, -1.0])
    derivative = quadratic_invariant_derivative_coefficients(
        coefficients,
        weights,
        coefficient_atol=1e-12,
        orthogonality_atol=1e-12,
    )

    # Leading degree-two support xy,xz is exactly orthogonal to the balance
    # weight. The first nonzero invariant derivative terms are cubic.
    assert quadratic_invariant_breaking_degree(
        coefficients,
        weights,
        coefficient_atol=1e-12,
        orthogonality_atol=1e-12,
    ) == 3
    assert all(sum(exponent) >= 3 for exponent in derivative)
    assert any(sum(exponent) == 3 for exponent in derivative)


def test_full_shared_route_relative_drift_scales_with_degree_gap():
    coefficients = _shared_route_polynomial()
    weights = np.array([1.0, -1.0, -1.0])
    direction = np.array([1.0, 0.7, 0.5])

    scales = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    relative_drifts = []
    for scale in scales:
        initial = scale * direction
        final = _rk4_full_polynomial(coefficients, initial)
        drift = abs(
            evaluate_quadratic_invariant(final, weights)
            - evaluate_quadratic_invariant(initial, weights)
        )
        relative_drifts.append(drift / scale**2)

    slope = np.polyfit(np.log(scales), np.log(relative_drifts), 1)[0]
    # Leading degree d=2, first breaking degree p=3, so on a fixed local
    # construction-time window the normalized drift is O(delta^(p-d))=O(delta).
    assert np.isclose(slope, 1.0, atol=0.02)
