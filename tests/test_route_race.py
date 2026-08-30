import math

import numpy as np

from memory_frontier import delayed_repeat_source
from memory_frontier.order_barrier import binary_chain_readout
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import (
    evaluate_multivariate_polynomial_gradient,
    evaluate_multivariate_polynomial_hessian,
    leading_two_link_route_times,
    multivariate_polynomial_gradient_coefficients,
    multivariate_polynomial_hessian_coefficients,
    quadratic_two_link_route_times,
    two_link_leading_completion_time,
)


def _parallel_two_link_fixture():
    source = delayed_repeat_source(2, 2, 0.1)
    k = 5  # 0, A1, A2, B1, B2
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = []

    direction = np.zeros_like(base)
    direction[0, 0, 0] = -1.0
    direction[0, 0, 1] = 1.0
    directions.append(direction)

    direction = np.zeros_like(base)
    direction[1, :, 0] = -1.0
    direction[1, :, 2] = 1.0
    directions.append(direction)

    direction = np.zeros_like(base)
    direction[0, 0, 0] = -1.0
    direction[0, 0, 3] = 1.0
    directions.append(direction)

    direction = np.zeros_like(base)
    direction[3, :, 0] = -1.0
    direction[3, :, 4] = 1.0
    directions.append(direction)

    final_decoder = binary_chain_readout(2, 0.4)[2]
    readout = np.full((k, 2), 0.5, dtype=float)
    readout[2] = final_decoder
    readout[4] = final_decoder
    coefficients = multivariate_controller_loss_coefficients(
        source, base, np.asarray(directions), readout, horizon=12
    )
    return coefficients


def _projected_full_polynomial_race(
    coefficients,
    point,
    *,
    learning_rate=0.02,
    threshold=0.02,
    max_steps=1_000,
):
    point = np.asarray(point, dtype=float).copy()
    for step in range(max_steps + 1):
        route_a = min(point[0], point[1]) >= threshold
        route_b = min(point[2], point[3]) >= threshold
        if route_a or route_b:
            return step, point, route_a, route_b
        if step == max_steps:
            break

        point -= learning_rate * evaluate_multivariate_polynomial_gradient(
            coefficients, point
        )
        point = np.maximum(point, 0.0)

        # The two entrance links share one transition-probability row.
        entrance_mass = point[0] + point[2]
        if entrance_mass > 1.0:
            point[[0, 2]] /= entrance_mass
        point[[1, 3]] = np.minimum(point[[1, 3]], 1.0)
    return max_steps, point, False, False


def test_sparse_polynomial_gradient_and_hessian_are_exact():
    coefficients = {
        (2, 1): 3.0,
        (0, 2): -4.0,
        (0, 0): 7.0,
    }
    gradient = multivariate_polynomial_gradient_coefficients(coefficients)
    assert gradient[0] == {(1, 1): 6.0}
    assert gradient[1] == {(2, 0): 3.0, (0, 1): -8.0}
    assert np.allclose(
        evaluate_multivariate_polynomial_gradient(coefficients, [2.0, 3.0]),
        [36.0, -12.0],
    )

    hessian = multivariate_polynomial_hessian_coefficients(coefficients)
    assert hessian[0][0] == {(0, 1): 6.0}
    assert hessian[0][1] == {(1, 0): 6.0}
    assert hessian[1][0] == {(1, 0): 6.0}
    assert hessian[1][1] == {(0, 0): -8.0}
    assert np.allclose(
        evaluate_multivariate_polynomial_hessian(coefficients, [2.0, 3.0]),
        [[18.0, 12.0], [12.0, -8.0]],
    )


def test_two_link_completion_time_matches_exact_gradient_flow_solution():
    coefficient = -0.1
    threshold = 0.02

    symmetric = two_link_leading_completion_time(
        coefficient, 1e-6, 1e-6, threshold
    )
    assert np.isclose(
        symmetric,
        math.log(threshold / 1e-6) / 0.1,
        atol=1e-12,
    )

    first = 0.04
    second = 1e-8
    asymmetric = two_link_leading_completion_time(
        coefficient, first, second, threshold
    )
    z = 0.1 * asymmetric
    evolved_first = first * math.cosh(z) + second * math.sinh(z)
    evolved_second = second * math.cosh(z) + first * math.sinh(z)
    assert np.isclose(evolved_second, threshold, atol=1e-12)
    assert evolved_first > threshold


def test_leading_gradient_route_time_predicts_full_polynomial_winner_b():
    coefficients = _parallel_two_link_fixture()
    scale = 1e-12
    point = np.array([scale, scale, scale**0.1, scale**2.0])

    routes = leading_two_link_route_times(coefficients, point, 0.02)
    assert [route[0] for route in routes[:2]] == [(2, 3), (0, 1)]
    assert routes[0][2] < routes[1][2]

    steps, _, route_a, route_b = _projected_full_polynomial_race(
        coefficients, point
    )
    assert steps < 500
    assert route_b
    assert not route_a


def test_leading_gradient_route_time_predicts_full_polynomial_winner_a():
    coefficients = _parallel_two_link_fixture()
    scale = 1e-12
    point = np.array([scale**0.1, scale**2.0, scale, scale])

    routes = leading_two_link_route_times(coefficients, point, 0.02)
    assert [route[0] for route in routes[:2]] == [(0, 1), (2, 3)]
    assert routes[0][2] < routes[1][2]

    steps, _, route_a, route_b = _projected_full_polynomial_race(
        coefficients, point
    )
    assert steps < 500
    assert route_a
    assert not route_b


def test_curvature_correction_recovers_a_leading_route_race_reversal():
    coefficients = _parallel_two_link_fixture()
    scale = 1e-8
    point = scale ** np.array([0.2, 1.0, 0.6, 0.2])

    leading = leading_two_link_route_times(coefficients, point, 0.02)
    assert [route[0] for route in leading[:2]] == [(2, 3), (0, 1)]
    assert leading[0][2] < leading[1][2]

    corrected = quadratic_two_link_route_times(coefficients, point, 0.02)
    assert [route[0] for route in corrected[:2]] == [(0, 1), (2, 3)]
    assert corrected[0][2] < corrected[1][2]

    steps, _, route_a, route_b = _projected_full_polynomial_race(
        coefficients, point
    )
    assert steps < 500
    assert route_a
    assert not route_b
