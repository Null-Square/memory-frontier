import numpy as np

from memory_frontier import delayed_repeat_source
from memory_frontier.order_barrier import (
    binary_chain_readout,
    binary_soft_chain_transition,
)
from memory_frontier.perturbative import multivariate_controller_loss_coefficients


def _parallel_two_link_fixture():
    source = delayed_repeat_source(2, 2, 0.1)
    k = 5  # 0, A1, A2, B1, B2
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    directions = []

    # Route A: x enters A1 on token 0, y advances A1 -> A2.
    direction = np.zeros_like(base)
    direction[0, 0, 0] = -1.0
    direction[0, 0, 1] = 1.0
    directions.append(direction)

    direction = np.zeros_like(base)
    direction[1, :, 0] = -1.0
    direction[1, :, 2] = 1.0
    directions.append(direction)

    # Route B: u enters B1 on token 0, v advances B1 -> B2.
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
    return source, base, np.asarray(directions), readout


def _polynomial_gradient(coefficients, point):
    point = np.asarray(point, dtype=float)
    gradient = np.zeros_like(point)
    for exponent, coefficient in coefficients.items():
        exponent = np.asarray(exponent, dtype=int)
        for parameter, multiplicity in enumerate(exponent):
            if multiplicity == 0:
                continue
            term = float(coefficient) * multiplicity
            for index, power in enumerate(exponent):
                derivative_power = power - (1 if index == parameter else 0)
                if derivative_power:
                    term *= point[index] ** derivative_power
            gradient[parameter] += term
    return gradient


def _projected_race(coefficients, scale, *, learning_rate=0.02, threshold=0.02):
    point = np.array(
        [scale, scale, scale**0.1, scale**2.0], dtype=float
    )
    for step in range(1_000):
        route_a = min(point[0], point[1]) >= threshold
        route_b = min(point[2], point[3]) >= threshold
        if route_a or route_b:
            return step, point, route_a, route_b

        point -= learning_rate * _polynomial_gradient(coefficients, point)
        point = np.maximum(point, 0.0)

        # x and u share one transition row, so preserve x + u <= 1.
        entrance_mass = point[0] + point[2]
        if entrance_mass > 1.0:
            point[[0, 2]] /= entrance_mass
        point[[1, 3]] = np.minimum(point[[1, 3]], 1.0)

    return 1_000, point, False, False


def test_weighted_leading_loss_monomial_does_not_predict_first_built_route():
    source, base, directions, readout = _parallel_two_link_fixture()
    coefficients = multivariate_controller_loss_coefficients(
        source, base, directions, readout, horizon=12
    )

    leading = {
        exponent: coefficient
        for exponent, coefficient in coefficients.items()
        if sum(exponent) == 2 and abs(coefficient) > 1e-12
    }
    assert set(leading) == {(1, 1, 0, 0), (0, 0, 1, 1)}
    assert np.isclose(
        leading[(1, 1, 0, 0)], leading[(0, 0, 1, 1)], atol=1e-12
    )
    assert leading[(1, 1, 0, 0)] < 0.0

    weights = np.array([1.0, 1.0, 0.1, 2.0])
    route_a_order = float(np.dot(weights, (1, 1, 0, 0)))
    route_b_order = float(np.dot(weights, (0, 0, 1, 1)))
    assert route_a_order == 2.0
    assert np.isclose(route_b_order, 2.1)
    assert route_a_order < route_b_order

    # Nevertheless the exact gradient vector field strongly favors v in route B.
    scale = 1e-12
    point = np.array([scale, scale, scale**0.1, scale**2.0])
    gradient = _polynomial_gradient(coefficients, point)
    assert abs(gradient[3]) > 1e9 * max(abs(gradient[0]), abs(gradient[1]))

    steps, final_point, route_a, route_b = _projected_race(coefficients, scale)
    assert steps < 500
    assert route_b
    assert not route_a
    assert min(final_point[2], final_point[3]) >= 0.02
    assert min(final_point[0], final_point[1]) < 0.02


def test_reused_parameter_can_appear_with_exponent_greater_than_one():
    source = delayed_repeat_source(2, 2, 0.1)
    base = binary_soft_chain_transition(2, 0.0)
    full = binary_soft_chain_transition(2, 1.0)
    directions = np.asarray([full - base])
    readout = binary_chain_readout(2, 0.4)

    coefficients = multivariate_controller_loss_coefficients(
        source, base, directions, readout, horizon=10
    )
    nonconstant = {
        exponent: coefficient
        for exponent, coefficient in coefficients.items()
        if sum(exponent) > 0 and abs(coefficient) > 1e-12
    }

    assert (1,) not in nonconstant
    assert (2,) in nonconstant
    assert nonconstant[(2,)] < 0.0


def test_support_is_extracted_after_exact_coefficient_cancellation():
    source = delayed_repeat_source(2, 1, 0.2)
    k = 3
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    direction_zero = np.zeros_like(base)
    direction_zero[0, 0, 0] = -1.0
    direction_zero[0, 0, 1] = 1.0
    direction_one = np.zeros_like(base)
    direction_one[0, 1, 0] = -1.0
    direction_one[0, 1, 2] = 1.0

    readout = np.full((k, 2), 0.5, dtype=float)
    readout[1] = np.array([0.8, 0.2])
    readout[2] = np.array([0.6289589607495853, 1.0 - 0.6289589607495853])

    coefficients = multivariate_controller_loss_coefficients(
        source,
        base,
        np.asarray([direction_zero + direction_one]),
        readout,
        horizon=8,
    )

    assert all(
        abs(coefficient) < 1e-11
        for exponent, coefficient in coefficients.items()
        if sum(exponent) > 0
    )
