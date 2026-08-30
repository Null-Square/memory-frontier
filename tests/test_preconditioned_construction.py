import numpy as np

from memory_frontier.construction_time import (
    balanced_exponent_vector_completion_time,
    construction_time_divergence_power,
    monomial_gradient_flow_velocity,
)
from memory_frontier.order_barrier import binary_delay_chain_leading_gain_coefficient
from memory_frontier.preconditioned_construction import (
    diagonal_preconditioned_monomial_velocity,
    preconditioned_balanced_exponent_vector_completion_time,
    preconditioned_normalized_squared_coordinates,
)


def test_preconditioned_velocity_is_metric_times_euclidean_velocity():
    coefficient = -0.7
    exponents = (2, 1, 3)
    metric = np.array([0.5, 2.0, 1.7])
    point = np.array([0.4, 0.3, 0.2])

    euclidean = monomial_gradient_flow_velocity(coefficient, exponents, point)
    preconditioned = diagonal_preconditioned_monomial_velocity(
        coefficient, exponents, metric, point
    )

    assert np.allclose(preconditioned, metric * euclidean, atol=1e-15)


def test_metric_normalized_square_derivatives_are_identical():
    coefficient = -0.9
    alpha = np.array([2, 1, 3], dtype=int)
    metric = np.array([0.5, 2.0, 1.7])
    point = np.array([0.4, 0.3, 0.2])
    velocity = diagonal_preconditioned_monomial_velocity(
        coefficient, alpha, metric, point
    )

    derivatives = 2.0 * point * velocity / (metric * alpha)
    product = float(np.prod(point ** alpha))
    expected = 2.0 * (-coefficient) * product

    assert np.allclose(derivatives, expected, atol=2e-15)


def test_metric_balanced_manifold_reduces_to_one_scalar_flow():
    coefficient = -0.7
    alpha = np.array([2, 1, 3], dtype=int)
    metric = np.array([0.5, 2.0, 1.7])
    scale = 0.2
    combined = metric * alpha
    point = np.sqrt(combined) * scale

    normalized = preconditioned_normalized_squared_coordinates(
        alpha, metric, point
    )
    assert np.allclose(normalized, scale**2, atol=2e-15)

    velocity = diagonal_preconditioned_monomial_velocity(
        coefficient, alpha, metric, point
    )
    scalar_velocities = velocity / np.sqrt(combined)
    prefactor = float(np.prod(combined ** (alpha.astype(float) / 2.0)))
    expected = (-coefficient) * prefactor * scale ** (int(alpha.sum()) - 1)

    assert np.allclose(scalar_velocities, expected, rtol=0.0, atol=2e-15)


def test_unit_metric_reduces_to_existing_exponent_vector_completion_time():
    coefficient = -0.3
    exponents = (3, 2, 1)
    initial = 0.02
    threshold = 0.1

    old = balanced_exponent_vector_completion_time(
        coefficient, exponents, initial, threshold
    )
    new = preconditioned_balanced_exponent_vector_completion_time(
        coefficient,
        exponents,
        np.ones(len(exponents)),
        initial,
        threshold,
    )

    assert np.isclose(new, old, rtol=0.0, atol=2e-13)


def test_finite_memory_degree_four_route_keeps_delta_minus_two_class():
    degree = 4
    gain = binary_delay_chain_leading_gain_coefficient(
        degree, 0.1, 12, 0.4
    )
    coefficient = -gain
    alpha = np.ones(degree, dtype=int)
    metric_a = np.ones(degree)
    metric_b = np.array([0.5, 2.0, 1.5, 3.0])
    initial = 0.01
    threshold = 0.1

    time_a = preconditioned_balanced_exponent_vector_completion_time(
        coefficient, alpha, metric_a, initial, threshold
    )
    time_b = preconditioned_balanced_exponent_vector_completion_time(
        coefficient, alpha, metric_b, initial, threshold
    )

    prefactor_a = float(np.prod(np.sqrt(metric_a)))
    prefactor_b = float(np.prod(np.sqrt(metric_b)))
    assert np.isclose(time_a * prefactor_a, time_b * prefactor_b, atol=2e-11)
    assert construction_time_divergence_power(degree) == 2


def test_metric_weights_must_be_strictly_positive():
    with np.testing.assert_raises(ValueError):
        preconditioned_balanced_exponent_vector_completion_time(
            -1.0, (1, 1), (1.0, 0.0), 0.01, 0.1
        )
