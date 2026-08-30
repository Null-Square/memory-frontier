import numpy as np

from memory_frontier.construction_time import (
    balanced_exponent_vector_completion_time,
    monomial_gradient_flow_velocity,
    normalized_squared_coordinates,
    symmetric_square_free_completion_time,
    tied_repeated_parameter_completion_time,
)


def test_weighted_squared_balancedness_is_exact_for_repeated_exponents():
    coefficient = -0.37
    for exponents, point in (
        ((2, 1), (0.3, 0.4)),
        ((3, 2, 1), (0.2, 0.35, 0.5)),
        ((4, 1, 2), (0.25, 0.31, 0.46)),
    ):
        alpha = np.asarray(exponents, dtype=float)
        x = np.asarray(point, dtype=float)
        velocity = monomial_gradient_flow_velocity(
            coefficient, exponents, point
        )
        normalized_square_derivative = 2.0 * x * velocity / alpha
        assert np.allclose(
            normalized_square_derivative,
            normalized_square_derivative[0],
            rtol=1e-13,
            atol=1e-14,
        )


def test_normalized_squared_coordinate_differences_are_conserved_to_first_order():
    coefficient = -0.21
    exponents = (3, 1, 2)
    point = np.array([0.27, 0.41, 0.33], dtype=float)
    velocity = monomial_gradient_flow_velocity(
        coefficient, exponents, point
    )
    step = 1e-7

    before = normalized_squared_coordinates(exponents, point)
    after = normalized_squared_coordinates(
        exponents, point + step * velocity
    )
    before_differences = before - before[0]
    after_differences = after - after[0]

    # Forward Euler changes a continuous-time invariant only at O(step^2).
    assert np.max(np.abs(after_differences - before_differences)) < 1e-13


def test_square_free_balanced_flow_reduces_to_existing_completion_time():
    coefficient = -0.17
    initial = 0.004
    threshold = 0.08
    for degree in range(1, 6):
        exponents = (1,) * degree
        generalized = balanced_exponent_vector_completion_time(
            coefficient,
            exponents,
            initial,
            threshold,
        )
        square_free = symmetric_square_free_completion_time(
            coefficient,
            degree,
            initial,
            threshold,
        )
        assert np.isclose(generalized, square_free, rtol=1e-13, atol=1e-13)


def test_single_repeated_parameter_matches_tied_flow_under_normalized_coordinates():
    coefficient = -0.19
    normalized_initial = 0.006
    normalized_threshold = 0.07

    for degree in range(1, 6):
        generalized = balanced_exponent_vector_completion_time(
            coefficient,
            (degree,),
            normalized_initial,
            normalized_threshold,
        )
        scale = np.sqrt(degree)
        tied = tied_repeated_parameter_completion_time(
            coefficient,
            degree,
            scale * normalized_initial,
            scale * normalized_threshold,
        )
        assert np.isclose(generalized, tied, rtol=1e-13, atol=1e-13)


def test_total_degree_controls_balanced_escape_exponent_while_multiplicity_changes_prefactor():
    coefficient = -0.2
    initial = 1e-5
    smaller = initial / 10.0
    threshold = 0.08

    # These exponent vectors all have total degree four but different
    # multiplicity structure.
    for exponents in ((1, 1, 1, 1), (2, 1, 1), (2, 2), (3, 1), (4,)):
        time = balanced_exponent_vector_completion_time(
            coefficient, exponents, initial, threshold
        )
        smaller_time = balanced_exponent_vector_completion_time(
            coefficient, exponents, smaller, threshold
        )
        # Degree four gives delta^-2 divergence for every multiplicity pattern.
        assert np.isclose(smaller_time / time, 100.0, rtol=2e-6)

    square_free = balanced_exponent_vector_completion_time(
        coefficient, (1, 1, 1, 1), initial, threshold
    )
    repeated = balanced_exponent_vector_completion_time(
        coefficient, (4,), initial, threshold
    )
    assert repeated < square_free
