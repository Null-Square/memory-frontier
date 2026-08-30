import math

import numpy as np

from memory_frontier.softmax_boundary import (
    symmetric_logit_boundary_leading_time,
    symmetric_logit_monomial_completion_time,
    symmetric_logit_monomial_logit_velocity,
    symmetric_logit_monomial_probability_velocity,
)


def test_logit_and_probability_velocity_match_chain_rule():
    coefficient = -0.37
    degree = 4
    probability = 0.08

    logit_velocity = symmetric_logit_monomial_logit_velocity(
        coefficient, degree, probability
    )
    probability_velocity = symmetric_logit_monomial_probability_velocity(
        coefficient, degree, probability
    )

    expected_logit = (-coefficient) * probability**degree * (1.0 - probability)
    expected_probability = probability * (1.0 - probability) * expected_logit
    assert math.isclose(logit_velocity, expected_logit, rel_tol=1e-14)
    assert math.isclose(probability_velocity, expected_probability, rel_tol=1e-14)


def test_exact_completion_time_has_delta_to_minus_d_asymptotic():
    coefficient = -0.41
    threshold = 0.12

    for degree in range(1, 6):
        delta = 1e-4
        exact = symmetric_logit_monomial_completion_time(
            coefficient, degree, delta, threshold
        )
        leading = symmetric_logit_boundary_leading_time(
            coefficient, degree, delta
        )
        # The correction is O(delta log(1/delta)) for d=1 and O(delta) for the
        # higher degrees in this fixed-threshold comparison.
        assert abs(exact / leading - 1.0) < 8e-4


def test_fitted_small_probability_time_exponents_are_degrees_one_through_five():
    coefficient = -0.23
    threshold = 0.08
    deltas = np.asarray([0.0015, 0.0022, 0.0033, 0.0050], dtype=float)

    for degree in range(1, 6):
        times = np.asarray(
            [
                symmetric_logit_monomial_completion_time(
                    coefficient, degree, float(delta), threshold
                )
                for delta in deltas
            ]
        )
        slope = float(np.polyfit(np.log(deltas), np.log(times), 1)[0])
        assert abs(slope + degree) < 0.08


def test_each_unit_of_scaffolding_improves_logit_boundary_exponent_by_one():
    coefficient = -0.5
    threshold = 0.1
    delta = 5e-5

    for degree in range(2, 6):
        slower = symmetric_logit_monomial_completion_time(
            coefficient, degree, delta, threshold
        )
        faster = symmetric_logit_monomial_completion_time(
            coefficient, degree - 1, delta, threshold
        )
        # tau_d / tau_{d-1} ~ ((d-1)/d) delta^{-1}
        rescaled = (slower / faster) * delta
        expected = (degree - 1) / degree
        assert abs(rescaled - expected) < 0.01


def test_probability_coordinate_speed_is_two_powers_smaller_under_logit_flow():
    coefficient = -0.7
    degree = 3
    probabilities = np.asarray([1e-2, 5e-3, 2.5e-3])

    # Direct Euclidean probability flow for a symmetric square-free monomial is
    # C p^(d-1). Logit-Euclidean flow mapped into probability is
    # C p^(d+1)(1-p)^2. Their ratio is exactly p^2(1-p)^2.
    for probability in probabilities:
        logit_induced = symmetric_logit_monomial_probability_velocity(
            coefficient, degree, float(probability)
        )
        direct_probability = (-coefficient) * probability ** (degree - 1)
        ratio = logit_induced / direct_probability
        assert math.isclose(
            ratio,
            probability**2 * (1.0 - probability) ** 2,
            rel_tol=1e-13,
        )
