import math

import numpy as np

from memory_frontier.construction_time import (
    construction_time_divergence_power,
    symmetric_square_free_completion_time,
    tied_repeated_parameter_completion_time,
)


def _explicit_symmetric_solution(strength, degree, initial, time):
    if degree == 1:
        return initial + strength * time
    if degree == 2:
        return initial * math.exp(strength * time)
    return (
        initial ** (2 - degree)
        - strength * (degree - 2) * time
    ) ** (-1.0 / (degree - 2))


def test_symmetric_completion_time_matches_exact_gradient_flow_solution():
    coefficient = -0.17
    initial = 0.003
    threshold = 0.08
    strength = -coefficient

    for degree in range(1, 6):
        time = symmetric_square_free_completion_time(
            coefficient, degree, initial, threshold
        )
        evolved = _explicit_symmetric_solution(
            strength, degree, initial, time
        )
        assert np.isclose(evolved, threshold, rtol=1e-11, atol=1e-12)


def test_construction_order_gives_distinct_escape_time_asymptotics():
    coefficient = -0.2
    threshold = 0.1
    initial = 1e-6
    smaller = initial / 10.0

    # Degree 1 remains finite as initialization vanishes.
    t1 = symmetric_square_free_completion_time(
        coefficient, 1, initial, threshold
    )
    t1_smaller = symmetric_square_free_completion_time(
        coefficient, 1, smaller, threshold
    )
    assert np.isclose(t1_smaller / t1, 1.0, rtol=1e-4)

    # Degree 2 diverges only logarithmically, so a decade reduction adds a
    # constant log(10)/C rather than multiplying time by a power of ten.
    t2 = symmetric_square_free_completion_time(
        coefficient, 2, initial, threshold
    )
    t2_smaller = symmetric_square_free_completion_time(
        coefficient, 2, smaller, threshold
    )
    assert np.isclose(
        t2_smaller - t2,
        math.log(10.0) / (-coefficient),
        rtol=1e-12,
    )

    # Degree d >= 3 has power-law divergence initial**(-(d-2)).
    for degree in (3, 4, 5):
        time = symmetric_square_free_completion_time(
            coefficient, degree, initial, threshold
        )
        smaller_time = symmetric_square_free_completion_time(
            coefficient, degree, smaller, threshold
        )
        expected_ratio = 10.0 ** (degree - 2)
        assert np.isclose(
            smaller_time / time,
            expected_ratio,
            rtol=2e-4,
        )
        assert construction_time_divergence_power(degree) == degree - 2

    assert construction_time_divergence_power(1) is None
    assert construction_time_divergence_power(2) is None


def test_parameter_tying_changes_gradient_time_without_changing_diagonal_loss_curve():
    coefficient = -0.13
    initial = 0.004
    threshold = 0.07

    for degree in range(1, 6):
        independent_time = symmetric_square_free_completion_time(
            coefficient, degree, initial, threshold
        )
        tied_time = tied_repeated_parameter_completion_time(
            coefficient, degree, initial, threshold
        )
        assert np.isclose(
            tied_time,
            independent_time / degree,
            rtol=1e-12,
            atol=1e-12,
        )


def test_zero_initialization_is_stuck_only_when_multiple_factors_are_required():
    coefficient = -0.1
    threshold = 0.05
    assert np.isclose(
        symmetric_square_free_completion_time(
            coefficient, 1, 0.0, threshold
        ),
        threshold / (-coefficient),
    )
    for degree in (2, 3, 4, 5):
        assert math.isinf(
            symmetric_square_free_completion_time(
                coefficient, degree, 0.0, threshold
            )
        )
