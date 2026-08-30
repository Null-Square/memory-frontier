import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.shared_routes import (
    shared_entrance_mode_coordinates,
    shared_entrance_quadratic_flow,
    shared_entrance_zero_exit_crossing_times,
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


def _leading_shared_strengths(coefficients, atol=1e-12):
    active = {
        exponent: coefficient
        for exponent, coefficient in coefficients.items()
        if sum(exponent) > 0 and abs(coefficient) > atol
    }
    degree = min(sum(exponent) for exponent in active)
    leading = {
        exponent: coefficient
        for exponent, coefficient in active.items()
        if sum(exponent) == degree
    }
    assert set(leading) == {(1, 1, 0), (1, 0, 1)}
    return np.array(
        [-leading[(1, 1, 0)], -leading[(1, 0, 1)]], dtype=float
    )


def test_exact_solution_freezes_orthogonal_exit_mode_and_preserves_balance():
    strengths = np.array([0.4, 0.2, 0.1])
    entrance = 0.31
    exits = np.array([0.22, 0.17, 0.29])
    time = 1.7

    initial_aligned, initial_orthogonal = shared_entrance_mode_coordinates(
        strengths, exits
    )
    evolved_entrance, evolved_exits = shared_entrance_quadratic_flow(
        strengths, entrance, exits, time
    )
    evolved_aligned, evolved_orthogonal = shared_entrance_mode_coordinates(
        strengths, evolved_exits
    )

    assert np.allclose(
        evolved_orthogonal,
        initial_orthogonal,
        rtol=1e-14,
        atol=1e-15,
    )
    initial_balance = entrance**2 - np.dot(exits, exits)
    evolved_balance = evolved_entrance**2 - np.dot(
        evolved_exits, evolved_exits
    )
    assert np.isclose(evolved_balance, initial_balance, rtol=1e-13, atol=1e-14)

    norm = np.linalg.norm(strengths)
    assert np.isclose(
        evolved_entrance,
        entrance * np.cosh(norm * time)
        + initial_aligned * np.sinh(norm * time),
        rtol=1e-14,
    )
    assert np.isclose(
        evolved_aligned,
        initial_aligned * np.cosh(norm * time)
        + entrance * np.sinh(norm * time),
        rtol=1e-14,
    )


def test_zero_exit_growth_is_exactly_coefficient_aligned():
    strengths = np.array([0.41, 0.23, 0.11])
    entrance = 0.02
    time = 3.2
    _, exits = shared_entrance_quadratic_flow(
        strengths,
        entrance,
        np.zeros_like(strengths),
        time,
    )

    # Every nonzero exit has the same y_j/a_j ratio at every positive time.
    ratios = exits / strengths
    assert np.allclose(ratios, ratios[0], rtol=1e-14, atol=1e-15)


def test_zero_exit_crossing_order_is_strength_order():
    strengths = np.array([0.09, 0.4, 0.2])
    times = shared_entrance_zero_exit_crossing_times(
        strengths,
        initial_entrance=1e-3,
        threshold=0.01,
    )
    assert np.array_equal(np.argsort(times), np.array([1, 2, 0]))

    entrance = 1e-3
    threshold = 0.01
    for route, time in enumerate(times):
        _, exits = shared_entrance_quadratic_flow(
            strengths,
            entrance,
            np.zeros_like(strengths),
            time,
        )
        assert np.isclose(exits[route], threshold, rtol=1e-13, atol=1e-14)


def test_finite_memory_shared_route_coefficients_predict_00_exit_first():
    coefficients = _shared_route_polynomial()
    strengths = _leading_shared_strengths(coefficients)

    # The suffix-00 computation is much stronger than suffix-01 in this frozen
    # witness, so the exact shared-route leading dynamics predict it crosses any
    # common threshold first from zero exits.
    assert strengths[0] > strengths[1]
    assert np.isclose(strengths[0] / strengths[1], 7.6383754942, rtol=1e-10)

    times = shared_entrance_zero_exit_crossing_times(
        strengths,
        initial_entrance=1e-3,
        threshold=0.01,
    )
    assert times[0] < times[1]
    assert np.isclose(times[0], 17.0095337251, rtol=1e-10)
    assert np.isclose(times[1], 28.4981558510, rtol=1e-10)
