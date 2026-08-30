import math

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.order_barrier import smooth_controller_finite_horizon_log_loss
from memory_frontier.perturbative import (
    leading_multivariate_total_degree,
    multivariate_controller_loss_coefficients,
)


def _second_order_pattern_source():
    # Source states are the last two emitted bits: 00, 01, 10, 11.
    # These probabilities are deliberately generic rather than a delayed-copy
    # or delayed-repeat construction.
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    emissions = np.column_stack([1.0 - p1, p1])
    transitions = np.array(
        [[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int
    )
    return UnifilarSource(emissions, transitions)


def _pattern_01_controller_family():
    k = 3
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    enter_after_zero = np.zeros_like(base)
    enter_after_zero[0, 0, 0] = -1.0
    enter_after_zero[0, 0, 1] = 1.0

    complete_after_one = np.zeros_like(base)
    complete_after_one[1, 1, 0] = -1.0
    complete_after_one[1, 1, 2] = 1.0

    readout = np.full((k, 2), 0.5, dtype=float)
    # Once suffix 01 has been recognized, state 01 emits token 1 with P=.8.
    readout[2] = np.array([0.2, 0.8])
    return base, enter_after_zero, complete_after_one, readout


def test_non_delayed_pattern_memory_has_order_two_without_scaffold():
    source = _second_order_pattern_source()
    base, first, second, readout = _pattern_01_controller_family()
    horizon = 12

    base_loss = smooth_controller_finite_horizon_log_loss(
        source, base, readout, horizon
    )
    assert np.isclose(base_loss, math.log(2.0), atol=1e-12)

    coefficients = multivariate_controller_loss_coefficients(
        source,
        base,
        np.asarray([first, second]),
        readout,
        horizon,
    )
    assert leading_multivariate_total_degree(coefficients, atol=1e-11) == 2
    assert abs(coefficients.get((1, 0), 0.0)) < 1e-12
    assert abs(coefficients.get((0, 1), 0.0)) < 1e-12
    assert np.isclose(
        coefficients[(1, 1)], -0.022945804407352086, atol=1e-12
    )


def test_dormant_prewiring_reduces_same_non_delayed_computation_to_order_one():
    source = _second_order_pattern_source()
    base, first, second, readout = _pattern_01_controller_family()
    horizon = 12

    prewired_base = base + second
    prewired_loss = smooth_controller_finite_horizon_log_loss(
        source, prewired_base, readout, horizon
    )
    assert np.isclose(prewired_loss, math.log(2.0), atol=1e-12)

    coefficients = multivariate_controller_loss_coefficients(
        source,
        prewired_base,
        np.asarray([first]),
        readout,
        horizon,
    )
    assert leading_multivariate_total_degree(coefficients, atol=1e-11) == 1
    assert np.isclose(
        coefficients[(1,)], -0.022945804407352086, atol=1e-12
    )

    # Activating the entrance link yields the same fully wired computation in
    # both parameterizations; only the behaviorally unreachable base topology
    # differs at the collapsed point.
    full_from_scratch = base + first + second
    full_from_scaffold = prewired_base + first
    assert np.allclose(full_from_scratch, full_from_scaffold, atol=1e-12)
    full_loss = smooth_controller_finite_horizon_log_loss(
        source, full_from_scratch, readout, horizon
    )
    assert full_loss < math.log(2.0) - 0.01
