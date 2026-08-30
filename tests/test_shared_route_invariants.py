import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.support_invariants import (
    quadratic_invariant_basis,
    quadratic_invariant_derivative,
)


def _second_order_pattern_source():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    emissions = np.column_stack([1.0 - p1, p1])
    transitions = np.array(
        [[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int
    )
    return UnifilarSource(emissions, transitions)


def _shared_entrance_controller():
    # One learned entrance x reaches memory state 1 after a 0. Two learned exits
    # then recognize suffixes 00 and 01, respectively.
    k = 4
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    entrance = np.zeros_like(base)
    entrance[0, 0, 0] = -1.0
    entrance[0, 0, 1] = 1.0

    exit_00 = np.zeros_like(base)
    exit_00[1, 0, 0] = -1.0
    exit_00[1, 0, 2] = 1.0

    exit_01 = np.zeros_like(base)
    exit_01[1, 1, 0] = -1.0
    exit_01[1, 1, 3] = 1.0

    readout = np.full((k, 2), 0.5, dtype=float)
    readout[2] = np.array([0.9, 0.1])  # exact predictor after suffix 00
    readout[3] = np.array([0.2, 0.8])  # exact predictor after suffix 01
    return base, np.asarray([entrance, exit_00, exit_01]), readout


def _leading_nonconstant_polynomial(coefficients, atol=1e-12):
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


def test_exact_finite_memory_shared_routes_have_predicted_nullspace_invariant():
    source = _second_order_pattern_source()
    base, directions, readout = _shared_entrance_controller()
    coefficients = multivariate_controller_loss_coefficients(
        source,
        base,
        directions,
        readout,
        horizon=12,
    )
    leading = _leading_nonconstant_polynomial(coefficients)

    # The first useful computations are exactly x*y (recognize 00) and x*z
    # (recognize 01). Both are beneficial.
    assert set(leading) == {(1, 1, 0), (1, 0, 1)}
    assert all(coefficient < 0.0 for coefficient in leading.values())

    basis = quadratic_invariant_basis(leading)
    assert basis.shape == (1, 3)
    expected = np.asarray([1.0, -1.0, -1.0], dtype=float)
    expected /= np.linalg.norm(expected)
    assert np.isclose(abs(np.dot(basis[0], expected)), 1.0, atol=1e-13)

    weights = np.asarray([1.0, -1.0, -1.0])
    for point in (
        (0.2, 0.3, 0.4),
        (0.5, 0.7, 0.6),
        (0.9, 0.4, 0.8),
    ):
        assert np.isclose(
            quadratic_invariant_derivative(leading, point, weights),
            0.0,
            atol=1e-14,
        )
