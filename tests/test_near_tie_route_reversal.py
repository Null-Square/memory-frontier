import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.near_tie import (
    near_tie_margin_exponent,
    rescaled_gradient_flow_loss_coefficients,
)
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient


TIE_DECODER_00 = 0.46869740167476925


def _second_order_pattern_source():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    return UnifilarSource(
        np.column_stack([1.0 - p1, p1]),
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int),
    )


def _shared_route_family(q00, *, max_total_degree=None):
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
    readout[2] = np.array([1.0 - q00, q00])
    readout[3] = np.array([0.2, 0.8])
    return multivariate_controller_loss_coefficients(
        source,
        base,
        directions,
        readout,
        horizon=12,
        max_total_degree=max_total_degree,
    )


def _exit_velocity_gap(coefficients, delta):
    point = np.array([delta, 0.0, 0.0], dtype=float)
    velocity = -evaluate_multivariate_polynomial_gradient(coefficients, point)
    return float(velocity[1] - velocity[2])


def test_local_rescaling_exposes_degree_gap_exactly():
    coefficients = {
        (1, 1): -0.7,
        (2, 1): 0.3,
        (4, 0): -0.2,
        (0, 0): 5.0,
    }
    rescaled = rescaled_gradient_flow_loss_coefficients(coefficients, 0.1)

    assert rescaled[(1, 1)] == -0.7
    assert np.isclose(rescaled[(2, 1)], 0.03)
    assert np.isclose(rescaled[(4, 0)], -0.002)
    assert (0, 0) not in rescaled
    assert near_tie_margin_exponent(2, 3) == 1


def test_tuned_decoder_is_a_quadratic_shared_route_tie():
    coefficients = _shared_route_family(TIE_DECODER_00, max_total_degree=3)
    strength_00 = -coefficients[(1, 1, 0)]
    strength_01 = -coefficients[(1, 0, 1)]

    assert np.isclose(strength_00, strength_01, rtol=0.0, atol=5e-14)
    # At the tie, the route-specific cubic x^2*y coefficients are unequal.
    assert np.isclose(
        coefficients[(2, 1, 0)] / strength_00,
        0.81,
        atol=2e-13,
    )
    assert np.isclose(
        coefficients[(2, 0, 1)] / strength_01,
        0.36,
        atol=2e-13,
    )


def test_cubic_terms_reverse_the_exact_initial_gradient_near_the_tie():
    delta = 0.01
    q00 = TIE_DECODER_00 - 0.0001
    quadratic = _shared_route_family(q00, max_total_degree=2)
    cubic = _shared_route_family(q00, max_total_degree=3)
    full = _shared_route_family(q00)

    strength_00 = -quadratic[(1, 1, 0)]
    strength_01 = -quadratic[(1, 0, 1)]
    assert strength_00 > strength_01
    assert _exit_velocity_gap(quadratic, delta) > 0.0

    # The first higher-order correction alone reverses the gradient direction,
    # and the complete horizon-12 polynomial keeps the reversal.
    assert _exit_velocity_gap(cubic, delta) < 0.0
    assert _exit_velocity_gap(full, delta) < 0.0


def test_reversal_window_scales_linearly_with_initialization():
    # q00=q_star-kappa*delta. The leading quadratic route always favors 00.
    # kappa=0.014 lies below the asymptotic reversal boundary (~0.0146453),
    # while kappa=0.0155 lies above it.
    for delta in (0.02, 0.01, 0.005, 0.0025):
        losing_kappa = 0.014
        losing = _shared_route_family(
            TIE_DECODER_00 - losing_kappa * delta
        )
        assert -losing[(1, 1, 0)] > -losing[(1, 0, 1)]
        assert _exit_velocity_gap(losing, delta) < 0.0

        winning_kappa = 0.0155
        winning = _shared_route_family(
            TIE_DECODER_00 - winning_kappa * delta
        )
        assert -winning[(1, 1, 0)] > -winning[(1, 0, 1)]
        assert _exit_velocity_gap(winning, delta) > 0.0
