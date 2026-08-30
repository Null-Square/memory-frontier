import math

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients


def _second_order_pattern_source():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    emissions = np.column_stack([1.0 - p1, p1])
    transitions = np.array(
        [[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int
    )
    return UnifilarSource(emissions, transitions)


def _pattern_controller():
    k = 3
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    first = np.zeros_like(base)
    first[0, 0, 0] = -1.0
    first[0, 0, 1] = 1.0

    second = np.zeros_like(base)
    second[1, 1, 0] = -1.0
    second[1, 1, 2] = 1.0
    return base, first, second


def _readout_from_gap(gap: float):
    # Final-state logits are (-a,+a); other decoder rows stay uniform.
    a = float(gap)
    q1 = 1.0 / (1.0 + math.exp(-2.0 * a))
    readout = np.full((3, 2), 0.5, dtype=float)
    readout[2] = np.array([1.0 - q1, q1])
    return readout


def _decoder_excess_cost(gap: float):
    a = float(gap)
    return math.log(math.cosh(a)) - (3.0 / 5.0) * a


def _closed_form_route_coefficient(gap: float):
    # Reaching the suffix-01 decoder has exact finite-horizon occupancy
    # coefficient 5/42. Conditional on suffix 01, P(next=1)=4/5, so the
    # decoder excess cross-entropy relative to uniform is
    # log(cosh(a))-(3/5)a for logits (-a,+a).
    return (5.0 / 42.0) * _decoder_excess_cost(gap)


def test_joint_decoder_transition_coefficient_has_exact_closed_form():
    source = _second_order_pattern_source()
    base, first, second = _pattern_controller()
    horizon = 12

    for gap in (0.0, 0.1, 0.4, 0.8):
        readout = _readout_from_gap(gap)
        expected = _closed_form_route_coefficient(gap)

        unscaffolded = multivariate_controller_loss_coefficients(
            source,
            base,
            np.asarray([first, second]),
            readout,
            horizon,
        )
        assert np.isclose(
            unscaffolded.get((1, 1), 0.0), expected, atol=1e-12
        )

        prewired = multivariate_controller_loss_coefficients(
            source,
            base + second,
            np.asarray([first]),
            readout,
            horizon,
        )
        assert np.isclose(
            prewired.get((1,), 0.0), expected, atol=1e-12
        )


def test_full_transition_polynomial_factorizes_through_decoder_excess_cost():
    source = _second_order_pattern_source()
    base, first, second = _pattern_controller()
    horizon = 12
    reference_gap = 0.4
    reference_cost = _decoder_excess_cost(reference_gap)

    for prewired in (False, True):
        transition_base = base + second if prewired else base
        directions = np.asarray([first]) if prewired else np.asarray([first, second])
        reference = multivariate_controller_loss_coefficients(
            source,
            transition_base,
            directions,
            _readout_from_gap(reference_gap),
            horizon,
        )

        occupancy_polynomial = {
            exponent: coefficient / reference_cost
            for exponent, coefficient in reference.items()
            if sum(exponent) > 0 and abs(coefficient) > 1e-12
        }

        for gap in (0.1, 0.7):
            current = multivariate_controller_loss_coefficients(
                source,
                transition_base,
                directions,
                _readout_from_gap(gap),
                horizon,
            )
            cost = _decoder_excess_cost(gap)
            for exponent, occupancy_coefficient in occupancy_polynomial.items():
                assert np.isclose(
                    current.get(exponent, 0.0),
                    occupancy_coefficient * cost,
                    atol=1e-12,
                )
            for exponent, coefficient in current.items():
                if sum(exponent) == 0:
                    continue
                if exponent not in occupancy_polynomial:
                    assert abs(coefficient) < 1e-12


def test_exact_decoder_symmetry_adds_one_joint_construction_factor():
    source = _second_order_pattern_source()
    base, first, second = _pattern_controller()
    horizon = 12
    symmetric_readout = _readout_from_gap(0.0)

    unscaffolded = multivariate_controller_loss_coefficients(
        source,
        base,
        np.asarray([first, second]),
        symmetric_readout,
        horizon,
    )
    prewired = multivariate_controller_loss_coefficients(
        source,
        base + second,
        np.asarray([first]),
        symmetric_readout,
        horizon,
    )
    assert all(
        abs(coefficient) < 1e-12
        for exponent, coefficient in unscaffolded.items()
        if sum(exponent) > 0
    )
    assert all(
        abs(coefficient) < 1e-12
        for exponent, coefficient in prewired.items()
        if sum(exponent) > 0
    )

    # From the exact factorization
    # (5/42)[log cosh(a) - (3/5)a], the decoder-linear coefficient is -1/14.
    decoder_linear_coefficient = (5.0 / 42.0) * (-(3.0 / 5.0))
    assert np.isclose(decoder_linear_coefficient, -1.0 / 14.0, atol=1e-15)
