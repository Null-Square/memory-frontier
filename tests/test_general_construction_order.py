import math

import numpy as np

from memory_frontier import UnifilarSource, delayed_repeat_source
from memory_frontier.construction_order import (
    construction_order_operator,
    construction_order_sandwich,
    minimum_readout_class_construction_cost,
)
from memory_frontier.order_barrier import binary_chain_readout
from memory_frontier.perturbative import multivariate_controller_loss_coefficients


def _independent_chain_directions(delay: int):
    k = delay + 1
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = []

    first = np.zeros_like(base)
    first[0, 0, 0] = -1.0
    first[0, 0, 1] = 1.0
    directions.append(first)
    for state in range(1, delay):
        direction = np.zeros_like(base)
        for symbol in range(2):
            direction[state, symbol, 0] = -1.0
            direction[state, symbol, state + 1] = 1.0
        directions.append(direction)
    return base, np.asarray(directions)


def _partial_scaffold(delay: int, missing_prefix: int):
    k = delay + 1
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    for link in range(missing_prefix + 1, delay + 1):
        state = link - 1
        base[state, :, :] = 0.0
        base[state, :, state + 1] = 1.0

    directions = []
    first = np.zeros_like(base)
    first[0, 0, 0] = -1.0
    first[0, 0, 1] = 1.0
    directions.append(first)
    for link in range(2, missing_prefix + 1):
        state = link - 1
        direction = np.zeros_like(base)
        for symbol in range(2):
            direction[state, symbol, 0] = -1.0
            direction[state, symbol, state + 1] = 1.0
        directions.append(direction)
    return base, np.asarray(directions)


def _neutral_decoder_root(p_zero: float) -> float:
    def gain(q_zero: float) -> float:
        return (
            math.log(2.0)
            + p_zero * math.log(q_zero)
            + (1.0 - p_zero) * math.log(1.0 - q_zero)
        )

    lo = p_zero
    hi = 1.0 - 1e-14
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if gain(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_construction_operator_reconstructs_exact_multivariate_loss_polynomial():
    source = delayed_repeat_source(2, 3, 0.13)
    base, directions = _independent_chain_directions(3)
    readout = binary_chain_readout(3, 0.37)
    horizon = 10

    operator = construction_order_operator(
        source, base, directions, readout, horizon
    )
    reconstructed = operator.reconstructed_loss_coefficients()
    direct = multivariate_controller_loss_coefficients(
        source, base, directions, readout, horizon
    )

    assert reconstructed.keys() == direct.keys()
    for exponent, coefficient in direct.items():
        assert np.isclose(reconstructed[exponent], coefficient, atol=2e-13)


def test_generic_chains_saturate_support_operator_loss_order_sandwich():
    for delay in (1, 2, 3, 4, 5):
        source = delayed_repeat_source(2, delay, 0.1)
        base, directions = _independent_chain_directions(delay)
        readout = binary_chain_readout(delay, 0.4)
        horizon = delay + 8

        support, operator_order, loss_order = construction_order_sandwich(
            source,
            base,
            directions,
            readout,
            horizon,
            max_total_degree=delay,
        )
        assert support == delay
        assert operator_order == delay
        assert loss_order == delay


def test_dormant_scaffolding_reduces_all_three_orders_without_forward_effect():
    delay = 5
    source = delayed_repeat_source(2, delay, 0.1)
    readout = binary_chain_readout(delay, 0.4)
    horizon = 13

    observed = []
    for missing_prefix in range(1, delay + 1):
        base, directions = _partial_scaffold(delay, missing_prefix)
        observed.append(
            construction_order_sandwich(
                source,
                base,
                directions,
                readout,
                horizon,
                max_total_degree=missing_prefix,
            )
        )
    assert observed == [(d, d, d) for d in range(1, delay + 1)]


def test_decoder_value_cancellation_strictly_raises_scalar_loss_order():
    emissions = np.array(
        [[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]], dtype=float
    )
    transitions = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int)
    source = UnifilarSource(emissions, transitions)

    k = 3
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    direction = np.zeros_like(base)
    direction[0, 0, 0] = -1.0
    direction[0, 0, 1] = 1.0
    direction[1, 0, 0] = -1.0
    direction[1, 0, 2] = 1.0
    directions = direction[None, ...]

    readout = np.full((k, 2), 0.5, dtype=float)
    q_neutral = _neutral_decoder_root(5.0 / 6.0)
    readout[1] = np.array([q_neutral, 1.0 - q_neutral])
    readout[2] = np.array([0.9, 0.1])

    operator = construction_order_operator(source, base, directions, readout, 10)
    assert minimum_readout_class_construction_cost(
        source, base, directions, readout, 10
    ) == 1
    assert operator.operator_order(atol=1e-11) == 1
    assert operator.decoder_cancellation_residual(1) < 1e-12
    assert operator.decoder_cancellation_residual(2) > 0.03
    assert operator.loss_order(atol=1e-11) == 2


def test_equal_readouts_quotient_all_transition_only_construction_effects():
    source = delayed_repeat_source(2, 3, 0.1)
    base, directions = _independent_chain_directions(3)
    readout = np.full((4, 2), 0.5, dtype=float)

    operator = construction_order_operator(source, base, directions, readout, 10)
    assert len(operator.readout_classes) == 1
    assert minimum_readout_class_construction_cost(
        source, base, directions, readout, 10
    ) is None
    assert operator.operator_order(atol=1e-11) is None
    assert operator.loss_order(atol=1e-11) is None


def test_shared_route_fixture_has_structural_order_two_before_route_competition():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    source = UnifilarSource(
        np.column_stack([1.0 - p1, p1]),
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int),
    )
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

    assert construction_order_sandwich(
        source, base, directions, readout, 12, max_total_degree=2
    ) == (2, 2, 2)
