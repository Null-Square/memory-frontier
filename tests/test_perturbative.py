import math

import numpy as np

from memory_frontier import UnifilarSource, delayed_repeat_source
from memory_frontier.order_barrier import (
    binary_chain_readout,
    binary_soft_chain_transition,
    smooth_controller_finite_horizon_log_loss,
)
from memory_frontier.perturbative import (
    affine_controller_loss_coefficients,
    leading_perturbative_order,
    minimum_decoder_construction_cost,
)


def _evaluate(coefficients: np.ndarray, epsilon: float) -> float:
    powers = epsilon ** np.arange(len(coefficients), dtype=float)
    return float(np.dot(coefficients, powers))


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


def test_affine_loss_polynomial_reconstructs_direct_evaluation():
    source = delayed_repeat_source(2, 2, 0.1)
    horizon = 9
    base = binary_soft_chain_transition(2, 0.0)
    direction = binary_soft_chain_transition(2, 1.0) - base
    readout = binary_chain_readout(2, 0.4)
    coefficients = affine_controller_loss_coefficients(
        source, base, direction, readout, horizon
    )

    for epsilon in (0.0, 0.03, 0.2, 0.7):
        direct = smooth_controller_finite_horizon_log_loss(
            source, base + epsilon * direction, readout, horizon
        )
        assert np.isclose(_evaluate(coefficients, epsilon), direct, atol=1e-12)


def test_chain_construction_distance_matches_generic_order():
    for delay in (2, 3, 4, 5):
        source = delayed_repeat_source(2, delay, 0.1)
        horizon = delay + 8
        base = binary_soft_chain_transition(delay, 0.0)
        direction = binary_soft_chain_transition(delay, 1.0) - base
        readout = binary_chain_readout(delay, 0.4)
        coefficients = affine_controller_loss_coefficients(
            source, base, direction, readout, horizon
        )
        distance = minimum_decoder_construction_cost(
            source, base, direction, readout, horizon
        )
        assert distance == delay
        assert leading_perturbative_order(coefficients, atol=1e-11) == delay


def test_prewired_dormant_scaffold_reduces_distance_and_order_to_one():
    delay = 4
    source = delayed_repeat_source(2, delay, 0.1)
    horizon = 14
    base_links = np.ones(delay, dtype=float)
    base_links[0] = 0.0
    base = binary_soft_chain_transition(delay, base_links)
    full = binary_soft_chain_transition(delay, np.ones(delay))
    direction = full - base
    readout = binary_chain_readout(delay, 0.4)
    coefficients = affine_controller_loss_coefficients(
        source, base, direction, readout, horizon
    )
    assert minimum_decoder_construction_cost(
        source, base, direction, readout, horizon
    ) == 1
    assert leading_perturbative_order(coefficients, atol=1e-11) == 1


def test_exact_cancellation_can_raise_order_above_structural_distance():
    # States are 00, 01, 10, 11. The source is second-order and symmetric,
    # with P(next=0 | most recent emitted token=0)=5/6.
    emissions = np.array(
        [[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]], dtype=float
    )
    transitions = np.array(
        [[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int
    )
    source = UnifilarSource(emissions, transitions)

    k = 3
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    direction = np.zeros_like(base)
    direction[0, 0, 0] = -1.0
    direction[0, 0, 1] = 1.0
    direction[1, 0, 0] = -1.0
    direction[1, 0, 2] = 1.0

    readout = np.full((k, 2), 0.5, dtype=float)
    q_neutral = _neutral_decoder_root(5.0 / 6.0)
    readout[1] = np.array([q_neutral, 1.0 - q_neutral])
    readout[2] = np.array([0.9, 0.1])

    coefficients = affine_controller_loss_coefficients(
        source, base, direction, readout, 10
    )
    distance = minimum_decoder_construction_cost(
        source, base, direction, readout, 10
    )

    assert distance == 1
    assert abs(coefficients[1]) < 1e-12
    assert coefficients[2] < -0.03
    assert leading_perturbative_order(coefficients, atol=1e-11) == 2


def _independent_chain_directions(delay: int) -> tuple[np.ndarray, np.ndarray]:
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


def test_independent_missing_links_first_appear_as_full_mixed_derivative():
    from memory_frontier.order_barrier import (
        binary_delay_chain_leading_gain_coefficient,
    )
    from memory_frontier.perturbative import (
        leading_multivariate_total_degree,
        multivariate_controller_loss_coefficients,
    )

    for delay in (2, 3, 4, 5):
        horizon = delay + 8
        source = delayed_repeat_source(2, delay, 0.1)
        base, directions = _independent_chain_directions(delay)
        readout = binary_chain_readout(delay, 0.4)
        coefficients = multivariate_controller_loss_coefficients(
            source, base, directions, readout, horizon
        )

        assert leading_multivariate_total_degree(coefficients, atol=1e-11) == delay
        leading_key = (1,) * delay
        expected_gain = binary_delay_chain_leading_gain_coefficient(
            delay, 0.1, horizon, 0.4
        )
        assert np.isclose(coefficients[leading_key], -expected_gain, atol=1e-12)
        for exponent, coefficient in coefficients.items():
            if 0 < sum(exponent) < delay:
                assert abs(coefficient) < 1e-12


def test_tying_parameters_can_hide_nonzero_independent_gradients():
    from memory_frontier.perturbative import (
        leading_multivariate_total_degree,
        multivariate_controller_loss_coefficients,
    )

    source = delayed_repeat_source(2, 1, 0.2)
    k = 3
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    direction_zero = np.zeros_like(base)
    direction_zero[0, 0, 0] = -1.0
    direction_zero[0, 0, 1] = 1.0
    direction_one = np.zeros_like(base)
    direction_one[0, 1, 0] = -1.0
    direction_one[0, 1, 2] = 1.0
    directions = np.stack([direction_zero, direction_one])

    readout = np.full((k, 2), 0.5, dtype=float)
    readout[1] = np.array([0.8, 0.2])
    readout[2] = np.array([0.6289589607495853, 1.0 - 0.6289589607495853])

    multi = multivariate_controller_loss_coefficients(
        source, base, directions, readout, 8
    )
    assert leading_multivariate_total_degree(multi, atol=1e-11) == 1
    assert multi[(1, 0)] < -0.08
    assert multi[(0, 1)] > 0.08
    assert np.isclose(multi[(1, 0)] + multi[(0, 1)], 0.0, atol=1e-12)

    tied = affine_controller_loss_coefficients(
        source, base, direction_zero + direction_one, readout, 8
    )
    assert leading_perturbative_order(tied, atol=1e-11) is None


def _partial_scaffold_family(delay: int, missing_prefix: int) -> tuple[np.ndarray, np.ndarray]:
    """Same collapsed predictor, with only a prefix of chain links left missing."""
    k = delay + 1
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0

    # Links after the missing prefix are prewired but behaviorally unreachable,
    # because the first link is always among the missing links.
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


def test_same_forward_function_realizes_every_optimization_order_one_through_delay():
    from memory_frontier.order_barrier import (
        binary_delay_chain_leading_gain_coefficient,
    )
    from memory_frontier.perturbative import (
        leading_multivariate_total_degree,
        multivariate_controller_loss_coefficients,
    )

    delay = 5
    horizon = 13
    source = delayed_repeat_source(2, delay, 0.1)
    readout = binary_chain_readout(delay, 0.4)
    expected_gain = binary_delay_chain_leading_gain_coefficient(
        delay, 0.1, horizon, 0.4
    )

    for missing_prefix in range(1, delay + 1):
        base, directions = _partial_scaffold_family(delay, missing_prefix)

        # All parameter points execute the same collapsed predictor at epsilon=0:
        # memory never leaves state 0, so the unreachable prewiring is invisible.
        base_loss = smooth_controller_finite_horizon_log_loss(
            source, base, readout, horizon
        )
        assert np.isclose(base_loss, math.log(2.0), atol=1e-12)

        coefficients = multivariate_controller_loss_coefficients(
            source,
            base,
            directions,
            readout,
            horizon,
            max_total_degree=missing_prefix,
        )
        assert (
            leading_multivariate_total_degree(coefficients, atol=1e-11)
            == missing_prefix
        )
        leading_key = (1,) * missing_prefix
        assert np.isclose(
            coefficients[leading_key], -expected_gain, atol=1e-12
        )
        for exponent, coefficient in coefficients.items():
            if 0 < sum(exponent) < missing_prefix:
                assert abs(coefficient) < 1e-12
