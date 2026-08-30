import math

import numpy as np

from memory_frontier import UnifilarSource, delayed_repeat_source
from memory_frontier.construction_order import construction_order_sandwich
from memory_frontier.forward_equivalence import (
    finite_horizon_forward_support,
    is_source_horizon_dormant_rewire,
)
from memory_frontier.order_barrier import (
    binary_chain_readout,
    smooth_controller_finite_horizon_log_loss,
)


def _collapsed_transition(k: int, alphabet: int = 2) -> np.ndarray:
    transition = np.zeros((k, alphabet, k), dtype=float)
    transition[:, :, 0] = 1.0
    return transition


def _partial_scaffold_family(
    delay: int, missing_prefix: int
) -> tuple[np.ndarray, np.ndarray]:
    k = delay + 1
    base = _collapsed_transition(k)

    # All links after the missing prefix are prewired. Because the first link is
    # always missing, these rows remain unreachable at the base point.
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


def test_source_aware_dormancy_can_include_row_of_reachable_memory():
    # The source emits only symbol 0. Memory 0 is reachable, but its symbol-1 row
    # is never exercisable and may be rewired without affecting forward behavior.
    source = UnifilarSource(
        np.array([[1.0, 0.0]], dtype=float),
        np.array([[0, 0]], dtype=int),
    )
    base = _collapsed_transition(2)
    candidate = base.copy()
    candidate[0, 1, :] = 0.0
    candidate[0, 1, 1] = 1.0

    support = finite_horizon_forward_support(source, base, 6)
    assert support.active_transition_rows[0, 0]
    assert not support.active_transition_rows[0, 1]
    assert is_source_horizon_dormant_rewire(source, base, candidate, 6)

    readout = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=float)
    base_loss = smooth_controller_finite_horizon_log_loss(
        source, base, readout, 6
    )
    candidate_loss = smooth_controller_finite_horizon_log_loss(
        source, candidate, readout, 6
    )
    assert np.isclose(candidate_loss, base_loss, atol=1e-14)


def test_change_to_active_row_is_not_certified_dormant_and_changes_loss():
    source = UnifilarSource(
        np.array([[1.0, 0.0]], dtype=float),
        np.array([[0, 0]], dtype=int),
    )
    base = _collapsed_transition(2)
    candidate = base.copy()
    candidate[0, 0, :] = 0.0
    candidate[0, 0, 1] = 1.0
    readout = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=float)

    assert not is_source_horizon_dormant_rewire(source, base, candidate, 6)
    base_loss = smooth_controller_finite_horizon_log_loss(
        source, base, readout, 6
    )
    candidate_loss = smooth_controller_finite_horizon_log_loss(
        source, candidate, readout, 6
    )
    assert candidate_loss > base_loss


def test_forward_equivalent_dormant_scaffolds_realize_orders_one_through_four():
    delay = 4
    horizon = 12
    source = delayed_repeat_source(2, delay, 0.1)
    readout = binary_chain_readout(delay, 0.4)
    collapsed, _ = _partial_scaffold_family(delay, delay)
    collapsed_loss = smooth_controller_finite_horizon_log_loss(
        source, collapsed, readout, horizon
    )
    assert np.isclose(collapsed_loss, math.log(2.0), atol=1e-12)

    for missing_prefix in range(1, delay + 1):
        base, directions = _partial_scaffold_family(delay, missing_prefix)

        # Rewiring only unreachable rows preserves the entire current forward
        # computation, not merely the scalar loss of this particular decoder.
        assert is_source_horizon_dormant_rewire(
            source, collapsed, base, horizon
        )
        base_loss = smooth_controller_finite_horizon_log_loss(
            source, base, readout, horizon
        )
        assert np.isclose(base_loss, collapsed_loss, atol=1e-12)

        support_order, operator_order, loss_order = construction_order_sandwich(
            source,
            base,
            directions,
            readout,
            horizon,
            max_total_degree=missing_prefix,
            coefficient_atol=1e-11,
        )
        assert (support_order, operator_order, loss_order) == (
            missing_prefix,
            missing_prefix,
            missing_prefix,
        )
