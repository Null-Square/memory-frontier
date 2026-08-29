import numpy as np

from memory_frontier import (
    alias_entropy,
    bayes_log_loss,
    best_deterministic_controller,
    best_recursive_quotient,
    best_static_partition,
    four_state_aliasing_witness,
    source_stationary_distribution,
)


def test_witness_stationary_distribution_and_bayes_loss():
    source = four_state_aliasing_witness()
    pi = source_stationary_distribution(source)
    expected_pi = np.array([0.4490291262, 0.2184466019, 0.2427184466, 0.0898058252])
    np.testing.assert_allclose(pi, expected_pi, atol=1e-9)
    assert abs(bayes_log_loss(source) - 0.4662344730440136) < 1e-9


def test_witness_three_frontiers_are_strict():
    source = four_state_aliasing_witness()
    static = best_static_partition(source, 2)
    online = best_deterministic_controller(source, 2)
    quotient = best_recursive_quotient(source, 2)

    assert abs(static.loss - 0.484069767312062) < 1e-9
    assert abs(online.loss - 0.6297909185) < 1e-9
    assert abs(quotient.loss - 0.6886553518041918) < 1e-9
    assert static.loss < online.loss < quotient.loss


def test_witness_optimal_controller_remembers_last_symbol():
    source = four_state_aliasing_witness()
    online = best_deterministic_controller(source, 2)
    expected = np.array([[0, 1], [0, 1]])
    np.testing.assert_array_equal(online.transition_table, expected)


def test_witness_optimal_controller_uses_history_dependent_aliasing():
    source = four_state_aliasing_witness()
    online = best_deterministic_controller(source, 2)
    h = alias_entropy(online.occupancy)
    assert h > 0.0
    # C (state index 2) has positive stationary mass in both memory states.
    assert online.occupancy[2, 0] > 0
    assert online.occupancy[2, 1] > 0
