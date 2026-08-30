import numpy as np

from memory_frontier import (
    delayed_repeat_source,
    dormant_chain_controller,
    edit_alignment_operators,
    hard_value_gradient,
)


def _canonical_logits(table: np.ndarray, margin: float = 0.7) -> np.ndarray:
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def test_linearized_hard_edit_gain_matches_finite_difference():
    source = delayed_repeat_source(2, 3, 0.1)
    table = dormant_chain_controller(3, 2)
    operators = edit_alignment_operators(
        source, _canonical_logits(table), np.zeros(2), 20, temperature=0.8
    )
    direction = np.zeros((4, 2), dtype=float)
    direction[3] = np.array([0.3, -0.3])

    edge_index = next(
        i
        for i, edge in enumerate(operators.edges)
        if edge.memory_state == 0 and edge.symbol == 0 and edge.new_target == 1
    )
    predicted = operators.directional_hard_gain(direction)[edge_index]
    edge = operators.edges[edge_index]
    other = table.copy()
    other[edge.memory_state, edge.symbol] = edge.new_target

    epsilon = 1e-6
    gains = []
    for sign in (-1.0, 1.0):
        readout = _softmax_rows(sign * epsilon * direction)
        base_loss = hard_value_gradient(source, table, readout, 20).loss
        other_loss = hard_value_gradient(source, other, readout, 20).loss
        gains.append(base_loss - other_loss)
    finite_difference = (gains[1] - gains[0]) / (2.0 * epsilon)
    assert abs(predicted - finite_difference) < 1e-9


def test_alignment_reports_pressure_and_gain_for_same_edges():
    source = delayed_repeat_source(2, 3, 0.1)
    table = dormant_chain_controller(3, 2)
    operators = edit_alignment_operators(
        source, _canonical_logits(table), np.zeros(2), 20, temperature=0.8
    )
    direction = np.zeros((4, 2), dtype=float)
    direction[3] = np.array([1.0, -1.0])
    pressure = operators.directional_pressure(direction)
    gain = operators.directional_hard_gain(direction)
    assert pressure.shape == gain.shape == (len(operators.edges),)
    assert -1.0 <= operators.frobenius_alignment() <= 1.0
    fidelity = operators.directional_sign_fidelity(direction)
    assert 0.0 <= fidelity <= 1.0
