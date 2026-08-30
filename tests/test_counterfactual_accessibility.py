import numpy as np

from memory_frontier import (
    counterfactual_accessibility_operator,
    delayed_repeat_source,
    dormant_chain_controller,
    gradient_accessibility_operator,
)


def _canonical_logits(table: np.ndarray, margin: float) -> np.ndarray:
    table = np.asarray(table, dtype=int)
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def test_counterfactual_operator_factors_exact_gao():
    source = delayed_repeat_source(2, 3, 0.1)
    table = np.array(
        [[0, 0], [2, 2], [3, 3], [0, 0]], dtype=int
    )
    logits = _canonical_logits(table, 0.7)
    intrinsic = counterfactual_accessibility_operator(
        source, table, np.zeros(2), 20
    )
    gao = gradient_accessibility_operator(
        source, logits, np.zeros(2), 20, temperature=0.8
    )
    np.testing.assert_allclose(
        intrinsic.push_through_transition_softmax(logits, 0.8),
        gao.matrix,
        atol=1e-12,
    )
    assert intrinsic.numerical_rank() == gao.numerical_rank()


def test_accessibility_rank_is_invariant_to_finite_surrogate_geometry():
    source = delayed_repeat_source(2, 3, 0.1)
    table = dormant_chain_controller(3, 2)
    intrinsic = counterfactual_accessibility_operator(
        source, table, np.zeros(2), 20
    )
    assert intrinsic.numerical_rank() == 1
    for margin, temperature in ((0.05, 0.2), (0.7, 0.8), (2.0, 1.0), (5.0, 3.0)):
        gao = gradient_accessibility_operator(
            source,
            _canonical_logits(table, margin),
            np.zeros(2),
            20,
            temperature=temperature,
        )
        assert gao.numerical_rank() == intrinsic.numerical_rank()


def test_intrinsic_rank_bound_and_forward_equivalent_witness():
    source = delayed_repeat_source(2, 3, 0.1)
    blind = np.zeros((4, 2), dtype=int)
    scaffold = dormant_chain_controller(3, 2)
    blind_operator = counterfactual_accessibility_operator(
        source, blind, np.zeros(2), 20
    )
    scaffold_operator = counterfactual_accessibility_operator(
        source, scaffold, np.zeros(2), 20
    )
    assert blind_operator.theoretical_rank_bound == 3
    assert blind_operator.numerical_rank() == 0
    assert scaffold_operator.numerical_rank() == 1
    assert scaffold_operator.numerical_rank() <= scaffold_operator.theoretical_rank_bound
