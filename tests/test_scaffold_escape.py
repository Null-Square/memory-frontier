import numpy as np

from memory_frontier import (
    binary_dormant_chain_readout_logits,
    delayed_repeat_source,
    dormant_chain_controller,
    exact_ste_gradient,
)


def _canonical_logits(table: np.ndarray, margin: float) -> np.ndarray:
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def _train(source, table, readout_logits, steps=20):
    logits = _canonical_logits(table, 0.7)
    initial = exact_ste_gradient(
        source, logits, readout_logits, 20, temperature=0.8
    )
    for _ in range(steps):
        gradient = exact_ste_gradient(
            source, logits, readout_logits, 20, temperature=0.8
        )
        logits -= 5.0 * gradient.transition_logit_gradient
    final = exact_ste_gradient(
        source, logits, readout_logits, 20, temperature=0.8
    )
    return initial, final


def test_forward_identical_dormant_scaffold_changes_exact_sgd_learnability():
    delay = 3
    source = delayed_repeat_source(2, delay, 0.1)
    blind = np.zeros((delay + 1, 2), dtype=int)
    scaffold = dormant_chain_controller(delay, 2)
    readout_logits = binary_dormant_chain_readout_logits(delay, 0.2)

    blind_initial, blind_final = _train(source, blind, readout_logits)
    scaffold_initial, scaffold_final = _train(
        source, scaffold, readout_logits
    )

    # From reset, both initial hard controllers stay forever in memory 0 and
    # therefore make exactly the same forward predictions.
    assert abs(blind_initial.loss - scaffold_initial.loss) < 1e-15
    assert abs(blind_initial.loss - np.log(2.0)) < 1e-15

    # Inert counterfactual structure reinforces the collapsed forward function.
    np.testing.assert_array_equal(blind_final.transition_table, blind)
    assert abs(blind_final.loss - blind_initial.loss) < 1e-15

    # The delay-matched unreachable chain changes only the backward field. It
    # activates a reachable transition and lowers the exact fixed-readout loss.
    assert scaffold_final.transition_table[0, 0] == 1
    assert scaffold_final.transition_table[0, 1] == 0
    assert scaffold_final.loss < scaffold_initial.loss - 0.02
