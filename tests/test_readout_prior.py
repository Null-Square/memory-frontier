import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier import finite_horizon_hard_landscape, four_state_aliasing_witness
from memory_frontier.readout_prior import (
    exact_distribution_gradient_snapshot,
    source_marginal_distribution,
    source_marginal_readout_logits,
    train_exact_distribution_ste_with_readout_prior,
)
from memory_frontier.surrogate import canonical_logits, canonical_ste_field
from memory_frontier.witnesses import balanced_markov_symmetry_trap


def test_collapsed_controller_with_source_marginal_readout_is_joint_stationary():
    source = four_state_aliasing_witness()
    table = np.zeros((2, 2), dtype=int)
    transition_logits = canonical_logits(table, margin=1.0).detach().cpu().numpy()
    readout_logits = source_marginal_readout_logits(source, 2)
    snapshot = exact_distribution_gradient_snapshot(
        source, transition_logits, readout_logits, 32, initial_memory=0
    )
    assert snapshot.transition_gradient_norm < 1e-12
    assert snapshot.readout_gradient_norm < 1e-12


def test_source_marginal_logits_decode_to_the_stationary_token_marginal():
    source = four_state_aliasing_witness()
    expected = source_marginal_distribution(source)
    logits = source_marginal_readout_logits(source, 3)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, np.tile(expected, (3, 1)), atol=1e-12)


def test_balanced_source_uniform_readout_is_exact_trap_despite_large_memory_gain():
    source = balanced_markov_symmetry_trap()
    np.testing.assert_allclose(
        source_marginal_distribution(source), np.array([0.5, 0.5]), atol=1e-12
    )

    table = np.zeros((2, 2), dtype=int)
    transition_logits = canonical_logits(table, margin=1.0).detach().cpu().numpy()
    uniform_readout_logits = np.zeros((2, 2), dtype=float)
    snapshot = exact_distribution_gradient_snapshot(
        source, transition_logits, uniform_readout_logits, 32, initial_memory=0
    )
    assert snapshot.transition_gradient_norm < 1e-12
    assert snapshot.readout_gradient_norm < 1e-12

    landscape = finite_horizon_hard_landscape(
        source, 2, 32, initial_memory=0
    )
    assert landscape.best_loss() < 0.36
    collapsed_loss = next(
        node.loss for node in landscape.nodes if node.signature == (0, 0, 0, 0)
    )
    assert abs(collapsed_loss - np.log(2.0)) < 1e-12
    assert collapsed_loss - landscape.best_loss() > 0.34


def test_decoder_asymmetry_creates_first_order_transition_signal():
    source = balanced_markov_symmetry_trap()
    table = np.zeros((2, 2), dtype=int)
    transition_logits = canonical_logits(table, margin=1.0).detach().cpu().numpy()
    base = source_marginal_readout_logits(source, 2)
    direction = np.array([[1.0, -1.0], [-1.0, 1.0]])

    small = exact_distribution_gradient_snapshot(
        source,
        transition_logits,
        base + 1e-5 * direction,
        32,
        initial_memory=0,
    )
    large = exact_distribution_gradient_snapshot(
        source,
        transition_logits,
        base + 1e-4 * direction,
        32,
        initial_memory=0,
    )
    ratio = large.transition_gradient_norm / small.transition_gradient_norm
    assert 9.99 < ratio < 10.01


def test_binary_canonical_surrogate_signs_are_temperature_invariant():
    source = four_state_aliasing_witness()
    table = np.array([[1, 0], [1, 0]], dtype=int)
    patterns = []
    for temperature in (0.25, 1.0, 4.0):
        field = canonical_ste_field(
            source, table, 32, margin=1.0, temperature=temperature
        )
        patterns.append(tuple(edge.surrogate_sign for edge in field.edges))
    assert patterns[0] == patterns[1] == patterns[2]


def test_explicit_readout_prior_training_is_reproducible():
    source = four_state_aliasing_witness()
    kwargs = dict(
        k=2,
        horizon=8,
        seeds=[0, 1, 2],
        readout_initialization="source_marginal",
        steps=8,
        learning_rate=0.05,
    )
    first = train_exact_distribution_ste_with_readout_prior(source, **kwargs)
    second = train_exact_distribution_ste_with_readout_prior(source, **kwargs)
    assert [
        (run.final_signature, run.final_oracle_loss) for run in first
    ] == [
        (run.final_signature, run.final_oracle_loss) for run in second
    ]
