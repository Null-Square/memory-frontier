import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier import (
    bayes_log_loss,
    controller_finite_horizon_log_loss,
    controller_log_loss,
    source_stationary_distribution,
)
from memory_frontier.families import (
    last_symbol_controller,
    symmetric_repeat_entropy_rate,
    symmetric_repeat_last_symbol_finite_horizon_loss,
    symmetric_repeat_source,
)
from memory_frontier.readout_prior import (
    exact_distribution_gradient_snapshot,
    source_marginal_readout_logits,
)
from memory_frontier.surrogate import canonical_logits


def test_symmetric_repeat_family_has_uniform_stationary_law_and_exact_entropy_rate():
    for q in (2, 3, 5):
        source = symmetric_repeat_source(q, 0.1)
        np.testing.assert_allclose(
            source_stationary_distribution(source),
            np.full(q, 1.0 / q),
            atol=1e-12,
        )
        expected = symmetric_repeat_entropy_rate(q, 0.1)
        assert abs(bayes_log_loss(source) - expected) < 1e-12
        assert abs(controller_log_loss(source, last_symbol_controller(q), 0) - expected) < 1e-12


def test_last_symbol_finite_horizon_closed_form_matches_exact_oracle_scorer():
    for q, rho, horizon in ((2, 0.1, 32), (3, 0.05, 17), (5, 0.2, 11)):
        source = symmetric_repeat_source(q, rho)
        table = last_symbol_controller(q)
        exact = controller_finite_horizon_log_loss(source, table, 0, horizon)
        closed = symmetric_repeat_last_symbol_finite_horizon_loss(q, rho, horizon)
        assert abs(exact - closed) < 1e-12


def test_collapsed_uniform_predictor_is_exact_straight_through_stationary_point():
    for q in (2, 3, 5):
        source = symmetric_repeat_source(q, 0.1)
        collapsed = np.zeros((q, q), dtype=int)
        transition_logits = canonical_logits(collapsed, margin=1.0).detach().cpu().numpy()
        readout_logits = source_marginal_readout_logits(source, q)
        snapshot = exact_distribution_gradient_snapshot(
            source,
            transition_logits,
            readout_logits,
            16,
            initial_memory=0,
        )
        assert abs(snapshot.loss - np.log(q)) < 1e-12
        assert snapshot.transition_gradient_norm < 1e-11
        assert snapshot.readout_gradient_norm < 1e-11


def test_symmetry_trap_can_hide_nearly_log_q_nats_of_memory_value():
    q = 8
    rho = 1e-6
    source = symmetric_repeat_source(q, rho)
    optimal_asymptotic = controller_log_loss(source, last_symbol_controller(q), 0)
    hidden_gain = np.log(q) - optimal_asymptotic
    assert hidden_gain > 0.99 * np.log(q)
