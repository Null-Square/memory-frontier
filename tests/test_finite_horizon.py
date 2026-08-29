import numpy as np

from memory_frontier.finite import (
    best_finite_horizon_deterministic_controller,
    controller_finite_horizon_log_loss,
)
from memory_frontier.witnesses import four_state_aliasing_witness


def test_horizon_one_memory_cannot_help_before_observing_any_symbol():
    source = four_state_aliasing_witness()
    f = np.array([[0, 1], [0, 1]])
    a = controller_finite_horizon_log_loss(source, f, 0, 1)
    b = controller_finite_horizon_log_loss(source, f, 1, 1)
    assert abs(a - b) < 1e-12


def test_finite_horizon_oracle_converges_toward_asymptotic_witness_value():
    source = four_state_aliasing_witness()
    losses = [
        best_finite_horizon_deterministic_controller(source, 2, t).loss
        for t in (8, 32, 128, 512)
    ]
    assert losses[-1] < losses[0]
    assert abs(losses[-1] - 0.6297909185) < 2e-3


def test_finite_horizon_spectrum_matches_direct_oracle():
    from memory_frontier.lab import deterministic_controller_spectrum

    source = four_state_aliasing_witness()
    for horizon in (1, 2, 3, 4, 16):
        direct = best_finite_horizon_deterministic_controller(source, 2, horizon)
        spectrum = deterministic_controller_spectrum(source, 2, horizon=horizon)
        assert abs(direct.loss - spectrum.optimum.loss) < 1e-12
