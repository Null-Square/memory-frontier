from math import log

import numpy as np

from memory_frontier import (
    binary_chain_readout,
    binary_delay_chain_leading_gain_coefficient,
    binary_soft_chain_transition,
    delayed_repeat_source,
    smooth_controller_finite_horizon_log_loss,
)


def test_delay_chain_first_nonzero_gain_has_order_equal_delay():
    rho = 0.1
    horizon = 30
    gap = 0.3
    epsilons = {2: 0.002, 3: 0.003, 4: 0.006, 5: 0.012}

    for delay, epsilon in epsilons.items():
        source = delayed_repeat_source(2, delay, rho)
        readout = binary_chain_readout(delay, gap)
        baseline = smooth_controller_finite_horizon_log_loss(
            source,
            binary_soft_chain_transition(delay, 0.0),
            readout,
            horizon,
        )
        perturbed = smooth_controller_finite_horizon_log_loss(
            source,
            binary_soft_chain_transition(delay, epsilon),
            readout,
            horizon,
        )
        coefficient = binary_delay_chain_leading_gain_coefficient(
            delay, rho, horizon, gap
        )
        observed = (baseline - perturbed) / epsilon**delay
        assert abs(baseline - log(2.0)) < 1e-12
        assert abs(observed - coefficient) / coefficient < 0.01


def test_missing_any_chain_link_erases_predictive_gain_exactly():
    delay = 4
    source = delayed_repeat_source(2, delay, 0.1)
    readout = binary_chain_readout(delay, 0.3)
    links = np.array([0.4, 0.3, 0.0, 0.8])
    loss = smooth_controller_finite_horizon_log_loss(
        source,
        binary_soft_chain_transition(delay, links),
        readout,
        30,
    )
    assert abs(loss - log(2.0)) < 1e-12


def test_prewired_downstream_chain_reduces_barrier_to_first_order():
    delay = 4
    rho = 0.1
    horizon = 30
    gap = 0.3
    epsilon = 0.002
    source = delayed_repeat_source(2, delay, rho)
    readout = binary_chain_readout(delay, gap)
    baseline_links = np.ones(delay)
    baseline_links[0] = 0.0
    perturbed_links = baseline_links.copy()
    perturbed_links[0] = epsilon

    baseline = smooth_controller_finite_horizon_log_loss(
        source,
        binary_soft_chain_transition(delay, baseline_links),
        readout,
        horizon,
    )
    perturbed = smooth_controller_finite_horizon_log_loss(
        source,
        binary_soft_chain_transition(delay, perturbed_links),
        readout,
        horizon,
    )
    coefficient = binary_delay_chain_leading_gain_coefficient(
        delay, rho, horizon, gap
    )
    observed = (baseline - perturbed) / epsilon
    assert abs(observed - coefficient) / coefficient < 0.01
