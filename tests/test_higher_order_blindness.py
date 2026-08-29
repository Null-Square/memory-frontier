import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier import (
    bayes_log_loss,
    controller_log_loss,
    four_state_aliasing_witness,
)
from memory_frontier.delayed import (
    delayed_repeat_entropy_rate,
    delayed_repeat_source,
    delayed_shift_register_controller,
)
from memory_frontier.higher_order import (
    predict_unifilar_centered_rescaled_pressure,
    predict_unifilar_raw_collapsed_pressure,
    stationary_token_markovization,
    unifilar_collapsed_pressure_accessibility_margin,
)
from memory_frontier.readout_prior import exact_distribution_gradient_snapshot
from memory_frontier.surrogate import canonical_logits


def _actual_pressure(source, memory_states, contrast, *, horizon, margin, temperature):
    mu, _ = stationary_token_markovization(source)
    collapsed = np.zeros((memory_states, source.alphabet_size), dtype=int)
    transition_logits = canonical_logits(collapsed, margin=margin).detach().cpu().numpy()
    readout_logits = np.tile(np.log(mu), (memory_states, 1))
    readout_logits[1] = np.log(mu) + np.asarray(contrast, dtype=float)
    snapshot = exact_distribution_gradient_snapshot(
        source,
        transition_logits,
        readout_logits,
        horizon,
        initial_memory=0,
        temperature=temperature,
    )
    gradient = snapshot.transition_gradient
    return np.array(
        [gradient[0, x, 0] - gradient[0, x, 1] for x in range(source.alphabet_size)]
    )


def test_generic_unifilar_local_pressure_depends_only_on_token_markovization():
    source = four_state_aliasing_witness()
    contrast = np.array([0.3, -0.2])
    kwargs = dict(horizon=13, margin=0.7, temperature=0.8)
    actual = _actual_pressure(source, 2, contrast, **kwargs)
    predicted = predict_unifilar_raw_collapsed_pressure(
        source, 2, contrast, **kwargs
    )
    np.testing.assert_allclose(actual, predicted, atol=1e-12, rtol=0.0)

    mu, _ = stationary_token_markovization(source)
    rescaled = actual / mu
    rescaled -= float(mu @ rescaled)
    centered = predict_unifilar_centered_rescaled_pressure(
        source, 2, contrast, **kwargs
    )
    np.testing.assert_allclose(rescaled, centered, atol=1e-12, rtol=0.0)


def test_delayed_repeat_has_no_one_step_token_predictability_but_large_memory_value():
    for q, delay, rho in ((2, 2, 0.1), (2, 3, 0.1), (3, 2, 0.2)):
        source = delayed_repeat_source(q, delay, rho)
        mu, token_q = stationary_token_markovization(source)
        np.testing.assert_allclose(mu, np.full(q, 1.0 / q), atol=1e-12)
        np.testing.assert_allclose(
            token_q, np.full((q, q), 1.0 / q), atol=1e-12
        )

        expected_bayes = delayed_repeat_entropy_rate(q, rho)
        assert abs(bayes_log_loss(source) - expected_bayes) < 1e-12

        shift = delayed_shift_register_controller(q, delay)
        assert abs(controller_log_loss(source, shift, 0) - expected_bayes) < 1e-12

        collapsed = np.zeros_like(shift)
        collapsed_loss = controller_log_loss(source, collapsed, 0)
        assert abs(collapsed_loss - np.log(q)) < 1e-12
        assert collapsed_loss > expected_bayes


def test_delayed_repeat_repels_every_nontrivial_decoder_specialization_locally():
    source = delayed_repeat_source(2, 2, 0.1)
    memory_states = 4
    kwargs = dict(horizon=16, margin=0.7, temperature=0.8)

    for contrast in (
        np.array([0.2, -0.2]),
        np.array([1.0, -1.0]),
        np.array([-0.4, 0.4]),
    ):
        actual = _actual_pressure(source, memory_states, contrast, **kwargs)
        predicted = predict_unifilar_raw_collapsed_pressure(
            source, memory_states, contrast, **kwargs
        )
        np.testing.assert_allclose(actual, predicted, atol=1e-12, rtol=0.0)
        assert np.all(actual < 0.0)
        assert unifilar_collapsed_pressure_accessibility_margin(
            source, contrast
        ) < 0.0

    zero = np.zeros(2)
    np.testing.assert_allclose(
        _actual_pressure(source, memory_states, zero, **kwargs),
        np.zeros(2),
        atol=1e-12,
    )


def test_higher_order_blindness_gap_can_approach_log_alphabet_size():
    q = 3
    rho = 1e-6
    source = delayed_repeat_source(q, 2, rho)
    shift = delayed_shift_register_controller(q, 2)
    gain = np.log(q) - controller_log_loss(source, shift, 0)
    assert gain > 0.99 * np.log(q)
