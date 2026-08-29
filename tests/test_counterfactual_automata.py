import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier import controller_finite_horizon_log_loss
from memory_frontier.counterfactual import (
    predict_self_loop_centered_rescaled_pressure,
    predict_self_loop_counterfactual_pressure,
    stationary_token_lag_conditionals,
)
from memory_frontier.delayed import delayed_repeat_source
from memory_frontier.higher_order import (
    predict_unifilar_raw_collapsed_pressure,
    stationary_token_markovization,
)
from memory_frontier.readout_prior import exact_distribution_gradient_snapshot
from memory_frontier.surrogate import canonical_logits


def _actual_pressure(source, table, contrast, *, horizon, margin, temperature):
    k = table.shape[0]
    mu, _ = stationary_token_markovization(source)
    transition_logits = canonical_logits(table, margin=margin).detach().cpu().numpy()
    readout_logits = np.tile(np.log(mu), (k, 1))
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
    pressure = np.array(
        [gradient[0, x, 0] - gradient[0, x, 1] for x in range(source.alphabet_size)]
    )
    return pressure, gradient


def test_unreachable_hard_states_can_change_gradient_without_changing_forward_behavior():
    source = delayed_repeat_source(2, 2, 0.1)
    fully_collapsed = np.array([[0, 0], [0, 0]], dtype=int)
    counterfactual_self_loop = np.array([[0, 0], [1, 1]], dtype=int)

    # Starting from memory 0, both hard controllers stay in memory 0 forever and
    # therefore have exactly the same optimal finite-horizon forward loss.
    for horizon in (4, 20):
        a = controller_finite_horizon_log_loss(source, fully_collapsed, 0, horizon)
        b = controller_finite_horizon_log_loss(source, counterfactual_self_loop, 0, horizon)
        assert abs(a - b) < 1e-12
        assert abs(a - np.log(2.0)) < 1e-12

    contrast = np.array([0.4, -0.4])
    kwargs = dict(horizon=20, margin=0.7, temperature=0.8)
    collapsed_pressure, collapsed_gradient = _actual_pressure(
        source, fully_collapsed, contrast, **kwargs
    )
    loop_pressure, loop_gradient = _actual_pressure(
        source, counterfactual_self_loop, contrast, **kwargs
    )

    np.testing.assert_allclose(
        collapsed_pressure,
        predict_unifilar_raw_collapsed_pressure(source, 2, contrast, **kwargs),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        loop_pressure,
        predict_self_loop_counterfactual_pressure(source, 2, contrast, **kwargs),
        atol=1e-12,
        rtol=0.0,
    )

    assert np.all(collapsed_pressure < 0.0)
    assert loop_pressure[0] > 0.0
    assert loop_pressure[1] < 0.0
    assert not np.allclose(collapsed_pressure, loop_pressure)

    # The unreachable state's own transition parameters still receive no direct
    # gradient. Its hard transition structure nevertheless changes the gradient
    # on the reachable state's parameters through counterfactual sensitivity.
    assert np.linalg.norm(collapsed_gradient[1]) < 1e-12
    assert np.linalg.norm(loop_gradient[1]) < 1e-12


def test_self_loop_counterfactual_pressure_integrates_exact_lag_operators():
    source = delayed_repeat_source(2, 2, 0.1)
    mu, operators = stationary_token_lag_conditionals(source, 6)
    np.testing.assert_allclose(mu, np.array([0.5, 0.5]), atol=1e-12)

    repeat = np.array([[0.9, 0.1], [0.1, 0.9]])
    uniform = np.full((2, 2), 0.5)
    for lag in range(1, 7):
        expected = uniform if lag % 2 else np.linalg.matrix_power(repeat, lag // 2)
        np.testing.assert_allclose(operators[lag - 1], expected, atol=1e-12)

    table = np.array([[0, 0], [1, 1]], dtype=int)
    contrast = np.array([0.4, -0.4])
    kwargs = dict(horizon=20, margin=0.7, temperature=0.8)
    actual, _ = _actual_pressure(source, table, contrast, **kwargs)
    rescaled = actual / mu
    rescaled -= float(mu @ rescaled)
    predicted = predict_self_loop_centered_rescaled_pressure(
        source, 2, contrast, **kwargs
    )
    np.testing.assert_allclose(rescaled, predicted, atol=1e-12, rtol=0.0)
