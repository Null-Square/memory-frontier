import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier import (
    delayed_repeat_source,
    exact_ste_gradient,
    four_state_aliasing_witness,
    hard_value_gradient,
    source_stationary_distribution,
)
from memory_frontier.families import symmetric_repeat_source
from memory_frontier.readout_prior import exact_distribution_gradient_snapshot


def _canonical_numpy_logits(table: np.ndarray, margin: float) -> np.ndarray:
    table = np.asarray(table, dtype=int)
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def test_exact_ste_gradient_matches_autograd_beyond_binary_k2():
    rng = np.random.default_rng(20260830)
    cases = (
        (four_state_aliasing_witness(), 3, 7, 0, 0.6),
        (symmetric_repeat_source(3, 0.2), 4, 5, 1, 1.3),
        (delayed_repeat_source(2, 2, 0.1), 3, 6, 2, 0.8),
    )
    for source, k, horizon, initial_memory, temperature in cases:
        for _ in range(3):
            transition_logits = rng.normal(
                size=(k, source.alphabet_size, k)
            )
            readout_logits = rng.normal(size=(k, source.alphabet_size))
            exact = exact_ste_gradient(
                source,
                transition_logits,
                readout_logits,
                horizon,
                initial_memory=initial_memory,
                temperature=temperature,
            )
            autograd = exact_distribution_gradient_snapshot(
                source,
                transition_logits,
                readout_logits,
                horizon,
                initial_memory=initial_memory,
                temperature=temperature,
            )
            assert abs(exact.loss - autograd.loss) < 1e-12
            np.testing.assert_allclose(
                exact.transition_logit_gradient,
                autograd.transition_gradient,
                atol=1e-12,
                rtol=1e-12,
            )
            np.testing.assert_allclose(
                exact.readout_logit_gradient,
                autograd.readout_gradient,
                atol=1e-12,
                rtol=1e-12,
            )


def test_pre_jacobian_transition_gradient_matches_finite_difference():
    source = four_state_aliasing_witness()
    table = np.array([[0, 1], [1, 0]], dtype=int)
    readout = np.array([[0.72, 0.28], [0.19, 0.81]], dtype=float)
    horizon = 6
    exact = hard_value_gradient(source, table, readout, horizon)

    k = table.shape[0]
    hard_tensor = np.zeros((k, source.alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(source.alphabet_size):
            hard_tensor[m, x, table[m, x]] = 1.0

    def relaxed_loss(transition_tensor: np.ndarray) -> float:
        stationary = source_stationary_distribution(source)
        dist = np.zeros((source.n_states, k), dtype=float)
        dist[:, 0] = stationary
        log_readout = np.log(readout)
        total = 0.0
        for _ in range(horizon):
            total -= np.einsum(
                "sm,sx,mx->", dist, source.emissions, log_readout
            )
            nxt = np.zeros_like(dist)
            for s in range(source.n_states):
                for m in range(k):
                    for x in range(source.alphabet_size):
                        s2 = source.transitions[s, x]
                        nxt[s2] += (
                            dist[s, m]
                            * source.emissions[s, x]
                            * transition_tensor[m, x]
                        )
            dist = nxt
        return float(total / horizon)

    epsilon = 1e-6
    coordinate = (0, 1, 0)
    plus = hard_tensor.copy()
    minus = hard_tensor.copy()
    plus[coordinate] += epsilon
    minus[coordinate] -= epsilon
    finite_difference = (
        relaxed_loss(plus) - relaxed_loss(minus)
    ) / (2 * epsilon)
    assert abs(
        finite_difference - exact.transition_tensor_gradient[coordinate]
    ) < 1e-8


def test_forward_equivalent_hard_controllers_can_have_different_exact_gradients():
    source = delayed_repeat_source(2, 2, 0.1)
    readout_logits = np.array([[0.0, 0.0], [0.2, -0.2]])
    collapsed = np.array([[0, 0], [0, 0]], dtype=int)
    scaffold = np.array([[0, 0], [1, 1]], dtype=int)
    margin = 0.7

    first = exact_ste_gradient(
        source,
        _canonical_numpy_logits(collapsed, margin),
        readout_logits,
        12,
    )
    second = exact_ste_gradient(
        source,
        _canonical_numpy_logits(scaffold, margin),
        readout_logits,
        12,
    )
    assert abs(first.loss - second.loss) < 1e-15

    first_pressure = np.array(
        [
            first.transition_logit_gradient[0, x, 0]
            - first.transition_logit_gradient[0, x, 1]
            for x in range(2)
        ]
    )
    second_pressure = np.array(
        [
            second.transition_logit_gradient[0, x, 0]
            - second.transition_logit_gradient[0, x, 1]
            for x in range(2)
        ]
    )
    assert np.all(first_pressure < 0.0)
    assert second_pressure[0] > 0.0
    assert second_pressure[1] < 0.0
