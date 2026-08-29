import numpy as np

from memory_frontier import (
    binary_delay_matched_first_order_pressure_coefficient,
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


def _row_zero_pressure(gradient) -> np.ndarray:
    return np.array(
        [
            gradient.transition_logit_gradient[0, x, 0]
            - gradient.transition_logit_gradient[0, x, 1]
            for x in range(2)
        ]
    )


def test_delay_matched_dormant_chain_restores_first_order_gradient_access():
    rho = 0.1
    horizon = 20
    margin = 0.7
    temperature = 0.8
    epsilon = 1e-5

    for delay in (2, 3, 4, 5):
        source = delayed_repeat_source(2, delay, rho)
        table = dormant_chain_controller(delay, 2)
        transition_logits = _canonical_logits(table, margin)
        plus = exact_ste_gradient(
            source,
            transition_logits,
            binary_dormant_chain_readout_logits(delay, epsilon),
            horizon,
            temperature=temperature,
        )
        minus = exact_ste_gradient(
            source,
            transition_logits,
            binary_dormant_chain_readout_logits(delay, -epsilon),
            horizon,
            temperature=temperature,
        )
        derivative = (
            _row_zero_pressure(plus) - _row_zero_pressure(minus)
        ) / (2 * epsilon)
        coefficient = binary_delay_matched_first_order_pressure_coefficient(
            delay, rho, horizon, margin, temperature
        )
        np.testing.assert_allclose(
            derivative,
            np.array([coefficient, -coefficient]),
            atol=1e-10,
            rtol=1e-9,
        )


def test_too_shallow_dormant_chain_is_first_order_blind_to_delayed_repeat():
    delay = 4
    rho = 0.1
    horizon = 20
    margin = 0.7
    temperature = 0.8
    epsilon = 1e-5
    source = delayed_repeat_source(2, delay, rho)

    for chain_depth in (1, 2, 3):
        table = dormant_chain_controller(chain_depth, 2)
        transition_logits = _canonical_logits(table, margin)
        plus = exact_ste_gradient(
            source,
            transition_logits,
            binary_dormant_chain_readout_logits(chain_depth, epsilon),
            horizon,
            temperature=temperature,
        )
        minus = exact_ste_gradient(
            source,
            transition_logits,
            binary_dormant_chain_readout_logits(chain_depth, -epsilon),
            horizon,
            temperature=temperature,
        )
        derivative = (
            _row_zero_pressure(plus) - _row_zero_pressure(minus)
        ) / (2 * epsilon)
        np.testing.assert_allclose(derivative, np.zeros(2), atol=1e-10)


def test_scaffold_states_are_forward_unreachable_from_reset():
    table = dormant_chain_controller(5, 2)
    memory = 0
    for token in (0, 1, 1, 0, 1, 0, 0, 1):
        memory = int(table[memory, token])
        assert memory == 0
