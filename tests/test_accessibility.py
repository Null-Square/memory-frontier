import numpy as np

from memory_frontier import (
    delayed_repeat_source,
    dormant_chain_controller,
    exact_ste_gradient,
    four_state_aliasing_witness,
    gradient_accessibility_operator,
    linearized_transition_gradient_from_readout_direction,
)


def _canonical_logits(table: np.ndarray, margin: float = 0.7) -> np.ndarray:
    table = np.asarray(table, dtype=int)
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def test_accessibility_direction_matches_exact_gradient_finite_difference_k3():
    source = four_state_aliasing_witness()
    transition_logits = np.array(
        [
            [[0.8, -0.2, 0.1], [0.0, 0.7, -0.4]],
            [[-0.1, 0.3, 0.9], [0.6, -0.2, 0.1]],
            [[0.2, 1.0, -0.3], [-0.5, 0.4, 0.8]],
        ],
        dtype=float,
    )
    common = np.array([0.25, -0.35], dtype=float)
    direction = np.array(
        [[0.4, -0.2], [-0.1, 0.5], [0.7, -0.3]], dtype=float
    )
    horizon = 7
    temperature = 0.83

    analytic = linearized_transition_gradient_from_readout_direction(
        source,
        transition_logits,
        common,
        direction,
        horizon,
        temperature=temperature,
    )
    epsilon = 1e-6
    plus = exact_ste_gradient(
        source,
        transition_logits,
        np.tile(common, (3, 1)) + epsilon * direction,
        horizon,
        temperature=temperature,
    ).transition_logit_gradient
    minus = exact_ste_gradient(
        source,
        transition_logits,
        np.tile(common, (3, 1)) - epsilon * direction,
        horizon,
        temperature=temperature,
    ).transition_logit_gradient
    finite_difference = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(
        analytic, finite_difference, atol=2e-10, rtol=2e-9
    )


def test_operator_apply_matches_directional_formula_and_has_decoder_gauge_nullspace():
    source = four_state_aliasing_witness()
    table = np.array([[0, 1], [1, 0]], dtype=int)
    logits = _canonical_logits(table)
    common = np.array([0.3, -0.1])
    operator = gradient_accessibility_operator(
        source, logits, common, 9, temperature=0.9
    )
    direction = np.array([[0.2, -0.4], [0.6, 0.1]])
    direct = linearized_transition_gradient_from_readout_direction(
        source, logits, common, direction, 9, temperature=0.9
    )
    np.testing.assert_allclose(operator.apply(direction), direct, atol=1e-13)

    # Constant token-logit shifts in each decoder row are softmax gauge modes.
    gauge = np.array([[1.7, 1.7], [-0.4, -0.4]])
    np.testing.assert_allclose(operator.apply(gauge), 0.0, atol=1e-13)


def test_forward_identical_delayed_controllers_have_different_accessibility_rank():
    source = delayed_repeat_source(2, 3, 0.1)
    blind = np.zeros((4, 2), dtype=int)
    scaffold = dormant_chain_controller(3, 2)
    common = np.zeros(2, dtype=float)

    blind_operator = gradient_accessibility_operator(
        source,
        _canonical_logits(blind),
        common,
        20,
        temperature=0.8,
    )
    scaffold_operator = gradient_accessibility_operator(
        source,
        _canonical_logits(scaffold),
        common,
        20,
        temperature=0.8,
    )

    assert abs(blind_operator.base_loss - scaffold_operator.base_loss) < 1e-15
    assert abs(blind_operator.base_loss - np.log(2.0)) < 1e-15
    assert blind_operator.numerical_rank() == 0
    assert scaffold_operator.numerical_rank() == 1
    assert blind_operator.leading_singular_value < 1e-12
    assert abs(scaffold_operator.leading_singular_value - 0.107354361) < 1e-9
