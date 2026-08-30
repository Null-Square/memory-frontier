import numpy as np

from memory_frontier import delayed_repeat_source
from memory_frontier.order_barrier import binary_chain_readout
from memory_frontier.parameterization import (
    compose_sparse_polynomial,
    diagonal_power_pullback_coefficients,
    diagonal_weighted_vanishing_order,
    linear_pullback_coefficients,
    polynomial_map_jacobian_at_zero,
    polynomial_vanishing_order,
)
from memory_frontier.perturbative import multivariate_controller_loss_coefficients


def _independent_chain_directions(delay: int):
    k = delay + 1
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = []

    first = np.zeros_like(base)
    first[0, 0, 0] = -1.0
    first[0, 0, 1] = 1.0
    directions.append(first)
    for state in range(1, delay):
        direction = np.zeros_like(base)
        for symbol in range(2):
            direction[state, symbol, 0] = -1.0
            direction[state, symbol, state + 1] = 1.0
        directions.append(direction)
    return base, np.asarray(directions)


def _delay_three_polynomial():
    delay = 3
    source = delayed_repeat_source(2, delay, 0.1)
    base, directions = _independent_chain_directions(delay)
    readout = binary_chain_readout(delay, 0.4)
    return multivariate_controller_loss_coefficients(
        source,
        base,
        directions,
        readout,
        horizon=11,
    )


def test_regular_nonlinear_polynomial_chart_preserves_vanishing_order():
    # L starts at degree three. The substitution has identity Jacobian but
    # nontrivial quadratic terms, so this freezes the nonlinear local-diffeomorphism
    # case rather than only a change of linear basis.
    coefficients = {
        (1, 1): -0.7,
        (3, 0): 0.2,
        (0, 4): -0.1,
        (0, 0): 2.0,
    }
    # Shift the example to leading degree two and verify it survives nonlinear
    # regular mixing exactly.
    substitutions = (
        {(1, 0): 1.0, (0, 2): 0.4},
        {(0, 1): 1.0, (2, 0): -0.3},
    )
    jacobian = polynomial_map_jacobian_at_zero(substitutions)
    pulled = compose_sparse_polynomial(coefficients, substitutions)

    assert np.allclose(jacobian, np.eye(2))
    assert polynomial_vanishing_order(coefficients) == 2
    assert polynomial_vanishing_order(pulled) == 2


def test_invertible_linear_reparameterization_preserves_exact_memory_order():
    coefficients = _delay_three_polynomial()
    matrix = np.array(
        [[1.0, 0.2, 0.0], [0.0, 1.0, -0.3], [0.1, 0.0, 1.0]],
        dtype=float,
    )
    assert abs(np.linalg.det(matrix)) > 0.9

    pulled = linear_pullback_coefficients(
        coefficients, matrix, coefficient_atol=1e-13
    )
    assert polynomial_vanishing_order(
        coefficients, coefficient_atol=1e-11
    ) == 3
    assert polynomial_vanishing_order(pulled, coefficient_atol=1e-10) == 3


def test_singular_linear_parameterization_can_raise_order_by_annihilating_leading_form():
    # L=(eps1-eps2)^2 + eps2^3 has order two. Restricting to the singular
    # rank-one chart eps1=eps2=theta1 kills the entire quadratic initial form,
    # exposing the cubic term.
    coefficients = {
        (2, 0): 1.0,
        (1, 1): -2.0,
        (0, 2): 1.0,
        (0, 3): 1.0,
    }
    singular = np.array([[1.0, 0.0], [1.0, 0.0]])
    pulled = linear_pullback_coefficients(coefficients, singular)

    assert np.linalg.matrix_rank(singular) == 1
    assert polynomial_vanishing_order(coefficients) == 2
    assert polynomial_vanishing_order(pulled) == 3
    assert pulled == {(3, 0): 1.0}


def test_diagonal_power_chart_replaces_total_degree_by_weighted_degree():
    coefficients = {
        (1, 1, 0): -2.0,
        (2, 0, 1): 0.7,
        (0, 0, 0): 3.0,
    }
    powers = (2, 3, 4)
    pulled = diagonal_power_pullback_coefficients(coefficients, powers)

    assert pulled[(2, 3, 0)] == -2.0
    assert pulled[(4, 0, 4)] == 0.7
    assert diagonal_weighted_vanishing_order(coefficients, powers) == 5


def test_uniform_singular_power_multiplies_exact_finite_memory_order():
    coefficients = _delay_three_polynomial()
    pulled = diagonal_power_pullback_coefficients(
        coefficients, (2, 2, 2), coefficient_atol=1e-13
    )

    assert polynomial_vanishing_order(
        coefficients, coefficient_atol=1e-11
    ) == 3
    assert polynomial_vanishing_order(pulled, coefficient_atol=1e-10) == 6


def test_regular_scaling_changes_euclidean_flow_rate_but_not_objective_order():
    # For L(eps)=-C eps^d and eps=a theta, the objective order stays d, while
    # theta-Euclidean flow mapped back to eps is accelerated by a^2 relative to
    # eps-Euclidean flow. Freeze the algebra for one representative degree.
    degree = 4
    coefficient = -0.7
    scale = 3.0
    original = {(degree,): coefficient}
    pulled = linear_pullback_coefficients(original, [[scale]])

    assert polynomial_vanishing_order(original) == degree
    assert polynomial_vanishing_order(pulled) == degree
    assert np.isclose(pulled[(degree,)], coefficient * scale**degree)

    epsilon = 0.02
    direct_velocity = -degree * coefficient * epsilon ** (degree - 1)
    theta = epsilon / scale
    theta_velocity = -degree * pulled[(degree,)] * theta ** (degree - 1)
    mapped_velocity = scale * theta_velocity
    assert np.isclose(mapped_velocity / direct_velocity, scale**2)
