import numpy as np

from memory_frontier.support_invariants import (
    evaluate_quadratic_invariant,
    exponent_support_matrix,
    quadratic_invariant_basis,
    quadratic_invariant_derivative,
)


def _projector(basis):
    basis = np.asarray(basis, dtype=float)
    return basis.T @ basis


def test_single_monomial_recovers_weighted_squared_balance_space():
    coefficients = {(2, 1, 3): -0.7}
    matrix = exponent_support_matrix(coefficients)
    basis = quadratic_invariant_basis(coefficients)

    assert matrix.shape == (1, 3)
    assert basis.shape == (2, 3)
    assert np.allclose(matrix @ basis.T, 0.0, atol=1e-13)

    expected = np.asarray(
        [
            [1.0 / 2.0, -1.0, 0.0],
            [1.0 / 2.0, 0.0, -1.0 / 3.0],
        ],
        dtype=float,
    )
    # Compare invariant subspaces rather than SVD basis orientation.
    expected, _ = np.linalg.qr(expected.T)
    assert np.allclose(
        _projector(basis),
        expected @ expected.T,
        atol=1e-13,
    )


def test_two_shared_routes_leave_exactly_one_quadratic_balance_law():
    # L = -a*x*y - b*y*z. The support matrix rows are (1,1,0),(0,1,1),
    # whose nullspace is span(1,-1,1).
    coefficients = {
        (1, 1, 0): -0.4,
        (0, 1, 1): -0.7,
    }
    basis = quadratic_invariant_basis(coefficients)
    assert basis.shape == (1, 3)

    expected = np.asarray([1.0, -1.0, 1.0], dtype=float)
    expected /= np.linalg.norm(expected)
    assert np.isclose(abs(np.dot(basis[0], expected)), 1.0, atol=1e-13)

    weights = np.asarray([1.0, -1.0, 1.0])
    for point in (
        (0.2, 0.3, 0.4),
        (0.7, 0.5, 0.9),
        (1.1, 0.8, 0.6),
    ):
        assert np.isclose(
            quadratic_invariant_derivative(coefficients, point, weights),
            0.0,
            atol=1e-14,
        )


def test_closing_triangle_destroys_all_nontrivial_quadratic_invariants():
    coefficients = {
        (1, 1, 0): -0.4,
        (0, 1, 1): -0.7,
        (1, 0, 1): -0.2,
    }
    matrix = exponent_support_matrix(coefficients)
    basis = quadratic_invariant_basis(coefficients)

    assert np.linalg.matrix_rank(matrix) == 3
    assert basis.shape == (0, 3)

    old_weights = np.asarray([1.0, -1.0, 1.0])
    derivative = quadratic_invariant_derivative(
        coefficients,
        (0.3, 0.4, 0.5),
        old_weights,
    )
    assert not np.isclose(derivative, 0.0, atol=1e-14)


def test_nullspace_condition_is_sufficient_for_arbitrary_supported_polynomial():
    coefficients = {
        (2, 1, 0, 1): -0.3,
        (0, 1, 2, 1): 0.5,
        (1, 0, 1, 2): -0.2,
        (0, 0, 0, 0): 7.0,
        (3, 0, 0, 0): 0.0,
    }
    basis = quadratic_invariant_basis(coefficients)
    matrix = exponent_support_matrix(coefficients)
    assert basis.shape[0] == 4 - np.linalg.matrix_rank(matrix)

    point = np.asarray([0.31, 0.42, 0.53, 0.64])
    for weights in basis:
        assert np.isclose(
            quadratic_invariant_derivative(coefficients, point, weights),
            0.0,
            atol=2e-14,
        )


def test_non_null_weight_has_nonzero_derivative_somewhere():
    coefficients = {
        (2, 0, 1): -0.3,
        (0, 1, 2): 0.4,
    }
    weights = np.asarray([1.0, 1.0, 1.0])
    matrix = exponent_support_matrix(coefficients)
    assert not np.allclose(matrix @ weights, 0.0)

    # Distinct monomials cannot cancel identically. One generic positive point
    # already witnesses failure of conservation for this fixture.
    assert not np.isclose(
        quadratic_invariant_derivative(
            coefficients,
            (0.37, 0.51, 0.68),
            weights,
        ),
        0.0,
        atol=1e-14,
    )


def test_quadratic_invariant_value_matches_definition():
    point = (0.2, 0.3, 0.4)
    weights = (1.0, -1.0, 1.0)
    assert np.isclose(
        evaluate_quadratic_invariant(point, weights),
        0.2**2 - 0.3**2 + 0.4**2,
    )
