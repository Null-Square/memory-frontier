import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.bilinear_routes import (
    bilinear_positive_growth_modes,
    project_symmetric_operator,
)
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_hessian


Q_STAR = 0.46869740167476925


def _source():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    return UnifilarSource(
        np.column_stack([1.0 - p1, p1]),
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int),
    )


def _diagonal_mode_family(q00, *, max_total_degree=None):
    source = _source()
    k = 7
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = np.zeros((4, k, 2, k), dtype=float)

    directions[0, 0, 0, 0] = -1.0
    directions[0, 0, 0, 1] = 1.0
    directions[1, 0, 1, 0] = -1.0
    directions[1, 0, 1, 2] = 1.0

    directions[2, 1, 0, 0] = -1.0
    directions[2, 1, 0, 3] = 1.0
    directions[2, 2, 0, 0] = -1.0
    directions[2, 2, 0, 5] = 1.0
    directions[3, 1, 1, 0] = -1.0
    directions[3, 1, 1, 4] = 1.0
    directions[3, 2, 1, 0] = -1.0
    directions[3, 2, 1, 6] = 1.0

    readout = np.full((k, 2), 0.5, dtype=float)
    readout[3] = np.array([1.0 - q00, q00])
    # Off-diagonal suffixes 01 and 10 remain uninformative so the degree-two
    # route matrix is diagonal.
    readout[4] = np.array([0.5, 0.5])
    readout[5] = np.array([0.5, 0.5])
    readout[6] = np.array([0.8, 0.2])

    return multivariate_controller_loss_coefficients(
        source,
        base,
        directions,
        readout,
        horizon=12,
        max_total_degree=max_total_degree,
    )


def _route_matrix(coefficients):
    return np.array(
        [
            [
                -coefficients.get((1, 0, 1, 0), 0.0),
                -coefficients.get((1, 0, 0, 1), 0.0),
            ],
            [
                -coefficients.get((0, 1, 1, 0), 0.0),
                -coefficients.get((0, 1, 0, 1), 0.0),
            ],
        ],
        dtype=float,
    )


def _canonical_positive_modes():
    first = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2.0)
    second = np.array([0.0, 1.0, 0.0, 1.0]) / np.sqrt(2.0)
    return np.column_stack([first, second])


def _top_full_growth_mode(coefficients, point):
    hessian = evaluate_multivariate_polynomial_hessian(coefficients, point)
    values, vectors = np.linalg.eigh(-hessian)
    index = int(np.argmax(values))
    return float(values[index]), vectors[:, index]


def test_positive_bilinear_growth_modes_are_jacobian_eigenvectors():
    matrix = np.array([[2.0, 0.0], [0.0, 0.7]])
    singular_values, modes = bilinear_positive_growth_modes(matrix)
    jacobian = np.block(
        [
            [np.zeros((2, 2)), matrix],
            [matrix.T, np.zeros((2, 2))],
        ]
    )

    assert np.allclose(modes.T @ modes, np.eye(2), atol=1e-14)
    assert np.allclose(jacobian @ modes, modes * singular_values, atol=1e-14)


def test_exact_degenerate_witness_has_nontrivial_cubic_mode_operator():
    coefficients = _diagonal_mode_family(Q_STAR, max_total_degree=3)
    matrix = _route_matrix(coefficients)
    strength = matrix[0, 0]

    assert np.allclose(matrix, strength * np.eye(2), atol=5e-14)
    assert np.isclose(strength, 0.02294580440735208, atol=5e-14)

    cubic = {
        exponent: coefficient
        for exponent, coefficient in coefficients.items()
        if sum(exponent) == 3
    }
    cubic_jacobian = -evaluate_multivariate_polynomial_hessian(
        cubic, np.array([1.0, 1.0, 0.0, 0.0])
    )
    projected = project_symmetric_operator(
        cubic_jacobian, _canonical_positive_modes()
    )

    expected = strength * np.array(
        [[-1.71, -0.405], [-0.405, -1.08]], dtype=float
    )
    assert np.allclose(projected, expected, atol=2e-14)


def test_o_delta_singular_gap_can_pick_opposite_full_local_mode():
    # q00=q_star-kappa*delta opens a positive quadratic singular gap favoring
    # the first canonical mode. For kappa below the cubic critical value, the
    # complete local Hessian still has larger projection onto the second mode.
    kappa = 0.018
    modes = _canonical_positive_modes()

    for delta in (0.02, 0.01, 0.005, 0.0025):
        coefficients = _diagonal_mode_family(Q_STAR - kappa * delta)
        matrix = _route_matrix(coefficients)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        assert matrix[0, 0] > matrix[1, 1]
        assert singular_values[0] > singular_values[1]

        growth_rate, mode = _top_full_growth_mode(
            coefficients, np.array([delta, delta, 0.0, 0.0])
        )
        projections = np.abs(modes.T @ mode)
        assert growth_rate > 0.0
        assert projections[1] > projections[0]


def test_above_asymptotic_spectral_boundary_recovers_leading_mode():
    kappa = 0.025
    modes = _canonical_positive_modes()

    for delta in (0.02, 0.01, 0.005, 0.0025):
        coefficients = _diagonal_mode_family(Q_STAR - kappa * delta)
        matrix = _route_matrix(coefficients)
        assert matrix[0, 0] > matrix[1, 1]

        _, mode = _top_full_growth_mode(
            coefficients, np.array([delta, delta, 0.0, 0.0])
        )
        projections = np.abs(modes.T @ mode)
        assert projections[0] > projections[1]


def test_asymptotic_critical_spectral_gap_constant():
    coefficients = _diagonal_mode_family(Q_STAR, max_total_degree=3)
    strength = _route_matrix(coefficients)[0, 0]
    derivative = (10.0 / 21.0) * (
        0.1 / Q_STAR - 0.9 / (1.0 - Q_STAR)
    )
    splitting_rate = -derivative

    # The cubic projected diagonal difference is 0.63*b. Equal mode weights are
    # reached when the O(delta) leading spectral splitting cancels that amount.
    critical = 0.63 * strength / splitting_rate
    assert np.isclose(critical, 0.020503478171891763, atol=2e-15)
