import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.bilinear_routes import (
    bilinear_global_balance,
    bilinear_modal_balance_invariants,
    bilinear_route_flow,
    bilinear_route_spectrum,
)
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.shared_routes import shared_entrance_quadratic_flow


def _two_by_two_route_witness():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    source = UnifilarSource(
        np.column_stack([1.0 - p1, p1]),
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int),
    )

    k = 7
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = np.zeros((4, k, 2, k), dtype=float)

    # Two entrance parameters encode the first symbol.
    directions[0, 0, 0, 0] = -1.0
    directions[0, 0, 0, 1] = 1.0
    directions[1, 0, 1, 0] = -1.0
    directions[1, 0, 1, 2] = 1.0

    # Two exit parameters are shared across both intermediate states. Each exit
    # therefore participates in two distinct useful length-two computations.
    directions[2, 1, 0, 0] = -1.0
    directions[2, 1, 0, 3] = 1.0
    directions[2, 2, 0, 0] = -1.0
    directions[2, 2, 0, 5] = 1.0
    directions[3, 1, 1, 0] = -1.0
    directions[3, 1, 1, 4] = 1.0
    directions[3, 2, 1, 0] = -1.0
    directions[3, 2, 1, 6] = 1.0

    readout = np.full((k, 2), 0.5, dtype=float)
    readout[3] = np.array([0.9, 0.1])
    readout[4] = np.array([0.2, 0.8])
    readout[5] = np.array([0.4, 0.6])
    readout[6] = np.array([0.8, 0.2])

    coefficients = multivariate_controller_loss_coefficients(
        source,
        base,
        directions,
        readout,
        horizon=12,
        max_total_degree=2,
    )
    route_matrix = np.array(
        [
            [
                -coefficients[(1, 0, 1, 0)],
                -coefficients[(1, 0, 0, 1)],
            ],
            [
                -coefficients[(0, 1, 1, 0)],
                -coefficients[(0, 1, 0, 1)],
            ],
        ],
        dtype=float,
    )
    return coefficients, route_matrix


def test_diagonal_bilinear_flow_decouples_into_hyperbolic_modes():
    matrix = np.diag([2.0, 1.0])
    left, right = bilinear_route_flow(
        matrix,
        initial_left=(1.0, 0.0),
        initial_right=(0.0, 1.0),
        time=0.3,
    )

    assert np.allclose(left, [np.cosh(0.6), np.sinh(0.3)], atol=1e-13)
    assert np.allclose(right, [np.sinh(0.6), np.cosh(0.3)], atol=1e-13)


def test_rank_one_solver_reduces_to_shared_entrance_solution():
    strengths = np.array([0.7, 0.2, 0.4])
    entrance = 0.3
    exits = np.array([0.1, 0.5, -0.2])
    time = 0.8

    shared_entrance, shared_exits = shared_entrance_quadratic_flow(
        strengths, entrance, exits, time
    )
    left, right = bilinear_route_flow(
        strengths[None, :],
        initial_left=(entrance,),
        initial_right=exits,
        time=time,
    )

    assert np.isclose(left[0], shared_entrance, atol=2e-14)
    assert np.allclose(right, shared_exits, atol=2e-14)


def test_singular_mode_balance_laws_and_nullspace_coordinates_are_exact():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    left0 = np.array([0.3, -0.4])
    right0 = np.array([0.2, 0.5, 0.7])
    invariants0 = bilinear_modal_balance_invariants(matrix, left0, right0)
    global0 = bilinear_global_balance(left0, right0)

    left1, right1 = bilinear_route_flow(matrix, left0, right0, time=1.1)
    invariants1 = bilinear_modal_balance_invariants(matrix, left1, right1)

    assert np.allclose(invariants1, invariants0, atol=2e-13)
    assert np.isclose(bilinear_global_balance(left1, right1), global0, atol=2e-13)
    assert np.isclose(right1[2], right0[2], atol=1e-14)


def test_exact_finite_memory_witness_has_full_rank_route_matrix():
    coefficients, matrix = _two_by_two_route_witness()

    active_quadratic = {
        exponent
        for exponent, coefficient in coefficients.items()
        if sum(exponent) == 2 and abs(coefficient) > 1e-12
    }
    assert active_quadratic == {
        (1, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 0, 1),
    }
    assert np.all(matrix > 0.0)
    assert np.linalg.matrix_rank(matrix) == 2

    u, singular_values, vt = bilinear_route_spectrum(matrix)
    assert np.allclose(u @ np.diag(singular_values) @ vt, matrix, atol=2e-15)
    assert np.allclose(
        singular_values,
        [0.17684673120575908, 0.02243003052995202],
        rtol=0.0,
        atol=2e-15,
    )


def test_dominant_singular_mode_controls_longer_time_growth_when_seeded():
    _, matrix = _two_by_two_route_witness()
    _, singular_values, vt = bilinear_route_spectrum(matrix)
    left0 = np.array([1e-3, 1e-3])
    right0 = np.zeros(2)

    _, right = bilinear_route_flow(matrix, left0, right0, time=30.0)
    dominant = vt.T[:, 0]

    # SVD signs are arbitrary, so compare one-dimensional subspaces.
    cosine = abs(float(np.dot(right, dominant))) / np.linalg.norm(right)
    assert singular_values[0] > 7.0 * singular_values[1]
    assert cosine > 0.999
