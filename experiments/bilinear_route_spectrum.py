"""Compare exact bilinear leading modes with the full finite-memory polynomial.

Outside CI by design. The witness has two learned entrances and two learned exits,
so its degree-two computation geometry is a full-rank 2x2 route matrix rather
than an independent-route or rank-one shared-entrance special case.
"""

import numpy as np
from scipy.integrate import solve_ivp

from memory_frontier import UnifilarSource
from memory_frontier.bilinear_routes import bilinear_route_flow, bilinear_route_spectrum
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient


def witness_coefficients(*, max_total_degree=None):
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    source = UnifilarSource(
        np.column_stack([1.0 - p1, p1]),
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int),
    )

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
    readout[3] = np.array([0.9, 0.1])
    readout[4] = np.array([0.2, 0.8])
    readout[5] = np.array([0.4, 0.6])
    readout[6] = np.array([0.8, 0.2])

    return multivariate_controller_loss_coefficients(
        source,
        base,
        directions,
        readout,
        horizon=12,
        max_total_degree=max_total_degree,
    )


def route_matrix(coefficients):
    return np.array(
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


def full_flow(coefficients, initial, time):
    def velocity(_, point):
        return -evaluate_multivariate_polynomial_gradient(coefficients, point)

    solution = solve_ivp(
        velocity,
        (0.0, time),
        np.asarray(initial, dtype=float),
        rtol=2e-10,
        atol=2e-13,
        max_step=0.05,
    )
    return solution.y[:, -1]


def main():
    full = witness_coefficients()
    quadratic = witness_coefficients(max_total_degree=2)
    matrix = route_matrix(quadratic)
    u, singular_values, vt = bilinear_route_spectrum(matrix)

    print("route matrix A:")
    print(matrix)
    print("singular values:", singular_values)
    print("left singular vectors U:")
    print(u)
    print("right singular vectors V:")
    print(vt.T)
    print("singular-value ratio:", singular_values[0] / singular_values[1])

    final_time = 20.0
    deltas = np.array([0.01, 0.005, 0.002, 0.001, 0.0005], dtype=float)
    errors = []
    print("\ndelta          relative full-vs-bilinear error")
    for delta in deltas:
        left0 = np.array([delta, delta], dtype=float)
        right0 = np.zeros(2, dtype=float)
        lead_left, lead_right = bilinear_route_flow(
            matrix, left0, right0, final_time
        )
        leading = np.concatenate([lead_left, lead_right])
        observed = full_flow(
            full,
            np.array([delta, delta, 0.0, 0.0], dtype=float),
            final_time,
        )
        error = np.linalg.norm(observed - leading) / np.linalg.norm(leading)
        errors.append(error)
        print(f"{delta:<14g} {error:.12f}")

    slope = np.polyfit(np.log(deltas[-4:]), np.log(errors[-4:]), 1)[0]
    print(f"\nsmall-delta log-log error slope: {slope:.6f}")


if __name__ == "__main__":
    main()
