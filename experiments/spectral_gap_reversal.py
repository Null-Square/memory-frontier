"""Audit local construction-mode reversal near a repeated bilinear singular value.

Outside CI by design. The quadratic route matrix is tuned to have singular gap
O(delta). The full exact horizon-12 Hessian contains an O(delta) cubic
correction, so the two effects compete in the local gradient-flow Jacobian.
"""

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_hessian


Q_STAR = 0.46869740167476925


def coefficients(q00, *, max_total_degree=None):
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
    readout[3] = np.array([1.0 - q00, q00])
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


def route_matrix(poly):
    return np.array(
        [
            [
                -poly.get((1, 0, 1, 0), 0.0),
                -poly.get((1, 0, 0, 1), 0.0),
            ],
            [
                -poly.get((0, 1, 1, 0), 0.0),
                -poly.get((0, 1, 0, 1), 0.0),
            ],
        ],
        dtype=float,
    )


def canonical_modes():
    first = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2.0)
    second = np.array([0.0, 1.0, 0.0, 1.0]) / np.sqrt(2.0)
    return np.column_stack([first, second])


def projection_difference(delta, kappa):
    poly = coefficients(Q_STAR - kappa * delta)
    jacobian = -evaluate_multivariate_polynomial_hessian(
        poly, np.array([delta, delta, 0.0, 0.0])
    )
    values, vectors = np.linalg.eigh(jacobian)
    mode = vectors[:, int(np.argmax(values))]
    projections = np.abs(canonical_modes().T @ mode)
    return float(projections[0] - projections[1]), projections


def finite_delta_boundary(delta):
    lower = 0.0
    upper = 0.05
    assert projection_difference(delta, lower)[0] < 0.0
    assert projection_difference(delta, upper)[0] > 0.0
    for _ in range(35):
        middle = 0.5 * (lower + upper)
        if projection_difference(delta, middle)[0] > 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def main():
    tie = coefficients(Q_STAR, max_total_degree=3)
    matrix = route_matrix(tie)
    strength = matrix[0, 0]
    cubic = {
        exponent: coefficient
        for exponent, coefficient in tie.items()
        if sum(exponent) == 3
    }
    cubic_jacobian = -evaluate_multivariate_polynomial_hessian(
        cubic, np.array([1.0, 1.0, 0.0, 0.0])
    )
    modes = canonical_modes()
    projected = modes.T @ cubic_jacobian @ modes

    derivative = (10.0 / 21.0) * (
        0.1 / Q_STAR - 0.9 / (1.0 - Q_STAR)
    )
    asymptotic = 0.63 * strength / (-derivative)

    print("degenerate quadratic route matrix:")
    print(matrix)
    print("cubic correction / tied strength:")
    print(projected / strength)
    print(f"asymptotic critical kappa: {asymptotic:.12f}")

    print("\nfinite-delta critical kappa")
    for delta in (0.04, 0.02, 0.01, 0.005, 0.0025):
        print(f"{delta:<10g} {finite_delta_boundary(delta):.12f}")

    print("\nfrozen below/above-boundary projections at delta=0.01")
    for kappa in (0.018, 0.025):
        difference, projections = projection_difference(0.01, kappa)
        print(
            f"kappa={kappa:.4f}: projections={projections}, "
            f"mode1-mode2={difference:.9f}"
        )


if __name__ == "__main__":
    main()
