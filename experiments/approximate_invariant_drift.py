"""Audit higher-order breaking of a leading shared-route balance law.

This experiment is intentionally outside CI. It integrates the exact full
finite-horizon polynomial for the non-delayed shared-route witness and measures
how the leading invariant Q=x^2-y^2-z^2 drifts as initialization shrinks.
"""

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient
from memory_frontier.support_invariants import (
    evaluate_quadratic_invariant,
    quadratic_invariant_breaking_degree,
)


def shared_route_polynomial():
    p1 = np.array([0.1, 0.8, 0.6, 0.2], dtype=float)
    source = UnifilarSource(
        np.column_stack([1.0 - p1, p1]),
        np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int),
    )
    k = 4
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = np.zeros((3, k, 2, k), dtype=float)
    directions[0, 0, 0, 0] = -1.0
    directions[0, 0, 0, 1] = 1.0
    directions[1, 1, 0, 0] = -1.0
    directions[1, 1, 0, 2] = 1.0
    directions[2, 1, 1, 0] = -1.0
    directions[2, 1, 1, 3] = 1.0
    readout = np.full((k, 2), 0.5, dtype=float)
    readout[2] = np.array([0.9, 0.1])
    readout[3] = np.array([0.2, 0.8])
    return multivariate_controller_loss_coefficients(
        source, base, directions, readout, horizon=12
    )


def rk4(coefficients, point, time=1.0, steps=1000):
    point = np.asarray(point, dtype=float).copy()
    h = time / steps

    def velocity(value):
        return -evaluate_multivariate_polynomial_gradient(coefficients, value)

    for _ in range(steps):
        k1 = velocity(point)
        k2 = velocity(point + 0.5 * h * k1)
        k3 = velocity(point + 0.5 * h * k2)
        k4 = velocity(point + h * k3)
        point += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return point


def main():
    coefficients = shared_route_polynomial()
    weights = np.array([1.0, -1.0, -1.0])
    direction = np.array([1.0, 0.7, 0.5])
    breaking_degree = quadratic_invariant_breaking_degree(
        coefficients,
        weights,
        coefficient_atol=1e-12,
        orthogonality_atol=1e-12,
    )
    leading_degree = min(
        sum(exponent)
        for exponent, coefficient in coefficients.items()
        if sum(exponent) > 0 and abs(coefficient) > 1e-12
    )
    print(f"leading degree d={leading_degree}")
    print(f"first breaking degree p={breaking_degree}")
    print(f"predicted relative-drift power p-d={breaking_degree-leading_degree}")

    scales = np.logspace(-2, -4, 9)
    relative_drifts = []
    for scale in scales:
        initial = scale * direction
        final = rk4(coefficients, initial)
        drift = abs(
            evaluate_quadratic_invariant(final, weights)
            - evaluate_quadratic_invariant(initial, weights)
        )
        relative = drift / scale**2
        relative_drifts.append(relative)
        print(f"delta={scale:.9g} relative_drift={relative:.12g}")

    slope = np.polyfit(np.log(scales), np.log(relative_drifts), 1)[0]
    print(f"log-log slope={slope:.9f}")


if __name__ == "__main__":
    main()
