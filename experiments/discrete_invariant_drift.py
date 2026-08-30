"""Separate higher-order loss drift from finite-step optimizer drift.

This exact-polynomial audit is outside CI. On the shared-route non-delayed
finite-memory witness, continuous full-loss drift scales as O(delta) after
normalizing Q by delta^2, while vanilla GD differs from continuous flow by
O(eta). The script also estimates the coefficient-level crossover.
"""

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient
from memory_frontier.support_invariants import evaluate_quadratic_invariant


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


def gradient_descent(coefficients, point, step_size, time=1.0):
    point = np.asarray(point, dtype=float).copy()
    steps = int(round(time / step_size))
    if not np.isclose(steps * step_size, time, atol=1e-14):
        raise ValueError("time must be an integer multiple of step_size")
    for _ in range(steps):
        point -= step_size * evaluate_multivariate_polynomial_gradient(
            coefficients, point
        )
    return point


def main():
    coefficients = shared_route_polynomial()
    weights = np.array([1.0, -1.0, -1.0])
    direction = np.array([1.0, 0.7, 0.5])

    # Continuous higher-order coefficient: normalized drift ~ A*delta.
    deltas = np.logspace(-2, -4, 9)
    continuous_drifts = []
    for delta in deltas:
        initial = delta * direction
        final = rk4(coefficients, initial)
        relative = abs(
            evaluate_quadratic_invariant(final, weights)
            - evaluate_quadratic_invariant(initial, weights)
        ) / delta**2
        continuous_drifts.append(relative)
    continuous_slope, continuous_log_coefficient = np.polyfit(
        np.log(deltas), np.log(continuous_drifts), 1
    )

    # Discretization coefficient: normalized GD-flow discrepancy ~ B*eta.
    delta = 1e-4
    initial = delta * direction
    continuous = rk4(coefficients, initial)
    etas = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    discrete_drifts = []
    for eta in etas:
        discrete = gradient_descent(coefficients, initial, eta)
        relative = abs(
            evaluate_quadratic_invariant(discrete, weights)
            - evaluate_quadratic_invariant(continuous, weights)
        ) / delta**2
        discrete_drifts.append(relative)
    discrete_slope, discrete_log_coefficient = np.polyfit(
        np.log(etas), np.log(discrete_drifts), 1
    )

    a = float(np.exp(continuous_log_coefficient))
    b = float(np.exp(discrete_log_coefficient))
    crossover_ratio = a / b

    print(f"continuous slope in delta: {continuous_slope:.9f}")
    print(f"continuous coefficient A: {a:.9f}")
    print(f"discrete-minus-flow slope in eta: {discrete_slope:.9f}")
    print(f"discrete coefficient B: {b:.9f}")
    print(f"estimated crossover eta/delta: {crossover_ratio:.6f}")


if __name__ == "__main__":
    main()
