"""Compare exact shared-route leading modes with the full finite-memory polynomial.

Outside CI by design. The frozen non-delayed witness has two beneficial routes
sharing one entrance: suffix 00 and suffix 01. Starting with a small positive
shared entrance and zero exits, the exact quadratic leading flow predicts both
exit threshold times from the route coefficients alone.
"""

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient
from memory_frontier.shared_routes import shared_entrance_zero_exit_crossing_times


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


def leading_strengths(coefficients):
    active = {
        exponent: coefficient
        for exponent, coefficient in coefficients.items()
        if sum(exponent) > 0 and abs(coefficient) > 1e-12
    }
    degree = min(sum(exponent) for exponent in active)
    leading = {
        exponent: coefficient
        for exponent, coefficient in active.items()
        if sum(exponent) == degree
    }
    return np.array(
        [-leading[(1, 1, 0)], -leading[(1, 0, 1)]], dtype=float
    )


def full_crossing_times(
    coefficients,
    *,
    initial_entrance=1e-3,
    threshold=0.01,
    step=0.005,
    max_time=40.0,
):
    point = np.array([initial_entrance, 0.0, 0.0], dtype=float)
    times = np.full(2, np.inf, dtype=float)
    time = 0.0

    def velocity(value):
        return -evaluate_multivariate_polynomial_gradient(coefficients, value)

    while time < max_time and not np.all(np.isfinite(times)):
        k1 = velocity(point)
        k2 = velocity(point + 0.5 * step * k1)
        k3 = velocity(point + 0.5 * step * k2)
        k4 = velocity(point + step * k3)
        point += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        time += step
        for route, parameter in enumerate((1, 2)):
            if not np.isfinite(times[route]) and point[parameter] >= threshold:
                times[route] = time
    return times


def main():
    coefficients = shared_route_polynomial()
    strengths = leading_strengths(coefficients)
    predicted = shared_entrance_zero_exit_crossing_times(
        strengths,
        initial_entrance=1e-3,
        threshold=0.01,
    )
    observed = full_crossing_times(coefficients)

    print(f"route strengths [00,01]: {strengths}")
    print(f"strength ratio 00/01: {strengths[0]/strengths[1]:.9f}")
    print(f"leading crossing times: {predicted}")
    print(f"full-polynomial crossing times: {observed}")
    print(f"relative timing errors: {(observed-predicted)/predicted}")


if __name__ == "__main__":
    main()
