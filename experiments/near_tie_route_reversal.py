"""Audit near-tie route reversal on the exact shared-route finite-memory witness.

Outside CI by design. The quadratic shared-mode theory picks the route with the
larger leading coefficient. Near the coefficient-tie surface, cubic terms enter
at the same local order as an O(delta) leading-strength advantage and can reverse
both the initial gradient direction and the first threshold crossing.
"""

import numpy as np
from scipy.integrate import solve_ivp

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient
from memory_frontier.shared_routes import shared_entrance_zero_exit_crossing_times


Q_STAR = 0.46869740167476925


def coefficients(q00, *, max_total_degree=None):
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
    readout[2] = np.array([1.0 - q00, q00])
    readout[3] = np.array([0.2, 0.8])
    return multivariate_controller_loss_coefficients(
        source,
        base,
        directions,
        readout,
        horizon=12,
        max_total_degree=max_total_degree,
    )


def strengths(poly):
    return np.array(
        [-poly[(1, 1, 0)], -poly[(1, 0, 1)]], dtype=float
    )


def exit_velocity_gap(poly, delta):
    velocity = -evaluate_multivariate_polynomial_gradient(
        poly, np.array([delta, 0.0, 0.0], dtype=float)
    )
    return float(velocity[1] - velocity[2])


def full_crossing_times(poly, delta, ratio=2.0, max_time=80.0):
    target = ratio * delta

    def velocity(_, point):
        return -evaluate_multivariate_polynomial_gradient(poly, point)

    def exit_00(_, point):
        return point[1] - target

    def exit_01(_, point):
        return point[2] - target

    for event in (exit_00, exit_01):
        event.direction = 1
        event.terminal = False

    solution = solve_ivp(
        velocity,
        (0.0, max_time),
        np.array([delta, 0.0, 0.0]),
        events=(exit_00, exit_01),
        rtol=3e-9,
        atol=1e-12,
        max_step=0.08,
    )
    return np.array(
        [events[0] if len(events) else np.inf for events in solution.t_events]
    )


def critical_offset(delta, *, crossing=False):
    lower = -0.002
    upper = 0.0

    def gap(offset):
        poly = coefficients(Q_STAR + offset)
        if crossing:
            times = full_crossing_times(poly, delta)
            return float(times[1] - times[0])
        return exit_velocity_gap(poly, delta)

    assert gap(lower) > 0.0
    assert gap(upper) < 0.0
    for _ in range(18 if crossing else 40):
        middle = 0.5 * (lower + upper)
        if gap(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def main():
    tie = coefficients(Q_STAR, max_total_degree=3)
    tie_strengths = strengths(tie)
    derivative = (10.0 / 21.0) * (
        0.1 / Q_STAR - 0.9 / (1.0 - Q_STAR)
    )
    kappa_star = 0.45 * tie_strengths[1] / (-derivative)

    print(f"quadratic tie decoder q*: {Q_STAR:.15f}")
    print(f"tied strengths: {tie_strengths}")
    print(
        "cubic ratios [00,01]:",
        tie[(2, 1, 0)] / tie_strengths[0],
        tie[(2, 0, 1)] / tie_strengths[1],
    )
    print(f"asymptotic initial-gradient kappa*: {kappa_star:.12f}")

    delta = 0.01
    q00 = Q_STAR - 0.0001
    full = coefficients(q00)
    lead = strengths(full)
    predicted = shared_entrance_zero_exit_crossing_times(
        lead, initial_entrance=delta, threshold=2.0 * delta
    )
    observed = full_crossing_times(full, delta)
    print("\nclean trajectory reversal")
    print(f"q00: {q00:.15f}")
    print(f"leading strengths [00,01]: {lead}")
    print(f"leading crossing times [00,01]: {predicted}")
    print(f"full crossing times [00,01]: {observed}")

    print("\ncritical decoder offset / delta")
    print("delta        initial-gradient     threshold-crossing")
    for local_scale in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        initial = critical_offset(local_scale, crossing=False) / local_scale
        route = critical_offset(local_scale, crossing=True) / local_scale
        print(f"{local_scale:<12g} {initial:<20.10f} {route:<20.10f}")


if __name__ == "__main__":
    main()
