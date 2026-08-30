"""Exact branching-memory falsification test for leading-monomial route prediction.

This experiment is intentionally outside CI. It compares the weighted order of
leading loss monomials against the route actually built first by deterministic
projected gradient descent on the exact finite-horizon loss polynomial.
"""

from __future__ import annotations

import numpy as np

from memory_frontier import delayed_repeat_source
from memory_frontier.order_barrier import (
    binary_chain_readout,
    binary_soft_chain_transition,
)
from memory_frontier.perturbative import multivariate_controller_loss_coefficients


def polynomial_gradient(coefficients, point):
    point = np.asarray(point, dtype=float)
    gradient = np.zeros_like(point)
    for exponent, coefficient in coefficients.items():
        exponent = np.asarray(exponent, dtype=int)
        for parameter, multiplicity in enumerate(exponent):
            if multiplicity == 0:
                continue
            term = float(coefficient) * multiplicity
            for index, power in enumerate(exponent):
                derivative_power = power - (1 if index == parameter else 0)
                if derivative_power:
                    term *= point[index] ** derivative_power
            gradient[parameter] += term
    return gradient


def parallel_two_link_fixture():
    source = delayed_repeat_source(2, 2, 0.1)
    k = 5  # 0, A1, A2, B1, B2
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    directions = []

    direction = np.zeros_like(base)
    direction[0, 0, 0] = -1.0
    direction[0, 0, 1] = 1.0
    directions.append(direction)

    direction = np.zeros_like(base)
    direction[1, :, 0] = -1.0
    direction[1, :, 2] = 1.0
    directions.append(direction)

    direction = np.zeros_like(base)
    direction[0, 0, 0] = -1.0
    direction[0, 0, 3] = 1.0
    directions.append(direction)

    direction = np.zeros_like(base)
    direction[3, :, 0] = -1.0
    direction[3, :, 4] = 1.0
    directions.append(direction)

    final_decoder = binary_chain_readout(2, 0.4)[2]
    readout = np.full((k, 2), 0.5, dtype=float)
    readout[2] = final_decoder
    readout[4] = final_decoder
    return source, base, np.asarray(directions), readout


def projected_race(
    coefficients,
    scale,
    *,
    learning_rate=0.02,
    threshold=0.02,
    max_steps=5_000,
):
    point = np.array([scale, scale, scale**0.1, scale**2.0], dtype=float)
    for step in range(max_steps + 1):
        route_a = min(point[0], point[1]) >= threshold
        route_b = min(point[2], point[3]) >= threshold
        if route_a or route_b:
            return step, point, route_a, route_b
        if step == max_steps:
            break

        point -= learning_rate * polynomial_gradient(coefficients, point)
        point = np.maximum(point, 0.0)
        entrance_mass = point[0] + point[2]
        if entrance_mass > 1.0:
            point[[0, 2]] /= entrance_mass
        point[[1, 3]] = np.minimum(point[[1, 3]], 1.0)

    return max_steps, point, False, False


def main():
    source, base, directions, readout = parallel_two_link_fixture()
    coefficients = multivariate_controller_loss_coefficients(
        source, base, directions, readout, horizon=12
    )
    leading = sorted(
        (exponent, coefficient)
        for exponent, coefficient in coefficients.items()
        if sum(exponent) == 2 and abs(coefficient) > 1e-12
    )

    weights = np.array([1.0, 1.0, 0.1, 2.0])
    print("leading degree-2 loss terms")
    for exponent, coefficient in leading:
        print(
            f"  {exponent}: coefficient={coefficient:.12f}, "
            f"weighted_order={np.dot(weights, exponent):.3f}"
        )

    print("\nprojected exact-polynomial gradient races")
    for scale in (1e-8, 1e-12, 1e-20):
        initial = np.array([scale, scale, scale**0.1, scale**2.0])
        gradient = polynomial_gradient(coefficients, initial)
        step, final, route_a, route_b = projected_race(coefficients, scale)
        winner = "A" if route_a and not route_b else "B" if route_b and not route_a else "none"
        print(
            f"  scale={scale:.0e}: weighted prediction=A, winner={winner}, "
            f"steps={step}, initial_grad={gradient}, final={final}"
        )

    # Reused-parameter audit: one parameter controls both missing delay-2 links.
    chain_base = binary_soft_chain_transition(2, 0.0)
    chain_full = binary_soft_chain_transition(2, 1.0)
    chain_coefficients = multivariate_controller_loss_coefficients(
        source,
        chain_base,
        np.asarray([chain_full - chain_base]),
        binary_chain_readout(2, 0.4),
        horizon=10,
    )
    chain_support = {
        exponent: coefficient
        for exponent, coefficient in chain_coefficients.items()
        if sum(exponent) > 0 and abs(coefficient) > 1e-12
    }
    print("\nreused-parameter leading support")
    first_degree = min(sum(exponent) for exponent in chain_support)
    print(
        {
            exponent: coefficient
            for exponent, coefficient in chain_support.items()
            if sum(exponent) == first_degree
        }
    )


if __name__ == "__main__":
    main()
