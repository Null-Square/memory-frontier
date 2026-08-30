"""Deterministic branching audit for gradient-vector-field route predictors.

This experiment stays outside CI. It compares two local predictors against
projected gradient descent on the full exact finite-horizon loss polynomial:

1. exact isolated two-link leading-monomial completion times;
2. a local Hessian/constant-acceleration correction using the full polynomial.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from memory_frontier import delayed_repeat_source
from memory_frontier.order_barrier import binary_chain_readout
from memory_frontier.perturbative import multivariate_controller_loss_coefficients
from memory_frontier.route_race import (
    leading_two_link_route_times,
    multivariate_polynomial_gradient_coefficients,
    quadratic_two_link_route_times,
)


ROUTE_A = (0, 1)
ROUTE_B = (2, 3)


def parallel_two_link_fixture():
    source = delayed_repeat_source(2, 2, 0.1)
    k = 5
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
    coefficients = multivariate_controller_loss_coefficients(
        source, base, np.asarray(directions), readout, horizon=12
    )
    return coefficients


def compile_polynomial(coefficients):
    if not coefficients:
        return np.zeros((0, 0), dtype=int), np.zeros(0, dtype=float)
    exponents = np.asarray(list(coefficients), dtype=int)
    values = np.asarray([coefficients[key] for key in coefficients], dtype=float)
    return exponents, values


def compile_gradient(coefficients):
    return [
        compile_polynomial(component)
        for component in multivariate_polynomial_gradient_coefficients(coefficients)
    ]


def evaluate_compiled(compiled, point):
    gradient = np.zeros(len(compiled), dtype=float)
    for parameter, (exponents, coefficients) in enumerate(compiled):
        if len(coefficients) == 0:
            continue
        monomials = np.prod(point[None, :] ** exponents, axis=1)
        gradient[parameter] = float(coefficients @ monomials)
    return gradient


def projected_winner(
    compiled_gradient,
    point,
    *,
    learning_rate=0.02,
    threshold=0.02,
    max_steps=10_000,
):
    point = np.asarray(point, dtype=float).copy()
    for step in range(max_steps + 1):
        route_a = min(point[0], point[1]) >= threshold
        route_b = min(point[2], point[3]) >= threshold
        if route_a or route_b:
            if route_a and route_b:
                return "tie", step
            return ("A" if route_a else "B"), step
        if step == max_steps:
            break

        point -= learning_rate * evaluate_compiled(compiled_gradient, point)
        point = np.maximum(point, 0.0)
        entrance_mass = point[0] + point[2]
        if entrance_mass > 1.0:
            point[[0, 2]] /= entrance_mass
        point[[1, 3]] = np.minimum(point[[1, 3]], 1.0)
    return "none", max_steps


def classify_route_times(routes, *, relative_tie_tolerance=1e-8):
    times = {pair: time for pair, _, time in routes}
    first = times[ROUTE_A]
    second = times[ROUTE_B]
    if np.isinf(first) and np.isinf(second):
        return "tie"
    scale = max(1.0, abs(first), abs(second))
    if abs(first - second) <= relative_tie_tolerance * scale:
        return "tie"
    return "A" if first < second else "B"


def main():
    coefficients = parallel_two_link_fixture()
    compiled_gradient = compile_gradient(coefficients)
    scale = 1e-8
    threshold = 0.02
    weights = (0.2, 0.6, 1.0)

    records = []
    for exponent_weights in product(weights, repeat=4):
        point = scale ** np.asarray(exponent_weights, dtype=float)
        if (
            min(point[0], point[1]) >= threshold
            or min(point[2], point[3]) >= threshold
        ):
            continue

        leading = classify_route_times(
            leading_two_link_route_times(coefficients, point, threshold)
        )
        curvature = classify_route_times(
            quadratic_two_link_route_times(coefficients, point, threshold)
        )
        observed, steps = projected_winner(
            compiled_gradient, point, threshold=threshold
        )
        records.append((exponent_weights, leading, curvature, observed, steps))

    print(f"route races: {len(records)}")
    print(f"observed ties: {sum(row[3] == 'tie' for row in records)}")

    for name, prediction_index in (("leading", 1), ("curvature", 2)):
        predictor_decisive = sum(
            row[prediction_index] != "tie" for row in records
        )
        comparable = [
            row
            for row in records
            if row[prediction_index] != "tie" and row[3] in {"A", "B"}
        ]
        correct = sum(row[prediction_index] == row[3] for row in comparable)
        print(
            f"{name}: predictor_decisive={predictor_decisive}, "
            f"comparable={len(comparable)}, correct={correct}"
        )
        for row in comparable:
            if row[prediction_index] != row[3]:
                print(
                    f"  mismatch weights={row[0]} "
                    f"predicted={row[prediction_index]} observed={row[3]} "
                    f"steps={row[4]}"
                )


if __name__ == "__main__":
    main()
