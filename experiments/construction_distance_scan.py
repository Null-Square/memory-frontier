from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.perturbative import (
    affine_controller_loss_coefficients,
    leading_perturbative_order,
    minimum_decoder_construction_cost,
)


def observable_markov_source(matrix: np.ndarray) -> UnifilarSource:
    matrix = np.asarray(matrix, dtype=float)
    q = matrix.shape[0]
    transitions = np.tile(np.arange(q, dtype=int), (q, 1))
    return UnifilarSource(matrix, transitions)


def random_dormant_family(k: int, rng: np.random.Generator):
    base = np.zeros((k, 2, k), dtype=float)
    base[:, :, 0] = 1.0
    direction = np.zeros_like(base)

    for symbol in range(2):
        if rng.random() < 0.5:
            target = int(rng.integers(1, k))
            direction[0, symbol, 0] = -1.0
            direction[0, symbol, target] = 1.0
    for memory in range(1, k):
        for symbol in range(2):
            draw = rng.random()
            if draw < 0.35:
                target = int(rng.integers(0, k))
                base[memory, symbol] = 0.0
                base[memory, symbol, target] = 1.0
            elif draw < 0.8:
                target = int(rng.integers(1, k))
                direction[memory, symbol, 0] = -1.0
                direction[memory, symbol, target] = 1.0
    return base, direction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=260830)
    parser.add_argument("--horizon", type=int, default=9)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    pairs: Counter[tuple[int, int]] = Counter()
    violations = 0
    cancellations = 0
    tested = 0

    for _ in range(args.trials):
        p00, p10 = rng.uniform(0.1, 0.9, size=2)
        source = observable_markov_source(
            np.array([[p00, 1.0 - p00], [p10, 1.0 - p10]])
        )
        k = int(rng.integers(3, 7))
        base, direction = random_dormant_family(k, rng)
        readout = np.full((k, 2), 0.5, dtype=float)
        target_count = int(rng.integers(1, min(3, k)))
        for memory in rng.choice(np.arange(1, k), size=target_count, replace=False):
            q = float(rng.uniform(0.1, 0.9))
            if abs(q - 0.5) < 0.03:
                q = 0.77
            readout[int(memory)] = np.array([q, 1.0 - q])

        distance = minimum_decoder_construction_cost(
            source, base, direction, readout, args.horizon
        )
        if distance is None or distance == 0:
            continue
        coefficients = affine_controller_loss_coefficients(
            source, base, direction, readout, args.horizon
        )
        order = leading_perturbative_order(coefficients, atol=1e-9)
        if order is None:
            continue
        tested += 1
        pairs[(distance, order)] += 1
        violations += int(order < distance)
        cancellations += int(order > distance)

    print(f"tested={tested}")
    print(f"lower_bound_violations={violations}")
    print(f"generic_cancellations={cancellations}")
    print("distance_order_counts")
    for pair in sorted(pairs):
        print(f"  {pair}: {pairs[pair]}")


if __name__ == "__main__":
    main()
