"""Stratified exact-family audit of the construction-order sandwich.

Outside CI by design. We generate random source/controller families at planted
support depths 1 through 4. Each base controller remains inside one uniform-
readout memory class, while perturbative transition directions may advance by at
most one construction layer. A planted route guarantees the requested depth;
random branching, distractor edits, repeated parameter assignments, and random
source/readout statistics vary the actual polynomial geometry.

For every family we compare:

    d_support   combinatorial source-valid perturbative-edge distance
    d_operator  first nonzero quotient construction-operator degree
    d_loss      first nonzero scalar finite-horizon loss degree

The exact theorem predicts d_support <= d_operator <= d_loss. Generic continuous
readouts should normally saturate both inequalities, while deliberately tuned
cancellations are tested separately in deterministic regressions.
"""

from collections import Counter

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.construction_order import construction_order_sandwich


SOURCE_TRANSITIONS = np.array(
    [[0, 1], [2, 3], [0, 1], [2, 3]], dtype=int
)


def random_source(rng: np.random.Generator) -> UnifilarSource:
    p1 = rng.uniform(0.08, 0.92, size=4)
    return UnifilarSource(
        np.column_stack([1.0 - p1, p1]), SOURCE_TRANSITIONS
    )


def stratified_family(rng: np.random.Generator, depth: int):
    if depth < 1:
        raise ValueError("depth must be positive")

    alphabet = 2
    n_parameters = 4
    neutral = 2 * depth
    k = neutral + 2

    # Two behaviorally neutral memories per construction layer. Base transitions
    # remain within the current layer, so zero-cost dynamics can branch and mix
    # without ever advancing toward an informative decoder.
    base = np.zeros((k, alphabet, k), dtype=float)
    for level in range(depth):
        states = [2 * level, 2 * level + 1]
        for memory in states:
            for symbol in range(alphabet):
                base[memory, symbol, int(rng.choice(states))] = 1.0
    base[neutral:, :, 0] = 1.0

    directions = np.zeros((n_parameters, k, alphabet, k), dtype=float)

    # Plant one route that advances exactly one layer per perturbative edge.
    # Parameters are assigned randomly and may repeat along the route.
    for level in range(depth):
        memory = 2 * level
        symbol = int(rng.integers(0, alphabet))
        base_target = int(np.argmax(base[memory, symbol]))
        target = neutral if level == depth - 1 else 2 * (level + 1)
        parameter = int(rng.integers(0, n_parameters))
        weight = float(rng.uniform(0.4, 1.0))
        directions[parameter, memory, symbol, base_target] -= weight
        directions[parameter, memory, symbol, target] += weight

    # Add random distractors. They either stay inside a layer or advance by one
    # layer, so no shortcut can beat the planted support depth.
    for _ in range(3 * depth):
        level = int(rng.integers(0, depth))
        states = [2 * level, 2 * level + 1]
        memory = int(rng.choice(states))
        symbol = int(rng.integers(0, alphabet))
        base_target = int(np.argmax(base[memory, symbol]))
        parameter = int(rng.integers(0, n_parameters))

        if rng.random() < 0.5:
            candidates = [state for state in states if state != base_target]
            if not candidates:
                continue
            target = int(rng.choice(candidates))
        elif level == depth - 1:
            target = int(rng.choice([neutral, neutral + 1]))
        else:
            target = int(rng.choice([2 * (level + 1), 2 * (level + 1) + 1]))

        if target == base_target:
            continue
        weight = float(rng.uniform(0.1, 0.8))
        directions[parameter, memory, symbol, base_target] -= weight
        directions[parameter, memory, symbol, target] += weight

    readout = np.full((k, alphabet), 0.5, dtype=float)
    for memory in (neutral, neutral + 1):
        q = float(rng.uniform(0.08, 0.92))
        if abs(q - 0.5) < 0.05:
            q = 0.61 if q >= 0.5 else 0.39
        readout[memory] = np.array([1.0 - q, q])
    return base, directions, readout


def main(seed: int = 20260830, n_cases: int = 1000):
    rng = np.random.default_rng(seed)
    triples = Counter()
    violations = []

    for case in range(n_cases):
        depth = 1 + case % 4
        source = random_source(rng)
        base, directions, readout = stratified_family(rng, depth)
        support, operator, loss = construction_order_sandwich(
            source,
            base,
            directions,
            readout,
            horizon=depth + 5,
            max_total_degree=depth + 1,
            coefficient_atol=2e-11,
        )
        triples[(support, operator, loss)] += 1
        if support != depth:
            violations.append((case, depth, support, operator, loss))
        elif operator is None or operator < support:
            violations.append((case, depth, support, operator, loss))
        elif loss is None or loss < operator:
            violations.append((case, depth, support, operator, loss))

    print(f"seed: {seed}")
    print(f"cases: {n_cases}")
    print(f"violations: {len(violations)}")
    if violations:
        print("first violations:", violations[:10])
    print("triples:")
    for triple, count in sorted(triples.items()):
        print(f"  {triple}: {count}")


if __name__ == "__main__":
    main()
