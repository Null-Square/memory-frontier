"""Random exact-family audit of the construction-order sandwich.

Outside CI by design. We generate many small source/controller families whose base
controller remains inside one uniform-readout memory class while perturbative
transition directions can open routes into two informative target memories.

For every accepted family we compare:

    d_support   combinatorial source-valid perturbative-edge distance
    d_operator  first nonzero quotient construction-operator degree
    d_loss      first nonzero scalar finite-horizon loss degree

The exact theorem predicts d_support <= d_operator <= d_loss. Generic continuous
readouts should usually saturate both inequalities, while deliberately tuned
cancellations can make them strict.
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


def random_family(rng: np.random.Generator):
    alphabet = 2
    k = 6
    neutral = 4
    n_parameters = 4

    # The base controller is deterministic and stays inside memories 0..3,
    # all of which have the same uniform readout. Thus its current predictor is
    # behaviorally collapsed even though the zero-cost neutral topology varies.
    base = np.zeros((k, alphabet, k), dtype=float)
    for memory in range(neutral):
        for symbol in range(alphabet):
            target = int(rng.integers(0, neutral))
            base[memory, symbol, target] = 1.0
    base[neutral:, :, 0] = 1.0

    directions = np.zeros((n_parameters, k, alphabet, k), dtype=float)
    for parameter in range(n_parameters):
        # Each parameter edits several source-memory rows. Every edit transfers
        # probability away from that row's base target into another memory.
        for _ in range(int(rng.integers(2, 5))):
            memory = int(rng.integers(0, neutral))
            symbol = int(rng.integers(0, alphabet))
            base_target = int(np.argmax(base[memory, symbol]))
            candidates = [target for target in range(k) if target != base_target]
            target = int(rng.choice(candidates))
            weight = float(rng.uniform(0.2, 1.0))
            directions[parameter, memory, symbol, base_target] -= weight
            directions[parameter, memory, symbol, target] += weight

    readout = np.full((k, alphabet), 0.5, dtype=float)
    for memory in range(neutral, k):
        q = float(rng.uniform(0.08, 0.92))
        if abs(q - 0.5) < 0.04:
            q = 0.58 if q >= 0.5 else 0.42
        readout[memory] = np.array([1.0 - q, q])
    return base, directions, readout


def main(seed: int = 20260830, accepted_target: int = 1000):
    rng = np.random.default_rng(seed)
    horizon = 7
    accepted = 0
    attempted = 0
    triples = Counter()
    support_orders = Counter()
    strict_support = 0
    strict_decoder = 0
    violations = []

    while accepted < accepted_target and attempted < 20 * accepted_target:
        attempted += 1
        source = random_source(rng)
        base, directions, readout = random_family(rng)
        support, operator, loss = construction_order_sandwich(
            source,
            base,
            directions,
            readout,
            horizon,
            max_total_degree=5,
            coefficient_atol=2e-11,
        )
        if support is None or support <= 0 or support > 5:
            continue
        accepted += 1
        triples[(support, operator, loss)] += 1
        support_orders[support] += 1

        if operator is None or operator < support:
            violations.append((support, operator, loss))
            continue
        if loss is None or loss < operator:
            violations.append((support, operator, loss))
            continue
        strict_support += int(operator > support)
        strict_decoder += int(loss > operator)

    print(f"seed: {seed}")
    print(f"attempted: {attempted}")
    print(f"accepted: {accepted}")
    print(f"violations: {len(violations)}")
    if violations:
        print("first violations:", violations[:10])
    print("support-order counts:", dict(sorted(support_orders.items())))
    print(f"operator > support: {strict_support}")
    print(f"loss > operator: {strict_decoder}")
    print("most common triples:")
    for triple, count in triples.most_common(20):
        print(f"  {triple}: {count}")


if __name__ == "__main__":
    main()
