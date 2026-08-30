"""Exact census of accessibility order inside one forward-equivalence class.

Outside CI by design. Every sampled controller has:

- the same binary persistent Markov source;
- the same six memory states;
- the same current reachable row (state 0 is absorbing);
- the same decoder;
- the same five trainable transition directions;
- the same collapsed current forward process and loss.

Only zero-cost transition wiring in the behaviorally unreachable states 1..4 is
randomized. The general construction-order theorem is then used to measure
``(d_support, d_operator, d_loss)`` for every variant.

The fixed reference run is intentionally a single forward-equivalence class, not
an aggregate over different sources or decoders.
"""

from collections import Counter

import numpy as np

from memory_frontier import UnifilarSource
from memory_frontier.construction_order import construction_order_sandwich
from memory_frontier.forward_equivalence import is_source_horizon_dormant_rewire


SEED = 20260830
N_SAMPLES = 1000
DEPTH = 5
HORIZON = 10
DORMANT_SKIP_PROBABILITY = 0.45
DORMANT_EDGE_STRENGTH = 0.5


def persistent_binary_source(switch_probability: float = 0.1) -> UnifilarSource:
    rho = float(switch_probability)
    return UnifilarSource(
        np.array([[1.0 - rho, rho], [rho, 1.0 - rho]], dtype=float),
        np.array([[0, 1], [0, 1]], dtype=int),
    )


def collapsed_controller(depth: int = DEPTH) -> np.ndarray:
    k = depth + 1
    transition = np.zeros((k, 2, k), dtype=float)
    transition[:, :, 0] = 1.0
    return transition


def fixed_trainable_directions(depth: int = DEPTH) -> np.ndarray:
    """One entrance edit followed by a fixed dormant chain of possible edits."""
    k = depth + 1
    directions = np.zeros((depth, k, 2, k), dtype=float)

    # The only reachable trainable row: token 0 can enter dormant state 1.
    directions[0, 0, 0, 0] = -1.0
    directions[0, 0, 0, 1] = 1.0

    # Inside the dormant region, parameter i can construct i -> i+1.
    for state in range(1, depth):
        for symbol in range(2):
            directions[state, state, symbol, 0] = -1.0
            directions[state, state, symbol, state + 1] = 1.0
    return directions


def fixed_readout(depth: int = DEPTH) -> np.ndarray:
    readout = np.full((depth + 1, 2), 0.5, dtype=float)
    readout[depth] = np.array([0.9, 0.1], dtype=float)
    return readout


def random_dormant_rewire(
    rng: np.random.Generator,
    *,
    depth: int = DEPTH,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Random acyclic zero-cost skip wiring in unreachable intermediate states.

    State 0 is never changed, so every variant has the exact same current
    forward process. For each dormant intermediate state i, with probability
    ``DORMANT_SKIP_PROBABILITY`` we install a zero-cost edge to one randomly
    chosen later dormant state. The remaining probability returns to reset.

    The trainable directions are not changed across variants. Consequently any
    accessibility-order variation is caused solely by dormant base topology.
    """
    transition = collapsed_controller(depth)
    targets: list[int] = []

    for state in range(1, depth):
        if rng.random() < DORMANT_SKIP_PROBABILITY:
            target = int(rng.integers(state + 1, depth + 1))
            transition[state, :, :] = 0.0
            transition[state, :, 0] = 1.0 - DORMANT_EDGE_STRENGTH
            transition[state, :, target] = DORMANT_EDGE_STRENGTH
            targets.append(target)
        else:
            targets.append(0)

    # Final informative state resets after its prediction.
    transition[depth, :, :] = 0.0
    transition[depth, :, 0] = 1.0
    return transition, tuple(targets)


def main() -> None:
    rng = np.random.default_rng(SEED)
    source = persistent_binary_source()
    reference = collapsed_controller()
    directions = fixed_trainable_directions()
    readout = fixed_readout()

    counts: Counter[tuple[int | None, int | None, int | None]] = Counter()
    first_pattern: dict[tuple[int | None, int | None, int | None], tuple[int, ...]] = {}
    violations: list[
        tuple[int, tuple[int | None, int | None, int | None], tuple[int, ...]]
    ] = []
    non_dormant = 0

    for sample in range(N_SAMPLES):
        transition, pattern = random_dormant_rewire(rng)

        if not is_source_horizon_dormant_rewire(
            source,
            reference,
            transition,
            HORIZON,
        ):
            non_dormant += 1
            continue

        orders = construction_order_sandwich(
            source,
            transition,
            directions,
            readout,
            HORIZON,
            max_total_degree=DEPTH,
            coefficient_atol=1e-11,
        )
        counts[orders] += 1
        first_pattern.setdefault(orders, pattern)

        support, operator, loss = orders
        if (
            support is None
            or operator is None
            or loss is None
            or not support <= operator <= loss
        ):
            violations.append((sample, orders, pattern))

    print(f"seed: {SEED}")
    print(f"samples: {N_SAMPLES}")
    print(f"non-dormant rewires: {non_dormant}")
    print(f"hierarchy violations: {len(violations)}")
    if violations:
        print("first violations:")
        for item in violations[:10]:
            print(" ", item)

    print("order triples:")
    for orders, count in sorted(counts.items(), key=lambda item: item[0]):
        print(f"  {orders}: {count}; example dormant targets={first_pattern[orders]}")


if __name__ == "__main__":
    main()
