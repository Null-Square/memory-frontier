from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np

from memory_frontier import (
    adiabatic_successor,
    controller_finite_horizon_log_loss,
    delayed_repeat_source,
    one_edit_tables,
)


def canonical_cycle(cycle: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    if len(cycle) <= 1:
        return cycle
    rotations = [cycle[i:] + cycle[:i] for i in range(len(cycle))]
    return min(rotations)


def main() -> None:
    source = delayed_repeat_source(2, 2, 0.1)
    k = 3
    horizon = 20
    signatures = [tuple(flat) for flat in product(range(k), repeat=k * 2)]
    losses: dict[tuple[int, ...], float] = {}
    successor: dict[tuple[int, ...], tuple[int, ...]] = {}

    for signature in signatures:
        table = np.asarray(signature, dtype=int).reshape(k, 2)
        losses[signature] = controller_finite_horizon_log_loss(
            source, table, 0, horizon
        )
        successor[signature] = adiabatic_successor(
            source, table, horizon
        ).target_signature

    best = min(losses.values())
    optima = {s for s, loss in losses.items() if abs(loss - best) < 1e-12}
    local_minima: set[tuple[int, ...]] = set()
    for signature in signatures:
        table = np.asarray(signature, dtype=int).reshape(k, 2)
        if all(
            losses[tuple(map(int, other.ravel()))] >= losses[signature] - 1e-12
            for _, _, _, other in one_edit_tables(table)
        ):
            local_minima.add(signature)

    attractor_basin: Counter[tuple[tuple[int, ...], ...]] = Counter()
    global_basin = 0
    hard_local_basin = 0
    nonlocal_fixed_basin = 0
    cycle_basin = 0
    for start in signatures:
        seen: dict[tuple[int, ...], int] = {}
        path: list[tuple[int, ...]] = []
        state = start
        while state not in seen:
            seen[state] = len(path)
            path.append(state)
            nxt = successor[state]
            if nxt == state:
                cycle = (state,)
                break
            state = nxt
        else:
            cycle = tuple(path[seen[state]:])
        cycle = canonical_cycle(cycle)
        attractor_basin[cycle] += 1
        if len(cycle) > 1:
            cycle_basin += 1
        elif cycle[0] in local_minima:
            hard_local_basin += 1
            global_basin += int(cycle[0] in optima)
        else:
            nonlocal_fixed_basin += 1

    fixed_points = [cycle for cycle in attractor_basin if len(cycle) == 1]
    nontrivial = [cycle for cycle in attractor_basin if len(cycle) > 1]
    print(f"controllers={len(signatures)}")
    print(f"global_optimum_loss={best:.12f}")
    print(f"fixed_points={len(fixed_points)}")
    print(f"nontrivial_cycles={len(nontrivial)}")
    print(f"cycle_lengths={sorted(Counter(map(len, nontrivial)).items())}")
    print(f"global_optimum_basin={global_basin} ({global_basin / len(signatures):.6f})")
    print(f"hard_local_minimum_basin={hard_local_basin}")
    print(f"surrogate_stable_nonlocal_basin={nonlocal_fixed_basin}")
    print(f"cycle_basin={cycle_basin}")
    print("largest_attractor_basins")
    for cycle, basin in attractor_basin.most_common(12):
        print(f"  basin={basin:3d} cycle_length={len(cycle)} cycle={cycle}")


if __name__ == "__main__":
    main()
