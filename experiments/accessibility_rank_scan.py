from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np

from memory_frontier import (
    delayed_repeat_source,
    gradient_accessibility_operator,
)


def canonical_logits(table: np.ndarray, margin: float) -> np.ndarray:
    table = np.asarray(table, dtype=int)
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def main() -> None:
    source = delayed_repeat_source(2, 3, 0.1)
    k = 4
    alphabet_size = 2
    ranks: Counter[int] = Counter()
    leading: list[float] = []

    # Row 0 is the only forward-reachable row and is fixed collapsed. Enumerate
    # every possible hard topology on the remaining three unreachable rows.
    for flat in product(range(k), repeat=(k - 1) * alphabet_size):
        table = np.zeros((k, alphabet_size), dtype=int)
        table[1:] = np.asarray(flat, dtype=int).reshape(k - 1, alphabet_size)
        operator = gradient_accessibility_operator(
            source,
            canonical_logits(table, 0.7),
            np.zeros(alphabet_size),
            20,
            temperature=0.8,
        )
        ranks[operator.numerical_rank()] += 1
        leading.append(operator.leading_singular_value)

    print("rank counts")
    for rank in sorted(ranks):
        print(f"  rank {rank}: {ranks[rank]}")
    print(f"topologies: {sum(ranks.values())}")
    print(f"leading singular min:    {min(leading):.12g}")
    print(f"leading singular median: {np.median(leading):.12g}")
    print(f"leading singular max:    {max(leading):.12g}")


if __name__ == "__main__":
    main()
