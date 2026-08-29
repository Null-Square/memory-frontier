from __future__ import annotations

import numpy as np

from memory_frontier import (
    binary_dormant_chain_readout_logits,
    delayed_repeat_source,
    dormant_chain_controller,
    exact_ste_gradient,
)


def canonical_logits(table: np.ndarray, margin: float) -> np.ndarray:
    table = np.asarray(table, dtype=int)
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def run_transition_only_sgd(
    source,
    table: np.ndarray,
    readout_logits: np.ndarray,
    *,
    horizon: int,
    margin: float,
    temperature: float,
    learning_rate: float,
    steps: int,
) -> list[tuple[int, tuple[int, ...], float]]:
    logits = canonical_logits(table, margin)
    trajectory: list[tuple[int, tuple[int, ...], float]] = []
    previous: tuple[int, ...] | None = None
    for step in range(steps + 1):
        gradient = exact_ste_gradient(
            source,
            logits,
            readout_logits,
            horizon,
            temperature=temperature,
        )
        signature = tuple(map(int, gradient.transition_table.ravel()))
        if signature != previous:
            trajectory.append((step, signature, gradient.loss))
            previous = signature
        if step < steps:
            logits -= learning_rate * gradient.transition_logit_gradient
    return trajectory


def main() -> None:
    delay = 3
    source = delayed_repeat_source(2, delay, 0.1)
    blind = np.zeros((delay + 1, 2), dtype=int)
    scaffold = dormant_chain_controller(delay, 2)
    readout_logits = binary_dormant_chain_readout_logits(delay, 0.2)

    common = dict(
        horizon=20,
        margin=0.7,
        temperature=0.8,
        learning_rate=5.0,
        steps=80,
    )
    for label, table in (("blind", blind), ("scaffold", scaffold)):
        print(label)
        for step, signature, loss in run_transition_only_sgd(
            source, table, readout_logits, **common
        ):
            print(f"  step={step:3d} loss={loss:.12f} table={signature}")


if __name__ == "__main__":
    main()
