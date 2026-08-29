from __future__ import annotations

from itertools import product

import numpy as np

from .core import UnifilarSource
from .families import symmetric_repeat_entropy_rate


def delayed_repeat_source(
    alphabet_size: int,
    delay: int,
    switch_probability: float,
) -> UnifilarSource:
    """Higher-order delayed-copy source with exactly known predictive memory.

    The source state is the last ``delay`` symbols, oldest first. The next symbol
    repeats the oldest stored symbol with probability ``1-rho`` and otherwise
    switches uniformly to another symbol. After emission, the state shifts left
    and appends the new symbol.

    For ``delay>=2`` the stationary adjacent-token conditional is uniform even
    though the full source is strongly predictable from deeper history.
    """
    q = int(alphabet_size)
    r = int(delay)
    rho = float(switch_probability)
    if q < 2:
        raise ValueError("alphabet_size must be at least 2")
    if r < 1:
        raise ValueError("delay must be positive")
    if not 0.0 < rho < 1.0:
        raise ValueError("switch_probability must lie strictly between 0 and 1")

    states = tuple(product(range(q), repeat=r))
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    emissions = np.full((n, q), rho / (q - 1), dtype=float)
    transitions = np.empty((n, q), dtype=int)

    for i, state in enumerate(states):
        oldest = state[0]
        emissions[i, oldest] = 1.0 - rho
        for symbol in range(q):
            successor = state[1:] + (symbol,) if r > 1 else (symbol,)
            transitions[i, symbol] = index[successor]

    return UnifilarSource(emissions, transitions)


def delayed_shift_register_controller(alphabet_size: int, delay: int) -> np.ndarray:
    """Exact q^delay-state shift-register update table."""
    # Transition dynamics do not depend on the emission noise, so any valid rho
    # yields the same table.
    return delayed_repeat_source(alphabet_size, delay, 0.5).transitions.copy()


def delayed_repeat_entropy_rate(
    alphabet_size: int,
    switch_probability: float,
) -> float:
    """Bayes next-token NLL of the delayed-repeat family."""
    return symmetric_repeat_entropy_rate(alphabet_size, switch_probability)
