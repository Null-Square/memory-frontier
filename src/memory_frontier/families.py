from __future__ import annotations

from math import log

import numpy as np

from .core import UnifilarSource


def symmetric_repeat_source(
    alphabet_size: int,
    switch_probability: float,
) -> UnifilarSource:
    """Symmetric q-symbol Markov source with one-step observable state.

    The source state is the previous symbol. From state ``s``, the next symbol
    repeats ``s`` with probability ``1-rho`` and switches uniformly to one of
    the other ``q-1`` symbols with total probability ``rho``. After emitting
    symbol ``x``, the next source state is exactly ``x``.

    The stationary source-state and token marginals are uniform. A q-state
    last-symbol controller reaches the Bayes entropy rate asymptotically.
    """
    q = int(alphabet_size)
    rho = float(switch_probability)
    if q < 2:
        raise ValueError("alphabet_size must be at least 2")
    if not 0.0 < rho < 1.0:
        raise ValueError("switch_probability must lie strictly between 0 and 1")

    emissions = np.full((q, q), rho / (q - 1), dtype=float)
    np.fill_diagonal(emissions, 1.0 - rho)
    transitions = np.tile(np.arange(q, dtype=int), (q, 1))
    return UnifilarSource(emissions, transitions)


def symmetric_repeat_entropy_rate(
    alphabet_size: int,
    switch_probability: float,
) -> float:
    """Bayes next-token NLL / entropy rate of ``symmetric_repeat_source``."""
    q = int(alphabet_size)
    rho = float(switch_probability)
    if q < 2:
        raise ValueError("alphabet_size must be at least 2")
    if not 0.0 < rho < 1.0:
        raise ValueError("switch_probability must lie strictly between 0 and 1")
    return -(1.0 - rho) * log(1.0 - rho) - rho * log(rho / (q - 1))


def last_symbol_controller(alphabet_size: int) -> np.ndarray:
    """q-state controller F(m, x)=x that stores the most recent symbol."""
    q = int(alphabet_size)
    if q < 2:
        raise ValueError("alphabet_size must be at least 2")
    return np.tile(np.arange(q, dtype=int), (q, 1))


def symmetric_repeat_last_symbol_finite_horizon_loss(
    alphabet_size: int,
    switch_probability: float,
    horizon: int,
) -> float:
    """Exact reset-sequence NLL of the last-symbol controller.

    Memory starts in state 0 while the source starts in stationarity. At t=0
    that reset memory contains no source information. From t>=1, memory equals
    the previous emitted symbol exactly. Because memory label 0 is used both by
    the reset and by genuine symbol 0, its single shared readout is a finite-T
    mixture; this formula includes that transient correction.
    """
    q = int(alphabet_size)
    rho = float(switch_probability)
    T = int(horizon)
    if q < 2:
        raise ValueError("alphabet_size must be at least 2")
    if not 0.0 < rho < 1.0:
        raise ValueError("switch_probability must lie strictly between 0 and 1")
    if T <= 0:
        raise ValueError("horizon must be positive")

    row = np.full(q, rho / (q - 1), dtype=float)
    row[0] = 1.0 - rho
    row_entropy = -float(np.sum(row * np.log(row)))

    reset_mixture = (1.0 + (T - 1) * row) / (q + T - 1)
    reset_entropy = -float(np.sum(reset_mixture * np.log(reset_mixture)))

    reset_mass = (q + T - 1) / (T * q)
    other_mass = (q - 1) * (T - 1) / (T * q)
    return reset_mass * reset_entropy + other_mass * row_entropy
