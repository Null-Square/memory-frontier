from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import UnifilarSource, source_stationary_distribution


def _transition_tensor(source: UnifilarSource, transition: np.ndarray) -> np.ndarray:
    tensor = np.asarray(transition, dtype=float)
    if tensor.ndim != 3:
        raise ValueError("transition must have shape (k, alphabet, k)")
    k, alphabet_size, k2 = tensor.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition shape mismatch")
    if (
        not np.all(np.isfinite(tensor))
        or np.any(tensor < 0.0)
        or not np.allclose(tensor.sum(axis=-1), 1.0, atol=1e-12)
    ):
        raise ValueError("transition rows must be probability distributions")
    return tensor


@dataclass(frozen=True)
class ForwardSupport:
    """Exact finite-horizon support of the source-memory forward process.

    ``reachable_product`` has shape ``(horizon, n_source_states, k)`` and records
    which source-memory pairs can occur with positive support under the reference
    controller. ``active_transition_rows[m, x]`` is true exactly when row
    ``(m, x)`` can be exercised during one of the ``horizon-1`` transition steps.
    """

    reachable_product: np.ndarray
    active_transition_rows: np.ndarray

    @property
    def dormant_transition_rows(self) -> np.ndarray:
        return ~self.active_transition_rows


def finite_horizon_forward_support(
    source: UnifilarSource,
    transition: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    support_atol: float = 1e-14,
) -> ForwardSupport:
    """Compute source-aware forward support for a stochastic memory controller."""
    tensor = _transition_tensor(source, transition)
    T = int(horizon)
    if T <= 0:
        raise ValueError("horizon must be positive")
    if support_atol < 0.0:
        raise ValueError("support_atol must be non-negative")
    k = tensor.shape[0]
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    reachable = np.zeros((T, source.n_states, k), dtype=bool)
    stationary = source_stationary_distribution(source)
    reachable[0, stationary > support_atol, initial_memory] = True
    active_rows = np.zeros((k, source.alphabet_size), dtype=bool)

    for time in range(T - 1):
        current = reachable[time]
        next_support = reachable[time + 1]
        for source_state in range(source.n_states):
            memories = np.flatnonzero(current[source_state])
            if memories.size == 0:
                continue
            for symbol in range(source.alphabet_size):
                if source.emissions[source_state, symbol] <= support_atol:
                    continue
                successor = int(source.transitions[source_state, symbol])
                for memory in memories:
                    memory = int(memory)
                    active_rows[memory, symbol] = True
                    targets = np.flatnonzero(
                        tensor[memory, symbol] > support_atol
                    )
                    next_support[successor, targets] = True

    return ForwardSupport(
        reachable_product=reachable,
        active_transition_rows=active_rows,
    )


def is_source_horizon_dormant_rewire(
    source: UnifilarSource,
    reference_transition: np.ndarray,
    candidate_transition: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    support_atol: float = 1e-14,
    difference_atol: float = 1e-12,
) -> bool:
    """Certify a transition rewire as behaviorally dormant over the horizon.

    The certificate is sufficient and exact: the candidate must agree with the
    reference controller on every transition row that the reference forward
    process can exercise. It may differ arbitrarily on all other rows.

    By induction on time, the candidate then has the same complete source-memory
    occupancy as the reference controller through the requested horizon, and
    hence the same finite-horizon prediction loss for *any* fixed decoder.

    The certificate is source-aware: even a row of a reachable memory state can
    be dormant when its symbol is impossible on every source state paired with
    that memory during the horizon.
    """
    reference = _transition_tensor(source, reference_transition)
    candidate = _transition_tensor(source, candidate_transition)
    if candidate.shape != reference.shape:
        raise ValueError("candidate transition shape mismatch")
    if difference_atol < 0.0:
        raise ValueError("difference_atol must be non-negative")

    support = finite_horizon_forward_support(
        source,
        reference,
        horizon,
        initial_memory=initial_memory,
        support_atol=support_atol,
    )
    changed_rows = np.max(np.abs(candidate - reference), axis=-1) > difference_atol
    return not bool(np.any(changed_rows & support.active_transition_rows))
