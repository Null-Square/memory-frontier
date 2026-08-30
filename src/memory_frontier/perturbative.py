from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .core import UnifilarSource, source_stationary_distribution


def _validate_affine_family(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_direction: np.ndarray,
    readout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(transition_base, dtype=float)
    direction = np.asarray(transition_direction, dtype=float)
    if base.ndim != 3:
        raise ValueError("transition_base must have shape (k, alphabet, k)")
    k, alphabet_size, k2 = base.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition_base shape mismatch")
    if direction.shape != base.shape:
        raise ValueError("transition_direction must match transition_base")
    if np.any(base < 0.0) or not np.allclose(
        base.sum(axis=-1), 1.0, atol=1e-12
    ):
        raise ValueError("transition_base rows must be probability distributions")
    if not np.allclose(direction.sum(axis=-1), 0.0, atol=1e-12):
        raise ValueError("transition_direction rows must sum to zero")

    decoder = np.asarray(readout, dtype=float)
    if decoder.shape != (k, alphabet_size):
        raise ValueError("readout shape mismatch")
    if np.any(decoder <= 0.0) or not np.allclose(
        decoder.sum(axis=-1), 1.0, atol=1e-12
    ):
        raise ValueError("readout rows must be strictly positive distributions")
    return base, direction, decoder


def affine_controller_loss_coefficients(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_direction: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
) -> np.ndarray:
    """Exact finite-horizon coefficients for ``P(eps)=P0+eps*P1``.

    With fixed readout and affine transition probabilities, the finite-horizon
    expected log loss is a polynomial of degree at most ``horizon-1``. The
    returned array ``c`` satisfies

        L(eps) = sum_d c[d] * eps**d

    exactly up to floating-point arithmetic.
    """
    base, direction, decoder = _validate_affine_family(
        source, transition_base, transition_direction, readout
    )
    T = int(horizon)
    if T <= 0:
        raise ValueError("horizon must be positive")
    k = base.shape[0]
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    n_states = source.n_states
    max_degree = T - 1
    distribution = np.zeros((max_degree + 1, n_states, k), dtype=float)
    distribution[0, :, initial_memory] = source_stationary_distribution(source)
    coefficients = np.zeros(max_degree + 1, dtype=float)
    log_readout = np.log(decoder)

    for time in range(T):
        coefficients -= np.einsum(
            "dsm,sx,mx->d", distribution, source.emissions, log_readout
        )
        if time == T - 1:
            break

        next_distribution = np.zeros_like(distribution)
        for degree in range(max_degree + 1):
            for source_state in range(n_states):
                for memory_state in range(k):
                    weight = distribution[degree, source_state, memory_state]
                    if weight == 0.0:
                        continue
                    for symbol in range(source.alphabet_size):
                        emission = source.emissions[source_state, symbol]
                        if emission == 0.0:
                            continue
                        successor = source.transitions[source_state, symbol]
                        next_distribution[degree, successor] += (
                            weight * emission * base[memory_state, symbol]
                        )
                        if degree < max_degree:
                            next_distribution[degree + 1, successor] += (
                                weight * emission * direction[memory_state, symbol]
                            )
        distribution = next_distribution

    return coefficients / T


def leading_perturbative_order(
    loss_coefficients: np.ndarray,
    *,
    atol: float = 1e-12,
) -> int | None:
    """First nonconstant power with nonzero loss coefficient."""
    coefficients = np.asarray(loss_coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) == 0:
        raise ValueError("loss_coefficients must be a non-empty vector")
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    for degree in range(1, len(coefficients)):
        if abs(float(coefficients[degree])) > atol:
            return degree
    return None


def minimum_decoder_construction_cost(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_direction: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    target_memories: Iterable[int] | None = None,
    support_atol: float = 1e-14,
    readout_atol: float = 1e-12,
) -> int | None:
    """Minimum number of positive perturbative links needed to reach a target.

    Base transition support has cost zero. Positive entries of the affine
    direction are candidate new-routing links and have cost one. The search is
    performed on the source-memory product graph for at most ``horizon-1``
    transitions, so impossible token histories are excluded.

    When ``target_memories`` is omitted, every memory whose decoder differs from
    the initial-memory decoder is treated as a target.
    """
    base, direction, decoder = _validate_affine_family(
        source, transition_base, transition_direction, readout
    )
    T = int(horizon)
    if T <= 0:
        raise ValueError("horizon must be positive")
    k = base.shape[0]
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")
    if support_atol < 0.0 or readout_atol < 0.0:
        raise ValueError("tolerances must be non-negative")

    if target_memories is None:
        reference = decoder[initial_memory]
        targets = {
            memory
            for memory in range(k)
            if not np.allclose(
                decoder[memory], reference, rtol=0.0, atol=readout_atol
            )
        }
    else:
        targets = {int(memory) for memory in target_memories}
        if any(memory < 0 or memory >= k for memory in targets):
            raise ValueError("target memory is out of range")
    if not targets:
        return None
    if initial_memory in targets:
        return 0

    stationary = source_stationary_distribution(source)
    current = {
        (source_state, initial_memory): 0
        for source_state in range(source.n_states)
        if stationary[source_state] > support_atol
    }
    best: int | None = None

    for _ in range(T - 1):
        next_cost: dict[tuple[int, int], int] = {}
        for (source_state, memory_state), cost in current.items():
            for symbol in range(source.alphabet_size):
                if source.emissions[source_state, symbol] <= support_atol:
                    continue
                successor = int(source.transitions[source_state, symbol])
                for target in range(k):
                    if base[memory_state, symbol, target] > support_atol:
                        key = (successor, target)
                        old = next_cost.get(key)
                        if old is None or cost < old:
                            next_cost[key] = cost
                    if direction[memory_state, symbol, target] > support_atol:
                        key = (successor, target)
                        candidate = cost + 1
                        old = next_cost.get(key)
                        if old is None or candidate < old:
                            next_cost[key] = candidate
        if not next_cost:
            break
        for (_, memory_state), cost in next_cost.items():
            if memory_state in targets and (best is None or cost < best):
                best = cost
        current = next_cost
    return best


def multivariate_controller_loss_coefficients(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_directions: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
    *,
    max_total_degree: int | None = None,
) -> dict[tuple[int, ...], float]:
    """Exact sparse loss polynomial for independent transition perturbations.

    The transition family is

        P(eps) = P0 + sum_j eps[j] * Pj.

    Keys are exponent tuples ``alpha`` and values are coefficients of
    ``prod_j eps[j]**alpha[j]``. When ``max_total_degree`` is supplied, terms of
    higher total degree are omitted exactly from the returned truncation.
    """
    directions = np.asarray(transition_directions, dtype=float)
    if directions.ndim != 4 or directions.shape[0] == 0:
        raise ValueError(
            "transition_directions must have shape (n_parameters, k, alphabet, k)"
        )
    base, _, decoder = _validate_affine_family(
        source, transition_base, directions[0], readout
    )
    if directions.shape[1:] != base.shape:
        raise ValueError("transition_directions shape mismatch")
    if not np.allclose(directions.sum(axis=-1), 0.0, atol=1e-12):
        raise ValueError("every transition direction row must sum to zero")

    T = int(horizon)
    if T <= 0:
        raise ValueError("horizon must be positive")
    k = base.shape[0]
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")
    degree_limit = T - 1 if max_total_degree is None else int(max_total_degree)
    if degree_limit < 0:
        raise ValueError("max_total_degree must be non-negative")
    degree_limit = min(degree_limit, T - 1)

    n_parameters = directions.shape[0]
    zero_exponent = (0,) * n_parameters
    initial = np.zeros((source.n_states, k), dtype=float)
    initial[:, initial_memory] = source_stationary_distribution(source)
    distribution: dict[tuple[int, ...], np.ndarray] = {zero_exponent: initial}
    coefficients: dict[tuple[int, ...], float] = {}
    log_readout = np.log(decoder)

    for time in range(T):
        for exponent, occupancy in distribution.items():
            value = -float(
                np.einsum(
                    "sm,sx,mx->", occupancy, source.emissions, log_readout
                )
            ) / T
            coefficients[exponent] = coefficients.get(exponent, 0.0) + value
        if time == T - 1:
            break

        next_distribution: dict[tuple[int, ...], np.ndarray] = {}
        for exponent, occupancy in distribution.items():
            base_out = _propagate_product_occupancy(source, occupancy, base)
            if np.any(base_out):
                if exponent in next_distribution:
                    next_distribution[exponent] += base_out
                else:
                    next_distribution[exponent] = base_out

            if sum(exponent) >= degree_limit:
                continue
            for parameter, direction in enumerate(directions):
                direction_out = _propagate_product_occupancy(
                    source, occupancy, direction
                )
                if not np.any(direction_out):
                    continue
                new_exponent = list(exponent)
                new_exponent[parameter] += 1
                key = tuple(new_exponent)
                if sum(key) > degree_limit:
                    continue
                if key in next_distribution:
                    next_distribution[key] += direction_out
                else:
                    next_distribution[key] = direction_out
        distribution = next_distribution

    return coefficients


def _propagate_product_occupancy(
    source: UnifilarSource,
    occupancy: np.ndarray,
    controller_tensor: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(occupancy)
    for source_state in range(source.n_states):
        for memory_state in range(occupancy.shape[1]):
            weight = occupancy[source_state, memory_state]
            if weight == 0.0:
                continue
            for symbol in range(source.alphabet_size):
                emission = source.emissions[source_state, symbol]
                if emission == 0.0:
                    continue
                successor = source.transitions[source_state, symbol]
                output[successor] += (
                    weight * emission * controller_tensor[memory_state, symbol]
                )
    return output


def leading_multivariate_total_degree(
    loss_coefficients: dict[tuple[int, ...], float],
    *,
    atol: float = 1e-12,
) -> int | None:
    """Smallest positive total degree with a nonzero multivariate coefficient."""
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    degrees = [
        sum(exponent)
        for exponent, coefficient in loss_coefficients.items()
        if sum(exponent) > 0 and abs(float(coefficient)) > atol
    ]
    return min(degrees) if degrees else None
