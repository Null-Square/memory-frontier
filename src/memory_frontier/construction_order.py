from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .core import UnifilarSource, source_stationary_distribution


PolynomialOperator = dict[tuple[int, ...], np.ndarray]


def _validate_family(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_directions: np.ndarray,
    readout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(transition_base, dtype=float)
    directions = np.asarray(transition_directions, dtype=float)
    decoder = np.asarray(readout, dtype=float)

    if base.ndim != 3:
        raise ValueError("transition_base must have shape (k, alphabet, k)")
    k, alphabet_size, k2 = base.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition_base shape mismatch")
    if np.any(base < 0.0) or not np.allclose(
        base.sum(axis=-1), 1.0, atol=1e-12
    ):
        raise ValueError("transition_base rows must be probability distributions")

    if directions.ndim != 4 or directions.shape[0] == 0:
        raise ValueError(
            "transition_directions must have shape (n_parameters, k, alphabet, k)"
        )
    if directions.shape[1:] != base.shape:
        raise ValueError("transition_directions shape mismatch")
    if not np.allclose(directions.sum(axis=-1), 0.0, atol=1e-12):
        raise ValueError("every transition direction row must sum to zero")

    if decoder.shape != (k, alphabet_size):
        raise ValueError("readout shape mismatch")
    if np.any(decoder <= 0.0) or not np.allclose(
        decoder.sum(axis=-1), 1.0, atol=1e-12
    ):
        raise ValueError("readout rows must be strictly positive distributions")
    return base, directions, decoder


def _readout_classes(
    readout: np.ndarray,
    *,
    atol: float,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    if atol < 0.0:
        raise ValueError("readout_atol must be non-negative")

    classes: list[list[int]] = []
    representatives: list[np.ndarray] = []
    assignment = np.empty(readout.shape[0], dtype=int)
    for memory, row in enumerate(readout):
        matched = None
        for index, representative in enumerate(representatives):
            if np.allclose(row, representative, rtol=0.0, atol=atol):
                matched = index
                break
        if matched is None:
            matched = len(classes)
            classes.append([])
            representatives.append(row.copy())
        classes[matched].append(memory)
        assignment[memory] = matched

    return (
        tuple(tuple(group) for group in classes),
        np.asarray(representatives, dtype=float),
        assignment,
    )


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
                successor = int(source.transitions[source_state, symbol])
                output[successor] += (
                    weight * emission * controller_tensor[memory_state, symbol]
                )
    return output


@dataclass(frozen=True)
class ConstructionOrderOperator:
    """Exact transition-construction operator after quotienting equal readouts.

    For the affine transition family

        P(eps) = P0 + sum_j eps_j Pj,

    the finite-horizon loss is a polynomial. For every exponent ``alpha`` this
    object stores a matrix ``G_alpha`` over readout classes and output symbols
    such that

        [eps**alpha] L = - <G_alpha, log Q_class>.

    Equal decoder rows are quotiented before ``G_alpha`` is stored. Therefore
    the first nonzero operator degree is an exact source/transition construction
    order for the specified readout-equivalence pattern, while the actual loss
    order can only be equal or larger because of decoder-value cancellation.
    """

    coefficients: Mapping[tuple[int, ...], np.ndarray]
    readout_classes: tuple[tuple[int, ...], ...]
    class_readout: np.ndarray
    horizon: int

    def operator_order(self, *, atol: float = 1e-12) -> int | None:
        if atol < 0.0:
            raise ValueError("atol must be non-negative")
        degrees = [
            sum(exponent)
            for exponent, matrix in self.coefficients.items()
            if sum(exponent) > 0 and np.linalg.norm(matrix) > atol
        ]
        return min(degrees) if degrees else None

    def reconstructed_loss_coefficients(self) -> dict[tuple[int, ...], float]:
        log_readout = np.log(self.class_readout)
        return {
            exponent: -float(np.sum(matrix * log_readout))
            for exponent, matrix in self.coefficients.items()
        }

    def loss_order(self, *, atol: float = 1e-12) -> int | None:
        if atol < 0.0:
            raise ValueError("atol must be non-negative")
        coefficients = self.reconstructed_loss_coefficients()
        degrees = [
            sum(exponent)
            for exponent, coefficient in coefficients.items()
            if sum(exponent) > 0 and abs(coefficient) > atol
        ]
        return min(degrees) if degrees else None

    def degree_matrix(
        self,
        degree: int,
        *,
        atol: float = 0.0,
    ) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
        """Stack flattened operator rows for one total degree."""
        target = int(degree)
        if target < 0:
            raise ValueError("degree must be non-negative")
        if atol < 0.0:
            raise ValueError("atol must be non-negative")
        selected = [
            (exponent, matrix)
            for exponent, matrix in self.coefficients.items()
            if sum(exponent) == target and np.linalg.norm(matrix) > atol
        ]
        selected.sort(key=lambda item: item[0])
        if not selected:
            width = self.class_readout.size
            return tuple(), np.empty((0, width), dtype=float)
        return (
            tuple(exponent for exponent, _ in selected),
            np.stack([matrix.ravel() for _, matrix in selected], axis=0),
        )

    def decoder_cancellation_residual(
        self,
        degree: int,
        *,
        atol: float = 0.0,
    ) -> float:
        """Norm of degree-``degree`` loss coefficients before thresholding.

        A zero residual means the decoder log-vector lies in the nullspace of the
        construction operator at that degree, even if the operator itself is
        nonzero.
        """
        _, matrix = self.degree_matrix(degree, atol=atol)
        if matrix.shape[0] == 0:
            return 0.0
        return float(np.linalg.norm(matrix @ np.log(self.class_readout).ravel()))


def construction_order_operator(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_directions: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    max_total_degree: int | None = None,
    readout_atol: float = 1e-12,
) -> ConstructionOrderOperator:
    """Construct the exact quotient construction operator.

    The operator is obtained by expanding the source-memory occupancy polynomial
    and then aggregating memories whose decoder rows are equal. This makes the
    separation exact:

    * source + transition geometry determine the operator coefficients;
    * the equality pattern of decoder rows determines the quotient;
    * decoder values determine whether the first nonzero operator degree is
      visible in the scalar loss or cancelled.
    """
    base, directions, decoder = _validate_family(
        source, transition_base, transition_directions, readout
    )
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

    classes, class_readout, assignment = _readout_classes(
        decoder, atol=readout_atol
    )
    n_parameters = directions.shape[0]
    zero_exponent = (0,) * n_parameters
    initial = np.zeros((source.n_states, k), dtype=float)
    initial[:, initial_memory] = source_stationary_distribution(source)
    distribution: dict[tuple[int, ...], np.ndarray] = {zero_exponent: initial}
    coefficients: PolynomialOperator = {}

    for time in range(T):
        for exponent, occupancy in distribution.items():
            matrix = coefficients.setdefault(
                exponent,
                np.zeros((len(classes), source.alphabet_size), dtype=float),
            )
            for memory in range(k):
                class_index = int(assignment[memory])
                matrix[class_index] += (
                    np.einsum("s,sx->x", occupancy[:, memory], source.emissions)
                    / T
                )
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

    return ConstructionOrderOperator(
        coefficients=coefficients,
        readout_classes=classes,
        class_readout=class_readout,
        horizon=T,
    )


def minimum_readout_class_construction_cost(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_directions: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    support_atol: float = 1e-14,
    readout_atol: float = 1e-12,
) -> int | None:
    """Minimum perturbative-edge count to reach a different readout class.

    Base-support transitions have cost zero. Every nonzero entry of any
    transition direction has cost one. The search is performed on the exact
    source-memory product graph for at most ``horizon-1`` transitions, so token
    histories that the source cannot emit are excluded.

    This is a combinatorial support lower bound. Signed path cancellation can
    make the quotient construction operator begin at a higher degree, and
    decoder-value cancellation can raise the scalar loss order further.
    """
    base, directions, decoder = _validate_family(
        source, transition_base, transition_directions, readout
    )
    T = int(horizon)
    if T <= 0:
        raise ValueError("horizon must be positive")
    k = base.shape[0]
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")
    if support_atol < 0.0:
        raise ValueError("support_atol must be non-negative")

    _, _, assignment = _readout_classes(decoder, atol=readout_atol)
    initial_class = int(assignment[initial_memory])
    targets = {
        memory for memory in range(k) if int(assignment[memory]) != initial_class
    }
    if not targets:
        return None

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
                    if np.any(
                        np.abs(directions[:, memory_state, symbol, target])
                        > support_atol
                    ):
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


def construction_order_sandwich(
    source: UnifilarSource,
    transition_base: np.ndarray,
    transition_directions: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    max_total_degree: int | None = None,
    coefficient_atol: float = 1e-12,
    support_atol: float = 1e-14,
    readout_atol: float = 1e-12,
) -> tuple[int | None, int | None, int | None]:
    """Return ``(support_cost, operator_order, scalar_loss_order)``.

    Whenever the support cost is positive and finite, the exact hierarchy is

        support_cost <= operator_order <= scalar_loss_order,

    with ``None`` denoting absence of a nonconstant effect within the horizon.
    """
    support = minimum_readout_class_construction_cost(
        source,
        transition_base,
        transition_directions,
        readout,
        horizon,
        initial_memory=initial_memory,
        support_atol=support_atol,
        readout_atol=readout_atol,
    )
    operator = construction_order_operator(
        source,
        transition_base,
        transition_directions,
        readout,
        horizon,
        initial_memory=initial_memory,
        max_total_degree=max_total_degree,
        readout_atol=readout_atol,
    )
    return (
        support,
        operator.operator_order(atol=coefficient_atol),
        operator.loss_order(atol=coefficient_atol),
    )
