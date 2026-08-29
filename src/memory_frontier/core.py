from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class UnifilarSource:
    """Finite-state unifilar stochastic source.

    ``emissions[s, x]`` is P(X_{t+1}=x | S_t=s).
    ``transitions[s, x]`` is the deterministic successor state after x.
    """

    emissions: np.ndarray
    transitions: np.ndarray

    def __post_init__(self) -> None:
        emissions = np.asarray(self.emissions, dtype=float)
        transitions = np.asarray(self.transitions, dtype=int)
        if emissions.ndim != 2:
            raise ValueError("emissions must have shape (n_states, alphabet_size)")
        if transitions.shape != emissions.shape:
            raise ValueError("transitions must have the same shape as emissions")
        if np.any(emissions < 0):
            raise ValueError("emission probabilities must be non-negative")
        if not np.allclose(emissions.sum(axis=1), 1.0, atol=1e-12):
            raise ValueError("each state's emission probabilities must sum to 1")
        n_states = emissions.shape[0]
        if np.any((transitions < 0) | (transitions >= n_states)):
            raise ValueError("transition targets must be valid state indices")
        object.__setattr__(self, "emissions", emissions)
        object.__setattr__(self, "transitions", transitions)

    @property
    def n_states(self) -> int:
        return self.emissions.shape[0]

    @property
    def alphabet_size(self) -> int:
        return self.emissions.shape[1]

    def state_transition_matrix(self) -> np.ndarray:
        p = np.zeros((self.n_states, self.n_states), dtype=float)
        for s in range(self.n_states):
            for x in range(self.alphabet_size):
                p[s, self.transitions[s, x]] += self.emissions[s, x]
        return p


def _tarjan_scc(adjacency: list[list[int]]) -> list[list[int]]:
    n = len(adjacency)
    index = 0
    stack: list[int] = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlink = [0] * n
    components: list[list[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        indices[v] = lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adjacency[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)
    return components


def _stationary_on_irreducible_class(p: np.ndarray) -> np.ndarray:
    """Stationary distribution for an irreducible row-stochastic matrix."""
    n = p.shape[0]
    a = p.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(a, b)
    pi[np.abs(pi) < 1e-15] = 0.0
    pi = np.maximum(pi, 0.0)
    return pi / pi.sum()


def cesaro_limit_distribution(
    p: np.ndarray,
    initial: np.ndarray,
    *,
    support_tol: float = 1e-15,
) -> np.ndarray:
    """Long-run Cesaro occupancy for a finite Markov chain.

    This is well-defined even when ``p`` has multiple recurrent classes or is
    periodic. ``initial`` is a row distribution over states.
    """
    p = np.asarray(p, dtype=float)
    initial = np.asarray(initial, dtype=float)
    n = p.shape[0]
    if p.shape != (n, n):
        raise ValueError("p must be square")
    if initial.shape != (n,):
        raise ValueError("initial must have shape (n_states,)")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("p must be row-stochastic")
    if not np.isclose(initial.sum(), 1.0, atol=1e-12):
        raise ValueError("initial must sum to 1")

    adjacency = [list(np.flatnonzero(p[i] > support_tol)) for i in range(n)]
    sccs = _tarjan_scc(adjacency)
    recurrent: list[list[int]] = []
    recurrent_nodes: set[int] = set()
    for comp in sccs:
        cset = set(comp)
        closed = all(all(j in cset for j in adjacency[i]) for i in comp)
        if closed:
            recurrent.append(sorted(comp))
            recurrent_nodes.update(comp)

    transient = [i for i in range(n) if i not in recurrent_nodes]
    out = np.zeros(n, dtype=float)

    class_weights = np.array([initial[c].sum() for c in recurrent], dtype=float)

    if transient:
        q = p[np.ix_(transient, transient)]
        eye_minus_q = np.eye(len(transient)) - q
        init_t = initial[transient]
        for j, comp in enumerate(recurrent):
            r = p[np.ix_(transient, comp)].sum(axis=1)
            h = np.linalg.solve(eye_minus_q, r)
            class_weights[j] += float(init_t @ h)

    for weight, comp in zip(class_weights, recurrent):
        if weight <= support_tol:
            continue
        sub = p[np.ix_(comp, comp)]
        pi_sub = _stationary_on_irreducible_class(sub)
        out[np.asarray(comp)] += weight * pi_sub

    out[np.abs(out) < 1e-14] = 0.0
    if out.sum() <= 0:
        raise RuntimeError("failed to find recurrent occupancy")
    return out / out.sum()


def source_stationary_distribution(source: UnifilarSource) -> np.ndarray:
    """Cesaro stationary occupancy of the source from a uniform start."""
    p = source.state_transition_matrix()
    initial = np.full(source.n_states, 1.0 / source.n_states)
    return cesaro_limit_distribution(p, initial)


def entropy_from_joint_symbol_mass(symbol_mass: np.ndarray) -> float:
    """Conditional symbol entropy from unnormalized rows p(m, x)."""
    symbol_mass = np.asarray(symbol_mass, dtype=float)
    total = 0.0
    for row in symbol_mass:
        mass = row.sum()
        if mass <= 0:
            continue
        total -= float(np.sum([v * log(v) for v in row if v > 0]))
        total += mass * log(mass)
    return total


def bayes_log_loss(source: UnifilarSource) -> float:
    pi = source_stationary_distribution(source)
    joint = pi[:, None] * source.emissions
    return entropy_from_joint_symbol_mass(joint)


def canonical_partitions(n_items: int, max_blocks: int) -> Iterator[tuple[int, ...]]:
    """Restricted-growth strings for set partitions using at most max_blocks."""
    if n_items <= 0:
        return
    labels = [0] * n_items

    def rec(i: int, current_max: int) -> Iterator[tuple[int, ...]]:
        if i == n_items:
            yield tuple(labels)
            return
        upper = min(current_max + 1, max_blocks - 1)
        for label in range(upper + 1):
            labels[i] = label
            yield from rec(i + 1, max(current_max, label))

    yield from rec(1, 0)


def partition_log_loss(source: UnifilarSource, assignment: tuple[int, ...]) -> float:
    if len(assignment) != source.n_states:
        raise ValueError("assignment length must equal number of source states")
    pi = source_stationary_distribution(source)
    k = max(assignment) + 1
    mass = np.zeros((k, source.alphabet_size), dtype=float)
    for s, m in enumerate(assignment):
        mass[m] += pi[s] * source.emissions[s]
    return entropy_from_joint_symbol_mass(mass)


def is_recursive_quotient(
    source: UnifilarSource,
    assignment: tuple[int, ...],
    *,
    support_tol: float = 1e-15,
) -> bool:
    """Whether a state partition can be updated from (memory, symbol) alone."""
    n_blocks = max(assignment) + 1
    for m in range(n_blocks):
        members = [s for s, label in enumerate(assignment) if label == m]
        for x in range(source.alphabet_size):
            targets = {
                assignment[source.transitions[s, x]]
                for s in members
                if source.emissions[s, x] > support_tol
            }
            if len(targets) > 1:
                return False
    return True


@dataclass(frozen=True)
class PartitionOptimum:
    loss: float
    assignment: tuple[int, ...]


def best_static_partition(source: UnifilarSource, k: int) -> PartitionOptimum:
    best = PartitionOptimum(float("inf"), ())
    for assignment in canonical_partitions(source.n_states, k):
        loss = partition_log_loss(source, assignment)
        if loss < best.loss - 1e-15:
            best = PartitionOptimum(loss, assignment)
    return best


def best_recursive_quotient(source: UnifilarSource, k: int) -> PartitionOptimum:
    best = PartitionOptimum(float("inf"), ())
    for assignment in canonical_partitions(source.n_states, k):
        if not is_recursive_quotient(source, assignment):
            continue
        loss = partition_log_loss(source, assignment)
        if loss < best.loss - 1e-15:
            best = PartitionOptimum(loss, assignment)
    return best


def product_transition_matrix(source: UnifilarSource, f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=int)
    k, alphabet_size = f.shape
    if alphabet_size != source.alphabet_size:
        raise ValueError("controller alphabet does not match source")
    if np.any((f < 0) | (f >= k)):
        raise ValueError("memory transitions must target valid memory states")
    n = source.n_states * k
    p = np.zeros((n, n), dtype=float)
    for s in range(source.n_states):
        for m in range(k):
            i = s * k + m
            for x in range(source.alphabet_size):
                s2 = source.transitions[s, x]
                m2 = f[m, x]
                j = s2 * k + m2
                p[i, j] += source.emissions[s, x]
    return p


def controller_occupancy(
    source: UnifilarSource,
    f: np.ndarray,
    initial_memory: int,
) -> np.ndarray:
    k = np.asarray(f).shape[0]
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory state")
    pi_source = source_stationary_distribution(source)
    initial = np.zeros(source.n_states * k, dtype=float)
    for s in range(source.n_states):
        initial[s * k + initial_memory] = pi_source[s]
    p = product_transition_matrix(source, f)
    return cesaro_limit_distribution(p, initial).reshape(source.n_states, k)


def controller_log_loss(
    source: UnifilarSource,
    f: np.ndarray,
    initial_memory: int = 0,
) -> float:
    occupancy = controller_occupancy(source, f, initial_memory)
    k = occupancy.shape[1]
    mass = np.zeros((k, source.alphabet_size), dtype=float)
    for s in range(source.n_states):
        for m in range(k):
            mass[m] += occupancy[s, m] * source.emissions[s]
    return entropy_from_joint_symbol_mass(mass)


@dataclass(frozen=True)
class ControllerOptimum:
    loss: float
    transition_table: np.ndarray
    initial_memory: int
    occupancy: np.ndarray


def enumerate_deterministic_controllers(k: int, alphabet_size: int) -> Iterator[np.ndarray]:
    for flat in product(range(k), repeat=k * alphabet_size):
        yield np.asarray(flat, dtype=int).reshape(k, alphabet_size)


def best_deterministic_controller(source: UnifilarSource, k: int) -> ControllerOptimum:
    best_loss = float("inf")
    best_f: np.ndarray | None = None
    best_m0 = 0
    best_occupancy: np.ndarray | None = None
    for f in enumerate_deterministic_controllers(k, source.alphabet_size):
        for m0 in range(k):
            occupancy = controller_occupancy(source, f, m0)
            mass = np.zeros((k, source.alphabet_size), dtype=float)
            for s in range(source.n_states):
                for m in range(k):
                    mass[m] += occupancy[s, m] * source.emissions[s]
            loss = entropy_from_joint_symbol_mass(mass)
            if loss < best_loss - 1e-15:
                best_loss = loss
                best_f = f.copy()
                best_m0 = m0
                best_occupancy = occupancy.copy()
    assert best_f is not None and best_occupancy is not None
    return ControllerOptimum(best_loss, best_f, best_m0, best_occupancy)


def alias_entropy(occupancy: np.ndarray) -> float:
    """H(M|S) under a stationary joint occupancy p(S,M)."""
    occupancy = np.asarray(occupancy, dtype=float)
    total = 0.0
    for row in occupancy:
        p_s = row.sum()
        if p_s <= 0:
            continue
        for joint in row:
            if joint > 0:
                total -= joint * log(joint / p_s)
    return total
