from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
import json
from math import inf

import numpy as np

from .core import (
    UnifilarSource,
    alias_entropy,
    bayes_log_loss,
    best_recursive_quotient,
    best_static_partition,
    controller_occupancy,
    entropy_from_joint_symbol_mass,
    enumerate_deterministic_controllers,
    source_stationary_distribution,
)
from .finite import controller_average_occupancy


def is_strongly_connected(source: UnifilarSource, *, support_tol: float = 1e-15) -> bool:
    """Whether the positive-probability source-state graph is strongly connected."""
    n = source.n_states
    adjacency = [
        {source.transitions[s, x] for x in range(source.alphabet_size) if source.emissions[s, x] > support_tol}
        for s in range(n)
    ]
    reverse = [set() for _ in range(n)]
    for s, targets in enumerate(adjacency):
        for t in targets:
            reverse[t].add(s)

    def reach(graph: list[set[int]], start: int) -> set[int]:
        seen = {start}
        stack = [start]
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    return len(reach(adjacency, 0)) == n and len(reach(reverse, 0)) == n


def find_synchronizing_word(source: UnifilarSource) -> tuple[int, ...] | None:
    """Shortest reset word for a complete deterministic source transition automaton.

    This routine assumes every symbol is possible from every source state. Under
    that condition a returned word is an observable token sequence that sends all
    possible source states to one state. Breadth-first search over state subsets is
    exact and tiny for the small sources used in this project.
    """
    if np.any(source.emissions <= 0):
        raise ValueError("exact synchronization check currently requires strictly positive emissions")
    start = frozenset(range(source.n_states))
    if len(start) <= 1:
        return ()
    queue: list[tuple[frozenset[int], tuple[int, ...]]] = [(start, ())]
    seen = {start}
    head = 0
    while head < len(queue):
        states, word = queue[head]
        head += 1
        for x in range(source.alphabet_size):
            nxt = frozenset(source.transitions[s, x] for s in states)
            new_word = word + (x,)
            if len(nxt) == 1:
                return new_word
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, new_word))
    return None


def is_exactly_synchronizing(source: UnifilarSource) -> bool:
    return find_synchronizing_word(source) is not None


def _rows_distinct(emissions: np.ndarray, min_l1_distance: float) -> bool:
    for i in range(len(emissions)):
        for j in range(i):
            if float(np.abs(emissions[i] - emissions[j]).sum()) < min_l1_distance:
                return False
    return True


def random_synchronizing_source(
    n_states: int,
    alphabet_size: int = 2,
    *,
    rng: np.random.Generator | int | None = None,
    emission_floor: float = 0.05,
    min_emission_l1_distance: float = 0.05,
    max_attempts: int = 100_000,
) -> UnifilarSource:
    """Sample a small irreducible, exactly synchronizing unifilar source.

    Emissions are strictly positive and pairwise distinct, so source states are
    already distinguishable by their one-step predictive distributions. Exact
    synchronization ensures the source state is recoverable from token history
    after a finite reset word, avoiding hidden-state-estimation confounds.
    """
    if n_states < 1 or alphabet_size < 2:
        raise ValueError("need at least one state and an alphabet of size >= 2")
    if not 0 <= emission_floor < 1.0 / alphabet_size:
        raise ValueError("emission_floor must be in [0, 1/alphabet_size)")
    gen = np.random.default_rng(rng)
    for _ in range(max_attempts):
        raw = gen.dirichlet(np.ones(alphabet_size), size=n_states)
        emissions = emission_floor + (1.0 - alphabet_size * emission_floor) * raw
        if not _rows_distinct(emissions, min_emission_l1_distance):
            continue
        transitions = gen.integers(0, n_states, size=(n_states, alphabet_size))
        source = UnifilarSource(emissions, transitions)
        if not is_strongly_connected(source):
            continue
        if find_synchronizing_word(source) is None:
            continue
        return source
    raise RuntimeError("failed to sample a valid source within max_attempts")


def _loss_from_occupancy(source: UnifilarSource, occupancy: np.ndarray) -> float:
    k = occupancy.shape[1]
    mass = np.zeros((k, source.alphabet_size), dtype=float)
    for s in range(source.n_states):
        for m in range(k):
            mass[m] += occupancy[s, m] * source.emissions[s]
    return entropy_from_joint_symbol_mass(mass)


def relabel_controller(
    transition_table: np.ndarray,
    initial_memory: int,
    relabel: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    """Rename memory states by the bijection old_state -> relabel[old_state]."""
    f = np.asarray(transition_table, dtype=int)
    k = f.shape[0]
    if tuple(sorted(relabel)) != tuple(range(k)):
        raise ValueError("relabel must be a permutation of memory states")
    out = np.empty_like(f)
    for old in range(k):
        new = relabel[old]
        for x in range(f.shape[1]):
            out[new, x] = relabel[f[old, x]]
    return out, relabel[initial_memory]


def canonical_controller_signature(
    transition_table: np.ndarray,
    initial_memory: int,
) -> tuple[int, ...]:
    """Canonical (F,m0) signature modulo memory-state renaming only."""
    f = np.asarray(transition_table, dtype=int)
    k = f.shape[0]
    candidates: list[tuple[int, ...]] = []
    for perm in permutations(range(k)):
        renamed, m0 = relabel_controller(f, initial_memory, perm)
        candidates.append((m0, *map(int, renamed.ravel())))
    return min(candidates)


@dataclass(frozen=True)
class ControllerClass:
    signature: tuple[int, ...]
    loss: float
    transition_table: np.ndarray
    initial_memory: int
    occupancy: np.ndarray
    labeled_multiplicity: int

    @property
    def alias_entropy(self) -> float:
        return alias_entropy(self.occupancy)


@dataclass(frozen=True)
class ControllerSpectrum:
    k: int
    classes: tuple[ControllerClass, ...]
    labeled_count: int

    def best_loss(self) -> float:
        return min(c.loss for c in self.classes)

    @property
    def optimum(self) -> ControllerClass:
        """Canonical representative of the numerically tied optimal class set.

        Raw linear-algebra noise at ~1e-16 must not decide which discrete
        controller is reported. Classes within 1e-12 of the minimum are treated
        as scientifically tied; the lexicographically smallest canonical
        signature is the deterministic representative.
        """
        candidates = self.optimal_classes()
        return min(candidates, key=lambda c: c.signature)

    def optimal_classes(self, *, atol: float = 1e-12) -> tuple[ControllerClass, ...]:
        best = self.best_loss()
        return tuple(
            sorted(
                (c for c in self.classes if abs(c.loss - best) <= atol),
                key=lambda c: c.signature,
            )
        )

    def runner_up(self, *, atol: float = 1e-12) -> ControllerClass | None:
        best = self.best_loss()
        candidates = [c for c in self.classes if c.loss > best + atol]
        return min(candidates, key=lambda c: (c.loss, c.signature)) if candidates else None

    def relabel_class_gap(self, *, atol: float = 1e-12) -> float:
        runner = self.runner_up(atol=atol)
        return inf if runner is None else runner.loss - self.best_loss()

    def near_optimal_class_count(self, epsilon: float) -> int:
        best = self.best_loss()
        return sum(c.loss <= best + epsilon for c in self.classes)

    def distinct_loss_levels(self, *, atol: float = 1e-12) -> tuple[float, ...]:
        """Sorted loss levels, merging numerical ties across controller classes."""
        levels: list[float] = []
        for c in self.classes:
            if not levels or abs(c.loss - levels[-1]) > atol:
                levels.append(c.loss)
        return tuple(levels)

    def distinct_loss_gap(self, *, atol: float = 1e-12) -> float:
        levels = self.distinct_loss_levels(atol=atol)
        return inf if len(levels) < 2 else levels[1] - levels[0]


def deterministic_controller_spectrum(
    source: UnifilarSource,
    k: int,
    *,
    horizon: int | None = None,
) -> ControllerSpectrum:
    """Exhaustive controller spectrum modulo state renaming.

    ``horizon=None`` scores asymptotic Cesaro occupancy. A positive integer
    scores the exact reset-sequence objective over that many predictions.
    """
    if horizon is not None and horizon <= 0:
        raise ValueError("horizon must be positive")
    grouped: dict[tuple[int, ...], dict[str, object]] = {}
    labeled_count = 0
    for f in enumerate_deterministic_controllers(k, source.alphabet_size):
        for m0 in range(k):
            labeled_count += 1
            sig = canonical_controller_signature(f, m0)
            occupancy = (
                controller_occupancy(source, f, m0)
                if horizon is None
                else controller_average_occupancy(source, f, m0, horizon)
            )
            loss = _loss_from_occupancy(source, occupancy)
            entry = grouped.get(sig)
            if entry is None:
                grouped[sig] = {
                    "loss": loss,
                    "transition_table": f.copy(),
                    "initial_memory": m0,
                    "occupancy": occupancy.copy(),
                    "multiplicity": 1,
                }
            else:
                entry["multiplicity"] = int(entry["multiplicity"]) + 1
                if abs(loss - float(entry["loss"])) > 1e-11:
                    raise RuntimeError("memory-state relabeling changed controller loss")
    classes = tuple(
        sorted(
            (
                ControllerClass(
                    signature=sig,
                    loss=float(v["loss"]),
                    transition_table=np.asarray(v["transition_table"], dtype=int),
                    initial_memory=int(v["initial_memory"]),
                    occupancy=np.asarray(v["occupancy"], dtype=float),
                    labeled_multiplicity=int(v["multiplicity"]),
                )
                for sig, v in grouped.items()
            ),
            key=lambda c: (c.loss, c.signature),
        )
    )
    return ControllerSpectrum(k=k, classes=classes, labeled_count=labeled_count)


def source_fingerprint(source: UnifilarSource) -> str:
    payload = {
        "emissions": np.asarray(source.emissions, float).tolist(),
        "transitions": np.asarray(source.transitions, int).tolist(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(blob).hexdigest()


def theory_card(source: UnifilarSource, k: int) -> dict[str, object]:
    """Pre-training ground-truth summary suitable for stable JSON serialization."""
    sync_word = find_synchronizing_word(source)
    if sync_word is None:
        raise ValueError("theory cards currently require an exactly synchronizing source")
    static = best_static_partition(source, k)
    quotient = best_recursive_quotient(source, k)
    spectrum = deterministic_controller_spectrum(source, k)
    optimum = spectrum.optimum
    optimal_classes = spectrum.optimal_classes()
    runner = spectrum.runner_up()
    pi = source_stationary_distribution(source)
    card: dict[str, object] = {
        "schema_version": 1,
        "source_sha256": source_fingerprint(source),
        "n_states": source.n_states,
        "alphabet_size": source.alphabet_size,
        "emissions": source.emissions.tolist(),
        "transitions": source.transitions.tolist(),
        "stationary_distribution": pi.tolist(),
        "synchronizing_word": list(sync_word),
        "bayes_log_loss": bayes_log_loss(source),
        "memory_states": k,
        "static_loss": static.loss,
        "static_assignment": list(static.assignment),
        "deterministic_loss": optimum.loss,
        "deterministic_transition_table": optimum.transition_table.tolist(),
        "deterministic_initial_memory": optimum.initial_memory,
        "deterministic_alias_entropy": optimum.alias_entropy,
        "quotient_loss": quotient.loss,
        "quotient_assignment": list(quotient.assignment),
        "static_to_deterministic_gap": 0.0 if abs(optimum.loss - static.loss) < 1e-12 else optimum.loss - static.loss,
        "deterministic_to_quotient_gap": 0.0 if abs(quotient.loss - optimum.loss) < 1e-12 else quotient.loss - optimum.loss,
        "labeled_controller_count": spectrum.labeled_count,
        "relabeling_class_count": len(spectrum.classes),
        "optimal_relabeling_class_count": len(optimal_classes),
        "optimal_labeled_multiplicity": sum(c.labeled_multiplicity for c in optimal_classes),
        "runner_up_loss": None if runner is None else runner.loss,
        "relabeling_class_gap": None if runner is None else runner.loss - optimum.loss,
        "distinct_loss_level_count": len(spectrum.distinct_loss_levels()),
        "distinct_loss_gap": None if spectrum.distinct_loss_gap() == inf else spectrum.distinct_loss_gap(),
        "near_optimal_classes_1e-3": spectrum.near_optimal_class_count(1e-3),
        "near_optimal_classes_1e-2": spectrum.near_optimal_class_count(1e-2),
    }
    canonical = json.dumps(card, sort_keys=True, separators=(",", ":")).encode()
    card["theory_card_sha256"] = sha256(canonical).hexdigest()
    return card


def theory_card_json(card: dict[str, object]) -> str:
    return json.dumps(card, sort_keys=True, indent=2) + "\n"
