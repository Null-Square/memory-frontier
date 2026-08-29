from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterator

import numpy as np

from .core import UnifilarSource, enumerate_deterministic_controllers
from .finite import controller_finite_horizon_log_loss


def table_signature(table: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(table, dtype=int).ravel())


def table_from_signature(
    signature: tuple[int, ...], k: int, alphabet_size: int
) -> np.ndarray:
    if len(signature) != k * alphabet_size:
        raise ValueError("signature length does not match table shape")
    return np.asarray(signature, dtype=int).reshape(k, alphabet_size)


def one_edit_tables(
    table: np.ndarray,
) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Yield every transition table differing in exactly one target choice."""
    f = np.asarray(table, dtype=int)
    k, alphabet_size = f.shape
    for m in range(k):
        for x in range(alphabet_size):
            current = int(f[m, x])
            for target in range(k):
                if target == current:
                    continue
                other = f.copy()
                other[m, x] = target
                yield m, x, target, other


@dataclass(frozen=True)
class HardEdge:
    source: tuple[int, ...]
    target: tuple[int, ...]
    memory_state: int
    symbol: int
    old_target: int
    new_target: int
    source_loss: float
    target_loss: float

    @property
    def gain(self) -> float:
        """Positive iff taking this one-edit edge lowers exact loss."""
        return self.source_loss - self.target_loss


@dataclass(frozen=True)
class HardNode:
    signature: tuple[int, ...]
    loss: float


@dataclass(frozen=True)
class HardLandscape:
    k: int
    alphabet_size: int
    horizon: int
    initial_memory: int
    nodes: tuple[HardNode, ...]
    edges: tuple[HardEdge, ...]

    def loss_map(self) -> dict[tuple[int, ...], float]:
        return {node.signature: node.loss for node in self.nodes}

    def global_minima(self, atol: float = 1e-12) -> tuple[HardNode, ...]:
        best = min(node.loss for node in self.nodes)
        return tuple(node for node in self.nodes if node.loss <= best + atol)

    def local_minima(self, atol: float = 1e-12) -> tuple[HardNode, ...]:
        improving = {edge.source for edge in self.edges if edge.gain > atol}
        return tuple(node for node in self.nodes if node.signature not in improving)

    def best_loss(self) -> float:
        return min(node.loss for node in self.nodes)

    def second_distinct_loss(self, atol: float = 1e-12) -> float:
        values = sorted(node.loss for node in self.nodes)
        best = values[0]
        for value in values[1:]:
            if value > best + atol:
                return value
        return inf


def finite_horizon_hard_landscape(
    source: UnifilarSource,
    k: int,
    horizon: int,
    initial_memory: int = 0,
) -> HardLandscape:
    """Exhaust the labeled hard-controller graph for a reset horizon.

    Nodes are deterministic transition tables. Directed edges change exactly one
    table entry to one alternative memory target. Losses are exact expected
    finite-horizon NLLs with the optimal shared readout for each hard controller.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    loss_map: dict[tuple[int, ...], float] = {}
    tables: dict[tuple[int, ...], np.ndarray] = {}
    for table in enumerate_deterministic_controllers(k, source.alphabet_size):
        signature = table_signature(table)
        tables[signature] = table.copy()
        loss_map[signature] = controller_finite_horizon_log_loss(
            source, table, initial_memory, horizon
        )

    nodes = tuple(HardNode(signature, loss_map[signature]) for signature in sorted(loss_map))
    edges: list[HardEdge] = []
    for signature in sorted(loss_map):
        table = tables[signature]
        for m, x, new_target, other in one_edit_tables(table):
            target = table_signature(other)
            edges.append(
                HardEdge(
                    source=signature,
                    target=target,
                    memory_state=m,
                    symbol=x,
                    old_target=int(table[m, x]),
                    new_target=new_target,
                    source_loss=loss_map[signature],
                    target_loss=loss_map[target],
                )
            )

    return HardLandscape(
        k=k,
        alphabet_size=source.alphabet_size,
        horizon=horizon,
        initial_memory=initial_memory,
        nodes=nodes,
        edges=tuple(edges),
    )
