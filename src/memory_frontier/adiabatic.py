from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cell_stability import hard_cell_stability
from .core import UnifilarSource
from .landscape import table_signature


@dataclass(frozen=True)
class AdiabaticStep:
    """One decoder-equilibrated surrogate policy-improvement step."""

    source_signature: tuple[int, ...]
    target_signature: tuple[int, ...]
    memory_state: int | None
    symbol: int | None
    old_target: int | None
    new_target: int | None
    improvement_advantage: float

    @property
    def is_fixed_point(self) -> bool:
        return self.source_signature == self.target_signature


@dataclass(frozen=True)
class AdiabaticOrbit:
    """Deterministic hard-table orbit under the adiabatic successor map."""

    path: tuple[tuple[int, ...], ...]
    cycle: tuple[tuple[int, ...], ...]

    @property
    def reaches_fixed_point(self) -> bool:
        return len(self.cycle) == 1


def adiabatic_successor(
    source: UnifilarSource,
    transition_table: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
    *,
    atol: float = 1e-12,
) -> AdiabaticStep:
    """Take the strongest intrinsic counterfactual edit after decoder equilibration.

    The decoder is set to its exact finite-horizon optimum for the current hard
    controller. Among all one-transition alternatives, choose the target with
    largest positive counterfactual improvement advantage. Ties are resolved by
    deterministic row/symbol/target iteration order. If no positive advantage
    exists, the controller is a fixed point.
    """
    table = np.asarray(transition_table, dtype=int)
    if table.ndim != 2:
        raise ValueError("transition_table must have shape (k, alphabet_size)")
    stability = hard_cell_stability(
        source, table, horizon, initial_memory
    )
    best_advantage = float(atol)
    best_edit: tuple[int, int, int] | None = None
    k, alphabet_size = table.shape
    for m in range(k):
        for x in range(alphabet_size):
            old_target = int(table[m, x])
            for new_target in range(k):
                if new_target == old_target:
                    continue
                advantage = float(
                    stability.improvement_advantage[m, x, new_target]
                )
                if advantage > best_advantage:
                    best_advantage = advantage
                    best_edit = (m, x, new_target)

    source_signature = table_signature(table)
    if best_edit is None:
        return AdiabaticStep(
            source_signature=source_signature,
            target_signature=source_signature,
            memory_state=None,
            symbol=None,
            old_target=None,
            new_target=None,
            improvement_advantage=0.0,
        )

    m, x, new_target = best_edit
    old_target = int(table[m, x])
    target = table.copy()
    target[m, x] = new_target
    return AdiabaticStep(
        source_signature=source_signature,
        target_signature=table_signature(target),
        memory_state=m,
        symbol=x,
        old_target=old_target,
        new_target=new_target,
        improvement_advantage=best_advantage,
    )


def adiabatic_orbit(
    source: UnifilarSource,
    transition_table: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
    *,
    atol: float = 1e-12,
    max_steps: int = 10000,
) -> AdiabaticOrbit:
    """Follow the exact adiabatic successor map until a fixed point or cycle."""
    table = np.asarray(transition_table, dtype=int).copy()
    seen: dict[tuple[int, ...], int] = {}
    path: list[tuple[int, ...]] = []
    for _ in range(max_steps):
        signature = table_signature(table)
        if signature in seen:
            start = seen[signature]
            return AdiabaticOrbit(
                path=tuple(path),
                cycle=tuple(path[start:]),
            )
        seen[signature] = len(path)
        path.append(signature)
        step = adiabatic_successor(
            source,
            table,
            horizon,
            initial_memory,
            atol=atol,
        )
        if step.is_fixed_point:
            return AdiabaticOrbit(
                path=tuple(path),
                cycle=(signature,),
            )
        table = np.asarray(step.target_signature, dtype=int).reshape(table.shape)
    raise RuntimeError("adiabatic orbit exceeded max_steps")
