from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cell_dynamics import HardCellOracle, build_hard_cell_oracle
from .core import UnifilarSource, source_stationary_distribution


@dataclass(frozen=True)
class HardCellStability:
    """Intrinsic counterfactual target stability after decoder equilibration."""

    transition_table: np.ndarray
    canonical_readout: np.ndarray
    hard_loss: float
    target_costs: np.ndarray
    improvement_advantage: np.ndarray

    def is_stable(self, atol: float = 1e-12) -> bool:
        return bool(np.max(self.improvement_advantage) <= atol)

    @property
    def max_improvement_advantage(self) -> float:
        return float(np.max(self.improvement_advantage))


def canonical_cell_readout(
    source: UnifilarSource,
    cell: HardCellOracle,
    *,
    support_tol: float = 1e-15,
) -> np.ndarray:
    """Bayes readout for occupied memory states; source marginal if unoccupied."""
    mass = cell.readout_symbol_mass
    row_mass = mass.sum(axis=-1, keepdims=True)
    stationary = source_stationary_distribution(source)
    marginal = stationary @ source.emissions
    readout = np.empty_like(mass)
    for m in range(cell.k):
        if row_mass[m, 0] > support_tol:
            readout[m] = mass[m] / row_mass[m, 0]
        else:
            readout[m] = marginal
    return readout


def hard_cell_stability(
    source: UnifilarSource,
    transition_table: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
) -> HardCellStability:
    """Exact surrogate fixed-point diagnostic for one hard controller.

    The decoder is first set to its exact finite-horizon optimum for the hard
    controller. ``target_costs[m,x,n]`` is then the pre-softmax counterfactual
    transition derivative for routing row ``(m,x)`` to target ``n``.

    ``improvement_advantage[m,x,n]`` is positive when target ``n`` has lower
    counterfactual cost than the table's current target. A cell is intrinsically
    stable when no target has positive advantage.
    """
    cell = build_hard_cell_oracle(
        source, transition_table, horizon, initial_memory
    )
    readout = canonical_cell_readout(source, cell)
    immediate_cost = -np.einsum(
        "sx,mx->sm", source.emissions, np.log(readout)
    )
    target_costs = np.einsum(
        "mxnsi,si->mxn",
        cell.transition_cost_kernel,
        immediate_cost,
    )
    table = cell.transition_table
    current_cost = np.take_along_axis(
        target_costs, table[..., None], axis=-1
    )
    advantage = current_cost - target_costs
    hard_loss = float(
        -np.sum(cell.readout_symbol_mass * np.log(readout))
    )
    return HardCellStability(
        transition_table=table.copy(),
        canonical_readout=readout,
        hard_loss=hard_loss,
        target_costs=target_costs,
        improvement_advantage=advantage,
    )
