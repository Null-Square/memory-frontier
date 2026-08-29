from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .core import UnifilarSource, source_stationary_distribution
from .finite import controller_average_occupancy, controller_finite_horizon_log_loss
from .landscape import one_edit_tables, table_signature


def canonical_readout(
    source: UnifilarSource,
    table: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
) -> np.ndarray:
    """Canonical readout used for table-local surrogate diagnostics.

    Occupied memory states get their exact Bayes readout under the hard
    controller's finite-horizon occupancy. An unoccupied memory state has no
    data-defined optimum, so it is assigned the source marginal. This convention
    makes the backward field explicit instead of silently differentiating through
    an undefined argmin.
    """
    occupancy = controller_average_occupancy(source, table, initial_memory, horizon)
    mass = np.einsum("sm,sx->mx", occupancy, source.emissions)
    row_mass = mass.sum(axis=1, keepdims=True)
    marginal = (
        source_stationary_distribution(source)[:, None] * source.emissions
    ).sum(axis=0)
    readout = np.empty_like(mass)
    for m in range(mass.shape[0]):
        readout[m] = mass[m] / row_mass[m, 0] if row_mass[m, 0] > 1e-15 else marginal
    return readout


def canonical_logits(
    table: np.ndarray,
    margin: float = 1.0,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Symmetric logit embedding of a hard transition table."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    table = np.asarray(table, dtype=int)
    k, alphabet_size = table.shape
    logits = torch.zeros((k, alphabet_size, k), dtype=dtype)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits.requires_grad_(True)


def _source_tensors(
    source: UnifilarSource,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    emissions = torch.as_tensor(source.emissions, dtype=dtype, device=device)
    symbol_dynamics = torch.zeros(
        (source.alphabet_size, source.n_states, source.n_states),
        dtype=dtype,
        device=device,
    )
    for s in range(source.n_states):
        for x in range(source.alphabet_size):
            symbol_dynamics[x, s, source.transitions[s, x]] = emissions[s, x]
    stationary = torch.as_tensor(
        source_stationary_distribution(source), dtype=dtype, device=device
    )
    return emissions, symbol_dynamics, stationary


def _straight_through_transition(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    k = logits.shape[-1]
    soft = torch.softmax(logits / temperature, dim=-1)
    hard = torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=k).to(
        logits.dtype
    )
    return hard + soft - soft.detach()


def ste_fixed_readout_loss(
    source: UnifilarSource,
    logits: torch.Tensor,
    readout: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Exact-distribution ST loss with a fixed readout.

    Forward transition values are deterministic one-hot choices. Only the
    backward Jacobian is taken from the temperature-scaled softmax.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    k, alphabet_size, k2 = logits.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("logit shape mismatch")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    transition = _straight_through_transition(logits, temperature)
    emissions, symbol_dynamics, stationary = _source_tensors(
        source, dtype=logits.dtype, device=logits.device
    )
    log_readout = torch.log(
        torch.as_tensor(readout, dtype=logits.dtype, device=logits.device)
    )
    dist = torch.zeros(
        (source.n_states, k), dtype=logits.dtype, device=logits.device
    )
    dist[:, initial_memory] = stationary
    total = logits.new_tensor(0.0)
    for _ in range(horizon):
        total = total - torch.einsum("sm,sx,mx->", dist, emissions, log_readout)
        dist = torch.einsum(
            "sm,xst,mxn->tn", dist, symbol_dynamics, transition
        )
    return total / horizon


@dataclass(frozen=True)
class SurrogateEdge:
    memory_state: int
    symbol: int
    old_target: int
    new_target: int
    exact_gain: float
    descent_pressure: float

    @property
    def exact_sign(self) -> int:
        return 0 if abs(self.exact_gain) <= 1e-12 else (1 if self.exact_gain > 0 else -1)

    @property
    def surrogate_sign(self) -> int:
        return (
            0
            if abs(self.descent_pressure) <= 1e-12
            else (1 if self.descent_pressure > 0 else -1)
        )

    @property
    def agrees(self) -> bool:
        return (
            self.exact_sign != 0
            and self.surrogate_sign != 0
            and self.exact_sign == self.surrogate_sign
        )


@dataclass(frozen=True)
class CanonicalSTEField:
    signature: tuple[int, ...]
    hard_loss: float
    memory_mass: tuple[float, ...]
    margin: float
    temperature: float
    edges: tuple[SurrogateEdge, ...]

    def sign_fidelity(self) -> float:
        comparable = [edge for edge in self.edges if edge.exact_sign and edge.surrogate_sign]
        if not comparable:
            return float("nan")
        return sum(edge.agrees for edge in comparable) / len(comparable)

    def surrogate_stable(self, atol: float = 1e-12) -> bool:
        """No canonical descent pressure favors a one-edit alternative."""
        return all(edge.descent_pressure <= atol for edge in self.edges)

    def fully_active(self, atol: float = 1e-12) -> bool:
        return all(mass > atol for mass in self.memory_mass)


def canonical_ste_field(
    source: UnifilarSource,
    table: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> CanonicalSTEField:
    """Compare exact one-edit gains with a canonical straight-through field.

    ``exact_gain = L(F) - L(F')`` is positive when changing one transition target
    improves the exact hard-controller objective. ``descent_pressure`` is
    ``g_current - g_alternative``; it is positive when gradient descent decreases
    the current-vs-alternative logit margin and therefore favors the same switch.
    """
    table = np.asarray(table, dtype=int)
    occupancy = controller_average_occupancy(source, table, initial_memory, horizon)
    base_loss = controller_finite_horizon_log_loss(
        source, table, initial_memory, horizon
    )
    readout = canonical_readout(source, table, horizon, initial_memory)
    logits = canonical_logits(table, margin)
    loss = ste_fixed_readout_loss(
        source,
        logits,
        readout,
        horizon,
        initial_memory=initial_memory,
        temperature=temperature,
    )
    loss.backward()
    gradient = logits.grad.detach().cpu().numpy()
    if abs(float(loss.detach().cpu()) - base_loss) > 1e-10:
        raise RuntimeError("ST forward loss disagrees with exact hard loss")

    edges: list[SurrogateEdge] = []
    for m, x, new_target, other in one_edit_tables(table):
        old_target = int(table[m, x])
        other_loss = controller_finite_horizon_log_loss(
            source, other, initial_memory, horizon
        )
        exact_gain = base_loss - other_loss
        pressure = float(
            gradient[m, x, old_target] - gradient[m, x, new_target]
        )
        edges.append(
            SurrogateEdge(
                memory_state=m,
                symbol=x,
                old_target=old_target,
                new_target=new_target,
                exact_gain=exact_gain,
                descent_pressure=pressure,
            )
        )

    return CanonicalSTEField(
        signature=table_signature(table),
        hard_loss=base_loss,
        memory_mass=tuple(float(value) for value in occupancy.sum(axis=0)),
        margin=margin,
        temperature=temperature,
        edges=tuple(edges),
    )


def margin_sweep(
    source: UnifilarSource,
    table: np.ndarray,
    horizon: int,
    margins: Iterable[float],
    initial_memory: int = 0,
    temperature: float = 1.0,
) -> tuple[CanonicalSTEField, ...]:
    return tuple(
        canonical_ste_field(
            source,
            table,
            horizon,
            initial_memory=initial_memory,
            margin=float(margin),
            temperature=temperature,
        )
        for margin in margins
    )


def _batched_trainable_losses(
    source: UnifilarSource,
    transition_logits: torch.Tensor,
    readout_logits: torch.Tensor,
    horizon: int,
    initial_memory: int,
    temperature: float,
    emissions: torch.Tensor,
    symbol_dynamics: torch.Tensor,
    stationary: torch.Tensor,
) -> torch.Tensor:
    batch, k, alphabet_size, k2 = transition_logits.shape
    if k != k2 or readout_logits.shape != (batch, k, alphabet_size):
        raise ValueError("shape mismatch")
    transition = _straight_through_transition(transition_logits, temperature)
    log_readout = torch.log_softmax(readout_logits, dim=-1)
    dist = torch.zeros(
        (batch, source.n_states, k),
        dtype=transition_logits.dtype,
        device=transition_logits.device,
    )
    dist[:, :, initial_memory] = stationary[None, :]
    total = torch.zeros(
        batch, dtype=transition_logits.dtype, device=transition_logits.device
    )
    for _ in range(horizon):
        total = total - torch.einsum(
            "bsm,sx,bmx->b", dist, emissions, log_readout
        )
        dist = torch.einsum(
            "bsm,xst,bmxn->btn", dist, symbol_dynamics, transition
        )
    return total / horizon


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    signature: tuple[int, ...]
    oracle_loss: float


@dataclass(frozen=True)
class TrainingRun:
    seed: int
    trajectory: tuple[TrajectoryPoint, ...]
    final_signature: tuple[int, ...]
    final_oracle_loss: float

    @property
    def initial_signature(self) -> tuple[int, ...]:
        return self.trajectory[0].signature


def train_exact_distribution_ste_batch(
    source: UnifilarSource,
    k: int,
    horizon: int,
    seeds: Sequence[int],
    *,
    steps: int = 180,
    learning_rate: float = 0.05,
    temperature: float = 1.0,
    initial_memory: int = 0,
    init_scale: float = 0.25,
    dtype: torch.dtype = torch.float32,
) -> tuple[TrainingRun, ...]:
    """Train independent hard-state predictors using exact expected sequence loss.

    Seeds are batched only for speed: each seed owns disjoint transition and
    readout parameters. No sequence sampling or minibatch noise is used.
    """
    if not seeds:
        return ()
    if horizon <= 0 or steps < 0:
        raise ValueError("invalid horizon/steps")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    emissions, symbol_dynamics, stationary = _source_tensors(source, dtype=dtype)
    initial_logits: list[torch.Tensor] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        initial_logits.append(
            torch.randn((k, source.alphabet_size, k), dtype=dtype) * init_scale
        )
    transition_logits = torch.stack(initial_logits).requires_grad_(True)
    readout_logits = torch.zeros(
        (len(seeds), k, source.alphabet_size), dtype=dtype, requires_grad=True
    )
    optimizer = torch.optim.Adam(
        [transition_logits, readout_logits], lr=learning_rate
    )

    trajectories: list[list[TrajectoryPoint]] = [[] for _ in seeds]
    previous: list[tuple[int, ...] | None] = [None for _ in seeds]
    for step in range(steps + 1):
        tables = transition_logits.detach().argmax(dim=-1).cpu().numpy()
        for i, table in enumerate(tables):
            signature = table_signature(table)
            if signature != previous[i]:
                trajectories[i].append(
                    TrajectoryPoint(
                        step=step,
                        signature=signature,
                        oracle_loss=controller_finite_horizon_log_loss(
                            source, table, initial_memory, horizon
                        ),
                    )
                )
                previous[i] = signature
        if step == steps:
            break
        optimizer.zero_grad()
        losses = _batched_trainable_losses(
            source,
            transition_logits,
            readout_logits,
            horizon,
            initial_memory,
            temperature,
            emissions,
            symbol_dynamics,
            stationary,
        )
        losses.sum().backward()
        optimizer.step()

    final_tables = transition_logits.detach().argmax(dim=-1).cpu().numpy()
    runs: list[TrainingRun] = []
    for i, seed in enumerate(seeds):
        final_signature = table_signature(final_tables[i])
        runs.append(
            TrainingRun(
                seed=int(seed),
                trajectory=tuple(trajectories[i]),
                final_signature=final_signature,
                final_oracle_loss=controller_finite_horizon_log_loss(
                    source, final_tables[i], initial_memory, horizon
                ),
            )
        )
    return tuple(runs)
