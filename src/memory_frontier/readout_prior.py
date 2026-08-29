from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from .core import UnifilarSource, source_stationary_distribution
from .finite import controller_finite_horizon_log_loss
from .landscape import table_signature
from .surrogate import (
    TrainingRun,
    TrajectoryPoint,
    _batched_trainable_losses,
    _source_tensors,
)

ReadoutInitialization = Literal["uniform", "source_marginal", "random"]


def source_marginal_distribution(source: UnifilarSource) -> np.ndarray:
    """Stationary one-token marginal p(x) of the source."""
    pi = source_stationary_distribution(source)
    return (pi[:, None] * source.emissions).sum(axis=0)


def source_marginal_readout_logits(source: UnifilarSource, k: int) -> np.ndarray:
    """Logits whose softmax equals the source marginal for every memory state."""
    if k <= 0:
        raise ValueError("k must be positive")
    marginal = source_marginal_distribution(source)
    if np.any(marginal <= 0):
        raise ValueError("source marginal must be strictly positive")
    return np.tile(np.log(marginal)[None, :], (k, 1))


def _readout_logits_for_seed(
    source: UnifilarSource,
    k: int,
    mode: ReadoutInitialization,
    *,
    random_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if random_scale < 0:
        raise ValueError("random_scale must be non-negative")
    if mode == "uniform":
        return torch.zeros((k, source.alphabet_size), dtype=dtype)
    if mode == "source_marginal":
        return torch.as_tensor(
            source_marginal_readout_logits(source, k), dtype=dtype
        )
    if mode == "random":
        return torch.randn((k, source.alphabet_size), dtype=dtype) * random_scale
    raise ValueError(f"unknown readout initialization: {mode!r}")


@dataclass(frozen=True)
class GradientSnapshot:
    loss: float
    transition_gradient: np.ndarray
    readout_gradient: np.ndarray

    @property
    def transition_gradient_norm(self) -> float:
        return float(np.linalg.norm(self.transition_gradient))

    @property
    def readout_gradient_norm(self) -> float:
        return float(np.linalg.norm(self.readout_gradient))


def exact_distribution_gradient_snapshot(
    source: UnifilarSource,
    transition_logits: np.ndarray | torch.Tensor,
    readout_logits: np.ndarray | torch.Tensor,
    horizon: int,
    *,
    initial_memory: int = 0,
    temperature: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> GradientSnapshot:
    """Exact expected ST gradients at one explicit parameter point.

    This is a diagnostic, not an optimizer step. The forward transition remains
    hard argmax; only its backward Jacobian uses the softmax surrogate.
    """
    t = torch.as_tensor(transition_logits, dtype=dtype).detach().clone()
    r = torch.as_tensor(readout_logits, dtype=dtype).detach().clone()
    if t.ndim != 3:
        raise ValueError("transition_logits must have shape (k, alphabet, k)")
    k, alphabet_size, k2 = t.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition logit shape mismatch")
    if r.shape != (k, source.alphabet_size):
        raise ValueError("readout logit shape mismatch")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    t.requires_grad_(True)
    r.requires_grad_(True)
    emissions, symbol_dynamics, stationary = _source_tensors(source, dtype=dtype)
    losses = _batched_trainable_losses(
        source,
        t[None, ...],
        r[None, ...],
        horizon,
        initial_memory,
        temperature,
        emissions,
        symbol_dynamics,
        stationary,
    )
    loss = losses[0]
    loss.backward()
    assert t.grad is not None and r.grad is not None
    return GradientSnapshot(
        loss=float(loss.detach().cpu()),
        transition_gradient=t.grad.detach().cpu().numpy().copy(),
        readout_gradient=r.grad.detach().cpu().numpy().copy(),
    )


def train_exact_distribution_ste_with_readout_prior(
    source: UnifilarSource,
    k: int,
    horizon: int,
    seeds: Sequence[int],
    *,
    readout_initialization: ReadoutInitialization,
    steps: int = 180,
    learning_rate: float = 0.05,
    temperature: float = 1.0,
    initial_memory: int = 0,
    transition_init_scale: float = 0.25,
    readout_random_scale: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> tuple[TrainingRun, ...]:
    """Exact-distribution hard-state training with an explicit decoder prior.

    ``readout_initialization`` is deliberately required. An unused discrete
    memory state has no data-defined readout, so its initialization is an
    exploration resource and must not be hidden in the benchmark protocol.
    """
    if not seeds:
        return ()
    if horizon <= 0 or steps < 0:
        raise ValueError("invalid horizon/steps")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")
    if transition_init_scale < 0:
        raise ValueError("transition_init_scale must be non-negative")

    emissions, symbol_dynamics, stationary = _source_tensors(source, dtype=dtype)
    transition_rows: list[torch.Tensor] = []
    readout_rows: list[torch.Tensor] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        transition_rows.append(
            torch.randn((k, source.alphabet_size, k), dtype=dtype)
            * transition_init_scale
        )
        readout_rows.append(
            _readout_logits_for_seed(
                source,
                k,
                readout_initialization,
                random_scale=readout_random_scale,
                dtype=dtype,
            )
        )

    transition_logits = torch.stack(transition_rows).requires_grad_(True)
    readout_logits = torch.stack(readout_rows).requires_grad_(True)
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
