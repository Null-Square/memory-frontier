from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import UnifilarSource, source_stationary_distribution
from .exact_gradient import transition_logit_gradient_from_tensor_gradient


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class HardCellEvaluation:
    """Exact STE quantities evaluated from a cached hard-controller cell."""

    loss: float
    transition_table: np.ndarray
    transition_tensor_gradient: np.ndarray
    transition_logit_gradient: np.ndarray
    readout_logit_gradient: np.ndarray


@dataclass(frozen=True)
class HardCellOracle:
    """Cached exact dynamics inside one hard argmax cell.

    The hard transition table fixes all source-memory occupancies. Decoder logits
    change prediction costs but not those occupancies until an argmax boundary is
    crossed. ``transition_cost_kernel`` is the exact linear map from immediate
    source-memory cross-entropies to the counterfactual transition derivative.
    """

    source: UnifilarSource
    transition_table: np.ndarray
    horizon: int
    initial_memory: int
    forward_occupancy: np.ndarray
    readout_symbol_mass: np.ndarray
    transition_cost_kernel: np.ndarray

    @property
    def k(self) -> int:
        return int(self.transition_table.shape[0])

    @property
    def alphabet_size(self) -> int:
        return int(self.transition_table.shape[1])

    def contains(self, transition_logits: np.ndarray) -> bool:
        logits = np.asarray(transition_logits, dtype=float)
        expected = (self.k, self.alphabet_size, self.k)
        if logits.shape != expected:
            raise ValueError(f"transition_logits must have shape {expected}")
        return np.array_equal(logits.argmax(axis=-1), self.transition_table)

    def evaluate(
        self,
        transition_logits: np.ndarray,
        readout_logits: np.ndarray,
        *,
        temperature: float = 1.0,
    ) -> HardCellEvaluation:
        """Evaluate exact hard-forward / softmax-backward gradients in this cell."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = np.asarray(transition_logits, dtype=float)
        if not self.contains(logits):
            raise ValueError("transition logits lie outside this hard-controller cell")
        decoder_logits = np.asarray(readout_logits, dtype=float)
        if decoder_logits.shape != (self.k, self.alphabet_size):
            raise ValueError(
                "readout_logits must have shape "
                f"({self.k}, {self.alphabet_size})"
            )

        readout = _softmax(decoder_logits)
        immediate_cost = -np.einsum(
            "sx,mx->sm", self.source.emissions, np.log(readout)
        )
        transition_tensor_gradient = np.einsum(
            "mxnsi,si->mxn",
            self.transition_cost_kernel,
            immediate_cost,
        )
        transition_logit_gradient = transition_logit_gradient_from_tensor_gradient(
            logits,
            transition_tensor_gradient,
            temperature,
        )
        row_mass = self.readout_symbol_mass.sum(axis=-1, keepdims=True)
        readout_logit_gradient = row_mass * readout - self.readout_symbol_mass
        loss = float(-np.sum(self.readout_symbol_mass * np.log(readout)))
        return HardCellEvaluation(
            loss=loss,
            transition_table=self.transition_table.copy(),
            transition_tensor_gradient=transition_tensor_gradient,
            transition_logit_gradient=transition_logit_gradient,
            readout_logit_gradient=readout_logit_gradient,
        )


def build_hard_cell_oracle(
    source: UnifilarSource,
    transition_table: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
) -> HardCellOracle:
    """Precompute exact within-cell dynamics for a hard transition table."""
    table = np.asarray(transition_table, dtype=int)
    if table.ndim != 2:
        raise ValueError("transition_table must have shape (k, alphabet_size)")
    k, alphabet_size = table.shape
    if alphabet_size != source.alphabet_size:
        raise ValueError("controller alphabet does not match source")
    if np.any((table < 0) | (table >= k)):
        raise ValueError("transition targets must be valid memory states")
    T = int(horizon)
    if T <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    n_states = source.n_states
    stationary = source_stationary_distribution(source)

    forward = np.zeros((T, n_states, k), dtype=float)
    forward[0, :, initial_memory] = stationary
    for t in range(T - 1):
        for s in range(n_states):
            for m in range(k):
                weight = forward[t, s, m]
                if weight == 0.0:
                    continue
                for x in range(alphabet_size):
                    probability = source.emissions[s, x]
                    if probability == 0.0:
                        continue
                    forward[
                        t + 1,
                        source.transitions[s, x],
                        table[m, x],
                    ] += weight * probability

    readout_symbol_mass = np.zeros((k, alphabet_size), dtype=float)
    for t in range(T):
        readout_symbol_mass += (
            np.einsum("sm,sx->mx", forward[t], source.emissions) / T
        )

    cost_dim = n_states * k
    immediate_basis = np.eye(cost_dim, dtype=float).reshape(
        n_states, k, cost_dim
    )
    future_kernel = np.zeros(
        (T + 1, n_states, k, cost_dim), dtype=float
    )
    for t in range(T - 1, -1, -1):
        future_kernel[t] = immediate_basis
        if t == T - 1:
            continue
        for s in range(n_states):
            for m in range(k):
                for x in range(alphabet_size):
                    probability = source.emissions[s, x]
                    if probability == 0.0:
                        continue
                    future_kernel[t, s, m] += (
                        probability
                        * future_kernel[
                            t + 1,
                            source.transitions[s, x],
                            table[m, x],
                        ]
                    )

    transition_kernel = np.zeros(
        (k, alphabet_size, k, cost_dim), dtype=float
    )
    for t in range(T - 1):
        for s in range(n_states):
            for m in range(k):
                weight = forward[t, s, m]
                if weight == 0.0:
                    continue
                for x in range(alphabet_size):
                    probability = source.emissions[s, x]
                    if probability == 0.0:
                        continue
                    transition_kernel[m, x] += (
                        weight
                        * probability
                        / T
                        * future_kernel[
                            t + 1,
                            source.transitions[s, x],
                        ]
                    )

    return HardCellOracle(
        source=source,
        transition_table=table.copy(),
        horizon=T,
        initial_memory=int(initial_memory),
        forward_occupancy=forward,
        readout_symbol_mass=readout_symbol_mass,
        transition_cost_kernel=transition_kernel.reshape(
            k, alphabet_size, k, n_states, k
        ),
    )
