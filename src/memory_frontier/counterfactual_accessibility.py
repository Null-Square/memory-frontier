from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import UnifilarSource
from .exact_gradient import hard_value_gradient


def _softmax_vector(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1:
        raise ValueError("common_readout_logits must be one-dimensional")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / weights.sum()


def _softmax_rows(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=float) / float(temperature)
    values = values - np.max(values, axis=-1, keepdims=True)
    weights = np.exp(values)
    return weights / weights.sum(axis=-1, keepdims=True)


def _linearized_transition_tensor_gradients(
    source: UnifilarSource,
    transition_table: np.ndarray,
    common_readout_logits: np.ndarray,
    directions: np.ndarray,
    horizon: int,
    initial_memory: int,
) -> tuple[float, np.ndarray]:
    """Exact decoder-to-counterfactual-transition mixed derivatives."""
    table = np.asarray(transition_table, dtype=int)
    if table.ndim != 2:
        raise ValueError("transition_table must have shape (k, alphabet_size)")
    k, alphabet_size = table.shape
    if alphabet_size != source.alphabet_size:
        raise ValueError("controller alphabet does not match source")
    if np.any((table < 0) | (table >= k)):
        raise ValueError("transition targets must be valid memory states")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")

    base_logits = np.asarray(common_readout_logits, dtype=float)
    if base_logits.shape != (alphabet_size,):
        raise ValueError(
            "common_readout_logits must have shape (alphabet_size,)"
        )
    perturbations = np.asarray(directions, dtype=float)
    if perturbations.ndim != 3 or perturbations.shape[1:] != (
        k,
        alphabet_size,
    ):
        raise ValueError(
            "directions must have shape (batch, k, alphabet_size)"
        )

    readout = _softmax_vector(base_logits)
    hard = hard_value_gradient(
        source,
        table,
        np.tile(readout[None, :], (k, 1)),
        horizon,
        initial_memory,
    )
    forward = hard.forward_occupancy
    batch = perturbations.shape[0]
    n_states = source.n_states

    centered_decoder = perturbations - np.einsum(
        "bmx,x->bm", perturbations, readout
    )[:, :, None]
    delta_immediate = -np.einsum(
        "sx,bmx->bsm", source.emissions, centered_decoder
    )

    delta_future = np.zeros(
        (horizon + 1, batch, n_states, k), dtype=float
    )
    for t in range(horizon - 1, -1, -1):
        delta_future[t] = delta_immediate
        if t == horizon - 1:
            continue
        for x in range(alphabet_size):
            successor_states = source.transitions[:, x]
            for m in range(k):
                target = int(table[m, x])
                mapped = delta_future[t + 1][:, successor_states, target]
                delta_future[t, :, :, m] += (
                    source.emissions[:, x][None, :] * mapped
                )

    delta_tensor_gradient = np.zeros(
        (batch, k, alphabet_size, k), dtype=float
    )
    for t in range(horizon - 1):
        for x in range(alphabet_size):
            successor_states = source.transitions[:, x]
            future_targets = delta_future[t + 1][:, successor_states, :]
            for m in range(k):
                source_weight = forward[t, :, m] * source.emissions[:, x]
                delta_tensor_gradient[:, m, x, :] += (
                    np.einsum("s,bsn->bn", source_weight, future_targets)
                    / horizon
                )
    return hard.loss, delta_tensor_gradient


@dataclass(frozen=True)
class CounterfactualAccessibilityOperator:
    """Intrinsic mixed derivative before the transition-softmax Jacobian.

    ``matrix`` maps flattened decoder-logit perturbations to target-centered
    counterfactual transition-value gradients. Target centering removes exactly
    the transition-row constant directions that every softmax Jacobian kills.
    """

    matrix: np.ndarray
    transition_shape: tuple[int, int, int]
    readout_shape: tuple[int, int]
    base_loss: float

    @property
    def singular_values(self) -> np.ndarray:
        return np.linalg.svd(self.matrix, compute_uv=False)

    @property
    def leading_singular_value(self) -> float:
        values = self.singular_values
        return float(values[0]) if len(values) else 0.0

    @property
    def theoretical_rank_bound(self) -> int:
        k, alphabet_size = self.readout_shape
        return (k - 1) * (alphabet_size - 1)

    def numerical_rank(
        self,
        *,
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> int:
        if rtol < 0 or atol < 0:
            raise ValueError("rank tolerances must be non-negative")
        values = self.singular_values
        if not len(values):
            return 0
        threshold = max(float(atol), float(rtol) * float(values[0]))
        return int(np.count_nonzero(values > threshold))

    def apply(self, readout_logit_direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(readout_logit_direction, dtype=float)
        if direction.shape != self.readout_shape:
            raise ValueError("readout_logit_direction has wrong shape")
        return (self.matrix @ direction.ravel()).reshape(self.transition_shape)

    def push_through_transition_softmax(
        self,
        transition_logits: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Map the intrinsic operator into the exact GAO matrix."""
        logits = np.asarray(transition_logits, dtype=float)
        if logits.shape != self.transition_shape:
            raise ValueError("transition_logits has wrong shape")
        probabilities = _softmax_rows(logits, temperature)
        output = np.empty_like(self.matrix)
        for column in range(self.matrix.shape[1]):
            value = self.matrix[:, column].reshape(self.transition_shape)
            mapped = np.empty_like(value)
            for m in range(self.transition_shape[0]):
                for x in range(self.transition_shape[1]):
                    p = probabilities[m, x]
                    jacobian = (
                        np.diag(p) - np.outer(p, p)
                    ) / float(temperature)
                    mapped[m, x] = jacobian @ value[m, x]
            output[:, column] = mapped.ravel()
        return output


def counterfactual_accessibility_operator(
    source: UnifilarSource,
    transition_table: np.ndarray,
    common_readout_logits: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
) -> CounterfactualAccessibilityOperator:
    """Construct the target-centered intrinsic accessibility operator."""
    table = np.asarray(transition_table, dtype=int)
    if table.ndim != 2:
        raise ValueError("transition_table must have shape (k, alphabet_size)")
    k, alphabet_size = table.shape
    input_dim = k * alphabet_size
    basis = np.eye(input_dim, dtype=float).reshape(
        input_dim, k, alphabet_size
    )
    base_loss, responses = _linearized_transition_tensor_gradients(
        source,
        table,
        common_readout_logits,
        basis,
        horizon,
        initial_memory,
    )
    centered = responses - responses.mean(axis=-1, keepdims=True)
    matrix = centered.reshape(input_dim, -1).T
    return CounterfactualAccessibilityOperator(
        matrix=matrix,
        transition_shape=(k, alphabet_size, k),
        readout_shape=(k, alphabet_size),
        base_loss=base_loss,
    )
