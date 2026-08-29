from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import UnifilarSource
from .exact_gradient import (
    hard_value_gradient,
    transition_logit_gradient_from_tensor_gradient,
)


def _softmax_vector(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1:
        raise ValueError("common_readout_logits must be one-dimensional")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / weights.sum()


def _linearized_transition_gradients(
    source: UnifilarSource,
    transition_logits: np.ndarray,
    common_readout_logits: np.ndarray,
    directions: np.ndarray,
    horizon: int,
    initial_memory: int,
    temperature: float,
) -> tuple[float, np.ndarray]:
    """Batch exact decoder-to-transition mixed derivatives.

    ``directions`` has shape ``(batch, k, alphabet_size)`` and contains decoder
    logit perturbations. The return value has shape
    ``(batch, k, alphabet_size, k)`` and is the corresponding first-order change
    in the hard-forward / soft-backward transition-logit gradient.
    """
    logits = np.asarray(transition_logits, dtype=float)
    if logits.ndim != 3:
        raise ValueError(
            "transition_logits must have shape (k, alphabet_size, k)"
        )
    k, alphabet_size, k2 = logits.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition logit shape mismatch")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= initial_memory < k:
        raise ValueError("invalid initial memory")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

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
    table = logits.argmax(axis=-1)
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

    # Derivative of log-softmax(a_m) in decoder-logit direction D_m.
    centered = perturbations - np.einsum(
        "bmx,x->bm", perturbations, readout
    )[:, :, None]
    delta_immediate = -np.einsum(
        "sx,bmx->bsm", source.emissions, centered
    )

    # Counterfactual future-value response under the fixed hard table.
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
                source_weight = (
                    forward[t, :, m] * source.emissions[:, x]
                )
                delta_tensor_gradient[:, m, x, :] += (
                    np.einsum("s,bsn->bn", source_weight, future_targets)
                    / horizon
                )

    output = np.empty_like(delta_tensor_gradient)
    for b in range(batch):
        output[b] = transition_logit_gradient_from_tensor_gradient(
            logits,
            delta_tensor_gradient[b],
            temperature,
        )
    return hard.loss, output


@dataclass(frozen=True)
class GradientAccessibilityOperator:
    """Mixed derivative from decoder logits to STE transition gradients.

    ``matrix`` maps flattened decoder-logit perturbations of shape
    ``readout_shape`` to flattened first-order transition-gradient changes of
    shape ``transition_shape`` at a point where all decoder rows are identical.
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
        output = self.matrix @ direction.ravel()
        return output.reshape(self.transition_shape)


def linearized_transition_gradient_from_readout_direction(
    source: UnifilarSource,
    transition_logits: np.ndarray,
    common_readout_logits: np.ndarray,
    readout_logit_direction: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Exact directional decoder-to-transition mixed derivative.

    All decoder rows are equal to ``common_readout_logits`` at the base point.
    The hard transition table is held fixed inside its current argmax cell.
    """
    logits = np.asarray(transition_logits, dtype=float)
    if logits.ndim != 3:
        raise ValueError(
            "transition_logits must have shape (k, alphabet_size, k)"
        )
    k, alphabet_size, k2 = logits.shape
    if k != k2:
        raise ValueError("transition logits must be square in memory targets")
    direction = np.asarray(readout_logit_direction, dtype=float)
    if direction.shape != (k, alphabet_size):
        raise ValueError("readout_logit_direction has wrong shape")
    _, response = _linearized_transition_gradients(
        source,
        logits,
        common_readout_logits,
        direction[None, ...],
        horizon,
        initial_memory,
        temperature,
    )
    return response[0]


def gradient_accessibility_operator(
    source: UnifilarSource,
    transition_logits: np.ndarray,
    common_readout_logits: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    temperature: float = 1.0,
) -> GradientAccessibilityOperator:
    """Construct the exact local Gradient Accessibility Operator (GAO).

    The GAO is the mixed derivative

        d(transition STE gradient) / d(decoder logits)

    at a decoder-symmetric base point. It is exact for the current hard argmax
    table and the specified softmax-backward transition geometry.
    """
    logits = np.asarray(transition_logits, dtype=float)
    if logits.ndim != 3:
        raise ValueError(
            "transition_logits must have shape (k, alphabet_size, k)"
        )
    k, alphabet_size, k2 = logits.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition logit shape mismatch")

    input_dim = k * alphabet_size
    basis = np.eye(input_dim, dtype=float).reshape(
        input_dim, k, alphabet_size
    )
    base_loss, responses = _linearized_transition_gradients(
        source,
        logits,
        common_readout_logits,
        basis,
        horizon,
        initial_memory,
        temperature,
    )
    matrix = responses.reshape(input_dim, -1).T
    return GradientAccessibilityOperator(
        matrix=matrix,
        transition_shape=(k, alphabet_size, k),
        readout_shape=(k, alphabet_size),
        base_loss=base_loss,
    )
