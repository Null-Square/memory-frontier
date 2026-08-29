from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import UnifilarSource, source_stationary_distribution


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=float) / float(temperature)
    z = z - np.max(z, axis=-1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=-1, keepdims=True)


def _validate_readout(readout: np.ndarray, k: int, alphabet_size: int) -> np.ndarray:
    r = np.asarray(readout, dtype=float)
    if r.shape != (k, alphabet_size):
        raise ValueError("readout must have shape (k, alphabet_size)")
    if np.any(r <= 0):
        raise ValueError("readout probabilities must be strictly positive")
    if not np.allclose(r.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("each readout row must sum to 1")
    return r


@dataclass(frozen=True)
class HardValueGradient:
    """Exact finite-horizon value derivatives for a hard controller.

    ``transition_tensor_gradient[m, x, n]`` is the derivative of the average
    sequence loss with respect to an independent transition weight routing
    ``(m, x)`` into target memory ``n``, evaluated at the hard controller.
    It is the pre-softmax-Jacobian object used by straight-through estimators.
    """

    loss: float
    transition_table: np.ndarray
    transition_tensor_gradient: np.ndarray
    readout_symbol_mass: np.ndarray
    forward_occupancy: np.ndarray
    future_value: np.ndarray


@dataclass(frozen=True)
class ExactSTEGradient:
    """Exact hard-forward / softmax-backward gradient, computed without autograd."""

    loss: float
    transition_table: np.ndarray
    transition_tensor_gradient: np.ndarray
    transition_logit_gradient: np.ndarray
    readout_logit_gradient: np.ndarray
    forward_occupancy: np.ndarray
    future_value: np.ndarray


def hard_value_gradient(
    source: UnifilarSource,
    transition_table: np.ndarray,
    readout: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
) -> HardValueGradient:
    """Exact loss and counterfactual transition values for a hard controller."""
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
    readout = _validate_readout(readout, k, alphabet_size)

    stationary = source_stationary_distribution(source)
    n_states = source.n_states
    forward = np.zeros((horizon, n_states, k), dtype=float)
    forward[0, :, initial_memory] = stationary
    for t in range(horizon - 1):
        for s in range(n_states):
            for m in range(k):
                weight = forward[t, s, m]
                if weight == 0.0:
                    continue
                for x in range(alphabet_size):
                    probability = source.emissions[s, x]
                    if probability == 0.0:
                        continue
                    s2 = source.transitions[s, x]
                    m2 = table[m, x]
                    forward[t + 1, s2, m2] += weight * probability

    log_readout = np.log(readout)
    immediate = -np.einsum("sx,mx->sm", source.emissions, log_readout)
    future = np.zeros((horizon + 1, n_states, k), dtype=float)
    for t in range(horizon - 1, -1, -1):
        future[t] = immediate
        if t == horizon - 1:
            continue
        for s in range(n_states):
            for m in range(k):
                tail = 0.0
                for x in range(alphabet_size):
                    probability = source.emissions[s, x]
                    if probability == 0.0:
                        continue
                    tail += probability * future[
                        t + 1, source.transitions[s, x], table[m, x]
                    ]
                future[t, s, m] += tail

    transition_gradient = np.zeros((k, alphabet_size, k), dtype=float)
    for t in range(horizon - 1):
        for s in range(n_states):
            for m in range(k):
                weight = forward[t, s, m]
                if weight == 0.0:
                    continue
                for x in range(alphabet_size):
                    probability = source.emissions[s, x]
                    if probability == 0.0:
                        continue
                    s2 = source.transitions[s, x]
                    transition_gradient[m, x] += (
                        weight
                        * probability
                        / horizon
                        * future[t + 1, s2]
                    )

    symbol_mass = np.zeros((k, alphabet_size), dtype=float)
    for t in range(horizon):
        symbol_mass += (
            np.einsum("sm,sx->mx", forward[t], source.emissions) / horizon
        )

    loss = float(np.sum(forward * immediate[None, :, :]) / horizon)
    return HardValueGradient(
        loss=loss,
        transition_table=table.copy(),
        transition_tensor_gradient=transition_gradient,
        readout_symbol_mass=symbol_mass,
        forward_occupancy=forward,
        future_value=future,
    )


def transition_logit_gradient_from_tensor_gradient(
    transition_logits: np.ndarray,
    transition_tensor_gradient: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Apply the full temperature-softmax Jacobian to a transition gradient."""
    logits = np.asarray(transition_logits, dtype=float)
    tensor_gradient = np.asarray(transition_tensor_gradient, dtype=float)
    if logits.shape != tensor_gradient.shape or logits.ndim != 3:
        raise ValueError(
            "transition logits and tensor gradient must share shape "
            "(k, alphabet, k)"
        )
    soft = _softmax(logits, temperature)
    baseline = np.sum(soft * tensor_gradient, axis=-1, keepdims=True)
    return soft * (tensor_gradient - baseline) / float(temperature)


def exact_ste_gradient(
    source: UnifilarSource,
    transition_logits: np.ndarray,
    readout_logits: np.ndarray,
    horizon: int,
    initial_memory: int = 0,
    temperature: float = 1.0,
) -> ExactSTEGradient:
    """Exact ST gradient for arbitrary hard controller logits and decoder logits.

    The forward transition is the argmax hard controller. The backward
    transition Jacobian is the temperature-scaled softmax Jacobian, matching the
    straight-through estimator used by the training code. No autograd or
    sequence sampling is used here.
    """
    logits = np.asarray(transition_logits, dtype=float)
    readout_logits = np.asarray(readout_logits, dtype=float)
    if logits.ndim != 3:
        raise ValueError(
            "transition_logits must have shape (k, alphabet_size, k)"
        )
    k, alphabet_size, k2 = logits.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition logit shape mismatch")
    if readout_logits.shape != (k, alphabet_size):
        raise ValueError("readout_logits must have shape (k, alphabet_size)")

    table = logits.argmax(axis=-1)
    readout = _softmax(readout_logits)
    hard = hard_value_gradient(
        source, table, readout, horizon, initial_memory
    )
    transition_logit_gradient = transition_logit_gradient_from_tensor_gradient(
        logits, hard.transition_tensor_gradient, temperature
    )
    row_mass = hard.readout_symbol_mass.sum(axis=-1, keepdims=True)
    readout_logit_gradient = row_mass * readout - hard.readout_symbol_mass
    return ExactSTEGradient(
        loss=hard.loss,
        transition_table=hard.transition_table,
        transition_tensor_gradient=hard.transition_tensor_gradient,
        transition_logit_gradient=transition_logit_gradient,
        readout_logit_gradient=readout_logit_gradient,
        forward_occupancy=hard.forward_occupancy,
        future_value=hard.future_value,
    )
