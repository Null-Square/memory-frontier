from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryCandidate:
    memory_state: int
    symbol: int
    old_target: int
    new_target: int
    margin: float
    descent_pressure: float
    estimated_steps: float


@dataclass(frozen=True)
class BoundaryRacePrediction:
    """Linearized race between hard argmax boundaries."""

    candidates: tuple[BoundaryCandidate, ...]

    @property
    def winner(self) -> BoundaryCandidate | None:
        return self.candidates[0] if self.candidates else None


def predict_boundary_race(
    transition_logits: np.ndarray,
    transition_logit_gradient: np.ndarray,
    learning_rate: float,
    *,
    pressure_tol: float = 1e-14,
) -> BoundaryRacePrediction:
    """Predict the next hard edit from current margins and gradient pressure.

    For current hard target ``j`` and alternative ``n`` in one transition row,
    define the positive argmax margin

        d = z_j - z_n

    and descent pressure

        p = grad_j - grad_n.

    Under a frozen-gradient SGD linearization, the margin shrinks by ``eta*p``
    per update, giving estimated boundary time ``d / (eta*p)`` when ``p>0``.
    Candidates are returned in increasing estimated-time order.
    """
    logits = np.asarray(transition_logits, dtype=float)
    gradient = np.asarray(transition_logit_gradient, dtype=float)
    if logits.ndim != 3 or logits.shape != gradient.shape:
        raise ValueError(
            "transition logits and gradient must share shape (k, alphabet, k)"
        )
    k, _, k2 = logits.shape
    if k != k2:
        raise ValueError("transition logits must be square in memory targets")
    eta = float(learning_rate)
    if eta <= 0:
        raise ValueError("learning_rate must be positive")
    if pressure_tol < 0:
        raise ValueError("pressure_tol must be non-negative")

    table = logits.argmax(axis=-1)
    candidates: list[BoundaryCandidate] = []
    for m in range(k):
        for x in range(logits.shape[1]):
            old = int(table[m, x])
            for new in range(k):
                if new == old:
                    continue
                margin = float(logits[m, x, old] - logits[m, x, new])
                pressure = float(
                    gradient[m, x, old] - gradient[m, x, new]
                )
                if margin <= 0.0 or pressure <= pressure_tol:
                    continue
                candidates.append(
                    BoundaryCandidate(
                        memory_state=m,
                        symbol=x,
                        old_target=old,
                        new_target=new,
                        margin=margin,
                        descent_pressure=pressure,
                        estimated_steps=margin / (eta * pressure),
                    )
                )
    candidates.sort(
        key=lambda candidate: (
            candidate.estimated_steps,
            candidate.memory_state,
            candidate.symbol,
            candidate.new_target,
        )
    )
    return BoundaryRacePrediction(candidates=tuple(candidates))
