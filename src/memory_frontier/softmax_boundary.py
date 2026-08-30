from __future__ import annotations

from math import log


def _validate_probability(probability: float, *, name: str) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def symmetric_logit_monomial_logit_velocity(
    coefficient: float,
    degree: int,
    probability: float,
) -> float:
    """Common logit velocity for an isolated square-free construction monomial.

    Consider

        L = c * prod_{i=1}^d p_i,  c < 0,
        p_i = sigmoid(z_i),

    with all probabilities/logits equal by symmetry. Euclidean gradient flow in
    the logits gives

        dz/dt = (-c) * p**d * (1-p).

    The probability-coordinate flow is returned separately below.
    """
    c = float(coefficient)
    d = int(degree)
    p = _validate_probability(probability, name="probability")
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if d < 1:
        raise ValueError("degree must be positive")
    return (-c) * p**d * (1.0 - p)


def symmetric_logit_monomial_probability_velocity(
    coefficient: float,
    degree: int,
    probability: float,
) -> float:
    """Probability velocity induced by Euclidean gradient flow in logits.

    Since ``dp/dz = p(1-p)``, the exact symmetric flow is

        dp/dt = (-c) * p**(d+1) * (1-p)**2.

    Near the simplex boundary this is ``(-c) p**(d+1) (1+O(p))``. Thus the
    boundary-softmax metric contributes two powers of the rare-edge probability
    relative to direct Euclidean probability-coordinate flow ``p**(d-1)``.
    """
    p = _validate_probability(probability, name="probability")
    return p * (1.0 - p) * symmetric_logit_monomial_logit_velocity(
        coefficient,
        degree,
        p,
    )


def _logit_escape_antiderivative(degree: int, probability: float) -> float:
    """Antiderivative of ``1 / (p**(d+1) * (1-p)**2)``."""
    d = int(degree)
    p = _validate_probability(probability, name="probability")
    if d < 1:
        raise ValueError("degree must be positive")

    value = (d + 1) * log(p / (1.0 - p)) + 1.0 / (1.0 - p)
    for power in range(1, d + 1):
        value -= (d + 1 - power) / power * p ** (-power)
    return value


def symmetric_logit_monomial_completion_time(
    coefficient: float,
    degree: int,
    initial_probability: float,
    threshold_probability: float,
) -> float:
    """Exact logit-gradient-flow time between two edge probabilities.

    For the symmetric binary-logit parameterization described above,

        dp/dt = C p**(d+1) (1-p)**2,  C=-coefficient > 0.

    Integrating this scalar ODE gives the exact time to move from
    ``initial_probability`` to ``threshold_probability``.
    """
    c = float(coefficient)
    d = int(degree)
    initial = _validate_probability(initial_probability, name="initial_probability")
    target = _validate_probability(
        threshold_probability, name="threshold_probability"
    )
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if d < 1:
        raise ValueError("degree must be positive")
    if initial >= target:
        return 0.0

    return (
        _logit_escape_antiderivative(d, target)
        - _logit_escape_antiderivative(d, initial)
    ) / (-c)


def symmetric_logit_boundary_leading_time(
    coefficient: float,
    degree: int,
    initial_probability: float,
) -> float:
    """Leading divergent term of logit-flow bootstrap time near probability zero.

    For fixed positive target probability and ``delta -> 0``, the exact time is

        tau(delta) = delta**(-d) / ((-c) d) * (1 + o(1)).

    This helper returns that leading term.
    """
    c = float(coefficient)
    d = int(degree)
    initial = _validate_probability(initial_probability, name="initial_probability")
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if d < 1:
        raise ValueError("degree must be positive")
    return initial ** (-d) / ((-c) * d)
