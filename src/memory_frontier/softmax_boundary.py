from __future__ import annotations

from collections.abc import Sequence
from math import fsum, log


def _validate_probability(probability: float, *, name: str) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def _validate_degree_and_coefficient(coefficient: float, degree: int) -> tuple[float, int]:
    c = float(coefficient)
    d = int(degree)
    if c >= 0.0:
        raise ValueError("coefficient must be negative for a beneficial route")
    if d < 1:
        raise ValueError("degree must be positive")
    return c, d


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
    the scalar binary logits gives

        dz/dt = (-c) * p**d * (1-p).

    The probability-coordinate flow is returned separately below.
    """
    c, d = _validate_degree_and_coefficient(coefficient, degree)
    p = _validate_probability(probability, name="probability")
    return (-c) * p**d * (1.0 - p)


def symmetric_logit_monomial_probability_velocity(
    coefficient: float,
    degree: int,
    probability: float,
) -> float:
    """Probability velocity induced by Euclidean flow in scalar binary logits.

    Since ``dp/dz = p(1-p)``, the exact symmetric flow is

        dp/dt = (-c) * p**(d+1) * (1-p)**2.

    Near the simplex boundary this is ``(-c) p**(d+1) (1+O(p))``. Thus the
    boundary-logit metric contributes two powers of the rare-edge probability
    relative to direct Euclidean probability-coordinate flow ``p**(d-1)``.
    """
    p = _validate_probability(probability, name="probability")
    return p * (1.0 - p) * symmetric_logit_monomial_logit_velocity(
        coefficient,
        degree,
        p,
    )


def softmax_target_jacobian_norm_squared(
    probabilities: Sequence[float],
    target_index: int,
) -> float:
    """Squared Euclidean logit-Jacobian norm of one softmax probability.

    For a full ``K``-way softmax row with target probability ``p`` and remaining
    probabilities ``q_a``, the target-probability logit gradient has components

        p(1-p)                       on the target logit,
        -p q_a                       on every other logit.

    Hence

        ||grad_z p||^2 = p^2 ((1-p)^2 + sum_a q_a^2).

    This quantity is the metric factor converting an objective derivative with
    respect to the target edge probability into target-probability speed under
    ordinary Euclidean gradient flow in all row logits.
    """
    values = tuple(float(value) for value in probabilities)
    if len(values) < 2:
        raise ValueError("probabilities must contain at least two categories")
    target = int(target_index)
    if not 0 <= target < len(values):
        raise ValueError("target_index is out of range")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("probabilities must lie strictly between zero and one")
    if abs(fsum(values) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")

    p = values[target]
    other_square_sum = fsum(
        value * value for index, value in enumerate(values) if index != target
    )
    return p * p * ((1.0 - p) ** 2 + other_square_sum)


def softmax_target_jacobian_norm_squared_bounds(
    target_probability: float,
    categories: int,
) -> tuple[float, float]:
    """Sharp dimension-only bounds for ``||grad_z p||^2`` in a full softmax row.

    If ``K=categories`` and the non-target probabilities sum to ``1-p``, then
    Cauchy--Schwarz and concentration on one category give

        p^2(1-p)^2 * K/(K-1)
        <= ||grad_z p||^2
        <= 2 p^2(1-p)^2.

    The lower bound is attained when all non-target probabilities are equal; the
    upper bound is approached when their mass concentrates on one category.
    """
    p = _validate_probability(target_probability, name="target_probability")
    k = int(categories)
    if k < 2:
        raise ValueError("categories must be at least two")
    base = p * p * (1.0 - p) ** 2
    return base * k / (k - 1), 2.0 * base


def symmetric_full_softmax_monomial_probability_velocity_bounds(
    coefficient: float,
    degree: int,
    target_probability: float,
    categories: int,
) -> tuple[float, float]:
    """Bounds on rare-edge speed for a degree-``d`` monomial under full softmax.

    Consider ``d`` identical rows whose target-edge probabilities all equal ``p``
    and an isolated objective ``L=c prod_i p_i`` with ``c<0``. Euclidean gradient
    flow in every logit of every row gives

        dp/dt = (-c) p**(d-1) ||grad_z p||^2.

    The returned lower/upper bounds are therefore constant multiples of
    ``p**(d+1)(1-p)**2``. In particular, for fixed ``K`` the rare-edge bootstrap
    exponent is the same ``delta**(-d)`` as in the exact scalar binary-logit
    calculation, although the exact prefactor depends on the non-target
    distribution and evolves with the row logits.
    """
    c, d = _validate_degree_and_coefficient(coefficient, degree)
    p = _validate_probability(target_probability, name="target_probability")
    lower_metric, upper_metric = softmax_target_jacobian_norm_squared_bounds(
        p, categories
    )
    objective_factor = (-c) * p ** (d - 1)
    return objective_factor * lower_metric, objective_factor * upper_metric


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
    """Exact scalar-binary-logit gradient-flow time between edge probabilities.

    For the symmetric binary-logit parameterization described above,

        dp/dt = C p**(d+1) (1-p)**2,  C=-coefficient > 0.

    Integrating this scalar ODE gives the exact time to move from
    ``initial_probability`` to ``threshold_probability``.
    """
    c, d = _validate_degree_and_coefficient(coefficient, degree)
    initial = _validate_probability(initial_probability, name="initial_probability")
    target = _validate_probability(
        threshold_probability, name="threshold_probability"
    )
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
    """Leading divergent term of scalar-logit bootstrap time near probability zero.

    For fixed positive target probability and ``delta -> 0``, the exact time is

        tau(delta) = delta**(-d) / ((-c) d) * (1 + o(1)).

    This helper returns that leading term.
    """
    c, d = _validate_degree_and_coefficient(coefficient, degree)
    initial = _validate_probability(initial_probability, name="initial_probability")
    return initial ** (-d) / ((-c) * d)
