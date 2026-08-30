from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


Polynomial = Mapping[tuple[int, ...], float]


def _parameter_count(coefficients: Polynomial) -> int:
    if not coefficients:
        raise ValueError("coefficients must be non-empty")
    lengths = {len(exponent) for exponent in coefficients}
    if len(lengths) != 1:
        raise ValueError("all exponent tuples must have the same length")
    n_parameters = next(iter(lengths))
    if n_parameters < 1:
        raise ValueError("exponent tuples must be non-empty")
    return n_parameters


def _evaluate_polynomial(
    coefficients: Polynomial,
    point: Sequence[float],
) -> float:
    n_parameters = _parameter_count(coefficients)
    x = np.asarray(tuple(point), dtype=float)
    if x.shape != (n_parameters,):
        raise ValueError("point must match polynomial dimension")
    value = 0.0
    for exponent, coefficient in coefficients.items():
        if any(power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative")
        term = float(coefficient)
        for coordinate, power in zip(x, exponent):
            if power:
                term *= coordinate**power
        value += term
    return float(value)


def _gradient_polynomial_coefficients(
    coefficients: Polynomial,
) -> list[dict[tuple[int, ...], float]]:
    n_parameters = _parameter_count(coefficients)
    gradients: list[defaultdict[tuple[int, ...], float]] = [
        defaultdict(float) for _ in range(n_parameters)
    ]
    for exponent, coefficient in coefficients.items():
        if any(power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative")
        c = float(coefficient)
        if c == 0.0:
            continue
        for parameter, power in enumerate(exponent):
            if power == 0:
                continue
            reduced = list(exponent)
            reduced[parameter] -= 1
            gradients[parameter][tuple(reduced)] += c * power
    return [dict(gradient) for gradient in gradients]


def exponent_support_matrix(
    coefficients: Polynomial,
    *,
    coefficient_atol: float = 0.0,
) -> np.ndarray:
    """Return the active nonconstant exponent vectors as matrix rows.

    Coefficients are assumed to be represented canonically by exponent tuple, so
    algebraically identical monomials have already been combined. Constant terms
    are omitted because they do not contribute to gradient flow.
    """
    n_parameters = _parameter_count(coefficients)
    atol = float(coefficient_atol)
    if atol < 0.0:
        raise ValueError("coefficient_atol must be non-negative")

    rows: list[tuple[int, ...]] = []
    for exponent, coefficient in coefficients.items():
        if any(power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative")
        if abs(float(coefficient)) <= atol or sum(exponent) == 0:
            continue
        rows.append(tuple(int(power) for power in exponent))

    if not rows:
        return np.empty((0, n_parameters), dtype=float)
    rows.sort()
    return np.asarray(rows, dtype=float)


def quadratic_invariant_basis(
    coefficients: Polynomial,
    *,
    coefficient_atol: float = 0.0,
    rank_rtol: float | None = None,
) -> np.ndarray:
    """Basis for global diagonal quadratic invariants of polynomial gradient flow.

    For

        L(x) = sum_r c_r x**alpha_r,

    consider diagonal quadratic forms

        Q_w(x) = sum_i w_i x_i**2.

    Along gradient flow ``dx/dt=-grad L``, ``Q_w`` is globally conserved iff
    every active exponent vector is orthogonal to ``w``. Therefore the invariant
    weights are exactly the nullspace of the exponent-support matrix.

    Returns one orthonormal nullspace basis vector per row.
    """
    matrix = exponent_support_matrix(
        coefficients, coefficient_atol=coefficient_atol
    )
    n_parameters = matrix.shape[1]
    if matrix.shape[0] == 0:
        return np.eye(n_parameters, dtype=float)

    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        return np.eye(n_parameters, dtype=float)

    if rank_rtol is None:
        tolerance = (
            max(matrix.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )
    else:
        rtol = float(rank_rtol)
        if rtol < 0.0:
            raise ValueError("rank_rtol must be non-negative")
        tolerance = rtol * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    return vh[rank:].copy()


def quadratic_invariant_derivative_coefficients(
    coefficients: Polynomial,
    weights: Sequence[float],
    *,
    coefficient_atol: float = 0.0,
    orthogonality_atol: float = 0.0,
) -> dict[tuple[int, ...], float]:
    """Sparse polynomial for the time derivative of a quadratic balance law.

    For ``Q_w(x)=sum_i w_i*x_i**2`` and Euclidean gradient flow of

        L(x)=sum_alpha c_alpha*x**alpha,

    the exact derivative polynomial is

        dQ_w/dt = sum_alpha [-2*c_alpha*(alpha dot w)] * x**alpha.

    Terms whose loss coefficient or support-weight inner product is below the
    requested tolerance are omitted. This makes the first symmetry-breaking
    degree directly observable after coefficient cancellation.
    """
    n_parameters = _parameter_count(coefficients)
    w = np.asarray(tuple(weights), dtype=float)
    if w.shape != (n_parameters,):
        raise ValueError("weights must match polynomial dimension")
    coefficient_tol = float(coefficient_atol)
    orthogonality_tol = float(orthogonality_atol)
    if coefficient_tol < 0.0 or orthogonality_tol < 0.0:
        raise ValueError("tolerances must be non-negative")

    derivative: dict[tuple[int, ...], float] = {}
    for exponent, coefficient in coefficients.items():
        if any(power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative")
        c = float(coefficient)
        if abs(c) <= coefficient_tol or sum(exponent) == 0:
            continue
        support_weight = float(np.dot(w, exponent))
        if abs(support_weight) <= orthogonality_tol:
            continue
        derivative[tuple(int(power) for power in exponent)] = (
            -2.0 * c * support_weight
        )
    return derivative


def quadratic_invariant_breaking_degree(
    coefficients: Polynomial,
    weights: Sequence[float],
    *,
    coefficient_atol: float = 0.0,
    orthogonality_atol: float = 0.0,
) -> int | None:
    """First total degree that breaks a candidate quadratic invariant.

    Returns the minimum ``sum(alpha)`` for which the exact derivative polynomial
    of ``Q_w`` has a nonzero monomial. Returns ``None`` when ``Q_w`` is an exact
    invariant of the complete polynomial.
    """
    derivative = quadratic_invariant_derivative_coefficients(
        coefficients,
        weights,
        coefficient_atol=coefficient_atol,
        orthogonality_atol=orthogonality_atol,
    )
    if not derivative:
        return None
    return min(sum(exponent) for exponent in derivative)


def quadratic_invariant_discretization_defect_coefficients(
    coefficients: Polynomial,
    weights: Sequence[float],
    *,
    coefficient_atol: float = 0.0,
) -> dict[tuple[int, ...], float]:
    """Exact finite-step defect polynomial for vanilla gradient descent.

    For one step ``x_plus=x-eta*grad L(x)`` and

        Q_w(x)=sum_i w_i*x_i**2,

    the quadratic identity is exact:

        Q_w(x_plus)-Q_w(x)
        = eta * dQ_w/dt + eta**2 * D_w(x),

    where ``dQ_w/dt`` is the continuous gradient-flow derivative and

        D_w(x)=sum_i w_i*(partial_i L(x))**2.

    This function returns the coefficient dictionary of ``D_w`` after exact
    exponent aggregation. If ``Q_w`` is a continuous-time invariant, this
    polynomial is the entire one-step violation divided by ``eta**2``.
    """
    n_parameters = _parameter_count(coefficients)
    w = np.asarray(tuple(weights), dtype=float)
    if w.shape != (n_parameters,):
        raise ValueError("weights must match polynomial dimension")
    atol = float(coefficient_atol)
    if atol < 0.0:
        raise ValueError("coefficient_atol must be non-negative")

    gradients = _gradient_polynomial_coefficients(coefficients)
    defect: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for parameter, gradient in enumerate(gradients):
        weight = float(w[parameter])
        if weight == 0.0:
            continue
        terms = list(gradient.items())
        for exponent_a, coefficient_a in terms:
            for exponent_b, coefficient_b in terms:
                exponent = tuple(
                    power_a + power_b
                    for power_a, power_b in zip(exponent_a, exponent_b)
                )
                defect[exponent] += weight * coefficient_a * coefficient_b

    return {
        exponent: float(coefficient)
        for exponent, coefficient in defect.items()
        if abs(float(coefficient)) > atol
    }


def quadratic_invariant_discretization_degree(
    coefficients: Polynomial,
    weights: Sequence[float],
    *,
    coefficient_atol: float = 0.0,
) -> int | None:
    """First total degree of the finite-step defect polynomial ``D_w``."""
    defect = quadratic_invariant_discretization_defect_coefficients(
        coefficients,
        weights,
        coefficient_atol=coefficient_atol,
    )
    if not defect:
        return None
    return min(sum(exponent) for exponent in defect)


def quadratic_invariant_gradient_descent_step_change(
    coefficients: Polynomial,
    point: Sequence[float],
    weights: Sequence[float],
    step_size: float,
) -> float:
    """Exact change in ``Q_w`` under one vanilla gradient-descent step.

    The result is evaluated through the exact decomposition

        delta Q = eta * dQ/dt + eta**2 * D_w.
    """
    eta = float(step_size)
    derivative = quadratic_invariant_derivative(coefficients, point, weights)
    defect = quadratic_invariant_discretization_defect_coefficients(
        coefficients, weights
    )
    return eta * derivative + eta * eta * _evaluate_polynomial(defect, point)


def quadratic_invariant_derivative(
    coefficients: Polynomial,
    point: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Exact time derivative of ``sum_i weights_i*x_i**2`` under gradient flow."""
    n_parameters = _parameter_count(coefficients)
    x = np.asarray(tuple(point), dtype=float)
    w = np.asarray(tuple(weights), dtype=float)
    if x.shape != (n_parameters,) or w.shape != (n_parameters,):
        raise ValueError("point and weights must match polynomial dimension")

    derivative = 0.0
    for exponent, coefficient in coefficients.items():
        if any(power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative")
        c = float(coefficient)
        if c == 0.0 or sum(exponent) == 0:
            continue
        monomial = c
        for value, power in zip(x, exponent):
            if power:
                monomial *= value**power
        derivative -= 2.0 * monomial * float(np.dot(w, exponent))
    return derivative


def evaluate_quadratic_invariant(
    point: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Evaluate ``sum_i weights_i*x_i**2``."""
    x = np.asarray(tuple(point), dtype=float)
    w = np.asarray(tuple(weights), dtype=float)
    if x.ndim != 1 or x.size == 0 or w.shape != x.shape:
        raise ValueError("point and weights must be matching non-empty vectors")
    return float(np.dot(w, x * x))
