from __future__ import annotations

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
