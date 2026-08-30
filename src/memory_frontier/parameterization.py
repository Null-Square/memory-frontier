from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


Polynomial = Mapping[tuple[int, ...], float]
SparsePolynomial = dict[tuple[int, ...], float]


def _polynomial_dimension(coefficients: Polynomial) -> int:
    if not coefficients:
        raise ValueError("coefficients must be non-empty")
    lengths = {len(exponent) for exponent in coefficients}
    if len(lengths) != 1:
        raise ValueError("all exponent tuples must have the same length")
    dimension = next(iter(lengths))
    if dimension < 1:
        raise ValueError("polynomials must have at least one variable")
    for exponent in coefficients:
        if any(int(power) != power or power < 0 for power in exponent):
            raise ValueError("exponents must be non-negative integers")
    return dimension


def polynomial_vanishing_order(
    coefficients: Polynomial,
    *,
    coefficient_atol: float = 0.0,
) -> int | None:
    """Smallest positive total degree with a nonzero coefficient.

    ``None`` means that the supplied polynomial is constant up to the chosen
    coefficient tolerance.
    """
    _polynomial_dimension(coefficients)
    atol = float(coefficient_atol)
    if atol < 0.0:
        raise ValueError("coefficient_atol must be non-negative")
    degrees = [
        sum(exponent)
        for exponent, coefficient in coefficients.items()
        if sum(exponent) > 0 and abs(float(coefficient)) > atol
    ]
    return min(degrees) if degrees else None


def _clean_polynomial(
    coefficients: Mapping[tuple[int, ...], float],
    *,
    coefficient_atol: float,
) -> SparsePolynomial:
    return {
        tuple(int(power) for power in exponent): float(coefficient)
        for exponent, coefficient in coefficients.items()
        if abs(float(coefficient)) > coefficient_atol
    }


def _multiply_polynomials(
    first: Polynomial,
    second: Polynomial,
    *,
    coefficient_atol: float,
) -> SparsePolynomial:
    if not first or not second:
        return {}
    first_dimension = _polynomial_dimension(first)
    second_dimension = _polynomial_dimension(second)
    if first_dimension != second_dimension:
        raise ValueError("polynomial dimensions do not match")

    output: SparsePolynomial = {}
    for exponent_first, coefficient_first in first.items():
        for exponent_second, coefficient_second in second.items():
            exponent = tuple(
                int(a + b) for a, b in zip(exponent_first, exponent_second)
            )
            output[exponent] = output.get(exponent, 0.0) + (
                float(coefficient_first) * float(coefficient_second)
            )
    return _clean_polynomial(output, coefficient_atol=coefficient_atol)


def _polynomial_power(
    coefficients: Polynomial,
    power: int,
    *,
    coefficient_atol: float,
) -> SparsePolynomial:
    exponent = int(power)
    if exponent != power or exponent < 0:
        raise ValueError("power must be a non-negative integer")
    dimension = _polynomial_dimension(coefficients)
    zero = (0,) * dimension
    if exponent == 0:
        return {zero: 1.0}

    result: SparsePolynomial = {zero: 1.0}
    base = _clean_polynomial(
        coefficients, coefficient_atol=coefficient_atol
    )
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply_polynomials(
                result, base, coefficient_atol=coefficient_atol
            )
        remaining >>= 1
        if remaining:
            base = _multiply_polynomials(
                base, base, coefficient_atol=coefficient_atol
            )
    return result


def compose_sparse_polynomial(
    coefficients: Polynomial,
    substitutions: Sequence[Polynomial],
    *,
    coefficient_atol: float = 0.0,
) -> SparsePolynomial:
    """Compose a sparse polynomial with a sparse polynomial map.

    ``substitutions[i]`` is the polynomial replacing the ``i``-th input
    coordinate. Every substitution must use the same output-coordinate
    dimension. This exact finite polynomial operation is useful for freezing
    regular and singular parameterization examples without finite differences.
    """
    input_dimension = _polynomial_dimension(coefficients)
    if len(substitutions) != input_dimension:
        raise ValueError("one substitution is required per input variable")
    atol = float(coefficient_atol)
    if atol < 0.0:
        raise ValueError("coefficient_atol must be non-negative")
    if not substitutions:
        raise ValueError("substitutions must be non-empty")

    output_dimensions = [_polynomial_dimension(sub) for sub in substitutions]
    if len(set(output_dimensions)) != 1:
        raise ValueError("all substitutions must use the same variables")
    output_dimension = output_dimensions[0]
    zero = (0,) * output_dimension

    output: SparsePolynomial = {}
    for exponent, coefficient in coefficients.items():
        term: SparsePolynomial = {zero: float(coefficient)}
        for variable, power in enumerate(exponent):
            if power == 0:
                continue
            factor = _polynomial_power(
                substitutions[variable],
                int(power),
                coefficient_atol=atol,
            )
            term = _multiply_polynomials(
                term, factor, coefficient_atol=atol
            )
            if not term:
                break
        for output_exponent, output_coefficient in term.items():
            output[output_exponent] = (
                output.get(output_exponent, 0.0) + output_coefficient
            )
    return _clean_polynomial(output, coefficient_atol=atol)


def polynomial_map_jacobian_at_zero(
    substitutions: Sequence[Polynomial],
) -> np.ndarray:
    """Jacobian at zero of a sparse polynomial parameterization map."""
    if not substitutions:
        raise ValueError("substitutions must be non-empty")
    output_dimensions = [_polynomial_dimension(sub) for sub in substitutions]
    if len(set(output_dimensions)) != 1:
        raise ValueError("all substitutions must use the same variables")
    output_dimension = output_dimensions[0]
    jacobian = np.zeros((len(substitutions), output_dimension), dtype=float)
    for row, substitution in enumerate(substitutions):
        for column in range(output_dimension):
            exponent = [0] * output_dimension
            exponent[column] = 1
            jacobian[row, column] = float(
                substitution.get(tuple(exponent), 0.0)
            )
    return jacobian


def linear_pullback_coefficients(
    coefficients: Polynomial,
    linear_map: Sequence[Sequence[float]],
    *,
    coefficient_atol: float = 0.0,
) -> SparsePolynomial:
    """Pull back ``L(eps)`` through ``eps = J theta`` exactly."""
    input_dimension = _polynomial_dimension(coefficients)
    matrix = np.asarray(linear_map, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != input_dimension:
        raise ValueError("linear_map rows must match polynomial variables")
    if matrix.shape[1] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("linear_map must be a finite non-empty matrix")
    output_dimension = matrix.shape[1]
    zero = (0,) * output_dimension
    substitutions: list[SparsePolynomial] = []
    for row in matrix:
        substitution: SparsePolynomial = {}
        for column, coefficient in enumerate(row):
            if coefficient == 0.0:
                continue
            exponent = [0] * output_dimension
            exponent[column] = 1
            substitution[tuple(exponent)] = float(coefficient)
        if not substitution:
            substitution = {zero: 0.0}
        substitutions.append(substitution)
    return compose_sparse_polynomial(
        coefficients,
        substitutions,
        coefficient_atol=coefficient_atol,
    )


def diagonal_power_pullback_coefficients(
    coefficients: Polynomial,
    powers: Sequence[int],
    *,
    coefficient_atol: float = 0.0,
) -> SparsePolynomial:
    """Pull back through the singular chart ``eps_i = theta_i**r_i``.

    Positive integer powers make the exponent map injective, so distinct input
    monomials never merge. Consequently the exact pulled-back vanishing order is

        min_alpha sum_i r_i * alpha_i

    over active nonconstant input monomials.
    """
    input_dimension = _polynomial_dimension(coefficients)
    if len(powers) != input_dimension:
        raise ValueError("powers must match polynomial variables")
    integer_powers = tuple(int(power) for power in powers)
    if any(power != original or power <= 0 for power, original in zip(integer_powers, powers)):
        raise ValueError("powers must be positive integers")

    output: SparsePolynomial = {}
    atol = float(coefficient_atol)
    if atol < 0.0:
        raise ValueError("coefficient_atol must be non-negative")
    for exponent, coefficient in coefficients.items():
        if abs(float(coefficient)) <= atol:
            continue
        pulled_exponent = tuple(
            int(power * multiplicity)
            for power, multiplicity in zip(integer_powers, exponent)
        )
        output[pulled_exponent] = output.get(pulled_exponent, 0.0) + float(
            coefficient
        )
    return _clean_polynomial(output, coefficient_atol=atol)


def diagonal_weighted_vanishing_order(
    coefficients: Polynomial,
    powers: Sequence[int],
    *,
    coefficient_atol: float = 0.0,
) -> int | None:
    """Exact vanishing order after ``eps_i = theta_i**r_i``."""
    pulled = diagonal_power_pullback_coefficients(
        coefficients,
        powers,
        coefficient_atol=coefficient_atol,
    )
    return polynomial_vanishing_order(
        pulled, coefficient_atol=coefficient_atol
    )
