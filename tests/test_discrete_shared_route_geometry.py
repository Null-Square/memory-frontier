import numpy as np

from memory_frontier.route_race import evaluate_multivariate_polynomial_gradient
from memory_frontier.support_invariants import evaluate_quadratic_invariant


def _shared_quadratic_routes(a, b):
    # L = -a*x*y - b*x*z.
    return {
        (1, 1, 0): -float(a),
        (1, 0, 1): -float(b),
    }


def test_finite_step_breaks_balanced_cone_except_coefficient_matched_ray():
    a = 0.43
    b = 0.17
    eta = 0.12
    coefficients = _shared_quadratic_routes(a, b)
    weights = np.array([1.0, -1.0, -1.0])

    y = 0.31
    z = 0.22
    x = np.sqrt(y * y + z * z)
    point = np.array([x, y, z])
    assert np.isclose(
        evaluate_quadratic_invariant(point, weights), 0.0, atol=1e-15
    )

    gradient = evaluate_multivariate_polynomial_gradient(coefficients, point)
    updated = point - eta * gradient
    updated_q = evaluate_quadratic_invariant(updated, weights)

    # On Q=0 the exact one-step defect simplifies to a negative square.
    expected = -eta**2 * (b * y - a * z) ** 2
    assert np.isclose(updated_q, expected, rtol=1e-13, atol=1e-15)
    assert updated_q < 0.0


def test_coefficient_matched_balanced_ray_is_exactly_preserved_by_finite_steps():
    a = 0.43
    b = 0.17
    eta = 0.12
    coefficients = _shared_quadratic_routes(a, b)
    weights = np.array([1.0, -1.0, -1.0])

    scale = 0.29
    norm = np.sqrt(a * a + b * b)
    point = scale * np.array([norm, a, b])
    gradient = evaluate_multivariate_polynomial_gradient(coefficients, point)
    updated = point - eta * gradient

    assert np.isclose(
        evaluate_quadratic_invariant(point, weights), 0.0, atol=1e-15
    )
    assert np.isclose(
        evaluate_quadratic_invariant(updated, weights), 0.0, atol=1e-15
    )

    # Every coordinate receives the same multiplicative factor 1+eta*norm.
    assert np.allclose(
        updated,
        (1.0 + eta * norm) * point,
        rtol=1e-14,
        atol=1e-15,
    )
