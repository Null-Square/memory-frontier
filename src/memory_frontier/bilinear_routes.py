from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _route_matrix(route_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(route_matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("route_matrix must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("route_matrix must contain only finite values")
    return matrix


def bilinear_route_spectrum(
    route_matrix: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of a quadratic entrance/exit route-coupling matrix.

    For the beneficial bilinear leading loss

        L = -x.T @ A @ y,

    the singular values of ``A`` are the exact growth rates of independent
    gradient-flow construction modes. The returned tuple is ``(U, s, Vt)`` with
    full orthogonal factors, so left and right nullspaces are retained as frozen
    coordinates.
    """
    matrix = _route_matrix(route_matrix)
    return np.linalg.svd(matrix, full_matrices=True)


def bilinear_positive_growth_modes(
    route_matrix: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Positive-growth eigenmodes of the bilinear gradient-flow Jacobian.

    For ``L=-x.T@A@y``, the gradient-flow Jacobian is

        J = [[0, A], [A.T, 0]].

    If ``A=U diag(s) V.T``, every singular triplet yields a normalized positive
    eigenvector ``[u_i; v_i]/sqrt(2)`` with eigenvalue ``s_i``. This function
    returns ``(s, G)`` where columns of ``G`` are those growth modes, ordered by
    decreasing singular value.
    """
    matrix = _route_matrix(route_matrix)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    count = len(singular_values)
    modes = np.zeros((matrix.shape[0] + matrix.shape[1], count), dtype=float)
    modes[: matrix.shape[0]] = u[:, :count]
    modes[matrix.shape[0] :] = vt.T[:, :count]
    modes /= np.sqrt(2.0)
    return singular_values, modes


def project_symmetric_operator(
    operator: Sequence[Sequence[float]],
    basis: Sequence[Sequence[float]],
) -> np.ndarray:
    """Project a symmetric local operator onto an orthonormal mode basis."""
    matrix = np.asarray(operator, dtype=float)
    vectors = np.asarray(basis, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be square")
    if vectors.ndim != 2 or vectors.shape[0] != matrix.shape[0]:
        raise ValueError("basis rows must match operator dimension")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("operator must be symmetric")
    if not np.allclose(vectors.T @ vectors, np.eye(vectors.shape[1]), atol=1e-12):
        raise ValueError("basis columns must be orthonormal")
    return vectors.T @ matrix @ vectors


def bilinear_route_modal_coordinates(
    route_matrix: Sequence[Sequence[float]],
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return singular values and left/right modal coordinates."""
    matrix = _route_matrix(route_matrix)
    x = np.asarray(tuple(left), dtype=float)
    y = np.asarray(tuple(right), dtype=float)
    if x.shape != (matrix.shape[0],):
        raise ValueError("left coordinates must match route_matrix rows")
    if y.shape != (matrix.shape[1],):
        raise ValueError("right coordinates must match route_matrix columns")

    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    return singular_values, u.T @ x, vt @ y


def bilinear_route_flow(
    route_matrix: Sequence[Sequence[float]],
    initial_left: Sequence[float],
    initial_right: Sequence[float],
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact gradient-flow solution for ``L=-x.T@A@y``.

    Gradient flow is

        dx/dt = A y,
        dy/dt = A.T x.

    In SVD coordinates ``A=U diag(s) V.T``, each coupled singular mode satisfies

        d xi_i/dt  = s_i eta_i,
        d eta_i/dt = s_i xi_i,

    while all left/right nullspace coordinates are frozen. The coupled modes
    therefore evolve by exact ``cosh``/``sinh`` blocks.
    """
    matrix = _route_matrix(route_matrix)
    x0 = np.asarray(tuple(initial_left), dtype=float)
    y0 = np.asarray(tuple(initial_right), dtype=float)
    if x0.shape != (matrix.shape[0],):
        raise ValueError("initial_left must match route_matrix rows")
    if y0.shape != (matrix.shape[1],):
        raise ValueError("initial_right must match route_matrix columns")
    t = float(time)
    if t < 0.0:
        raise ValueError("time must be non-negative")

    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=True)
    left_modes = u.T @ x0
    right_modes = vt @ y0
    evolved_left = left_modes.copy()
    evolved_right = right_modes.copy()

    for index, singular_value in enumerate(singular_values):
        argument = singular_value * t
        cosine = np.cosh(argument)
        sine = np.sinh(argument)
        evolved_left[index] = (
            left_modes[index] * cosine + right_modes[index] * sine
        )
        evolved_right[index] = (
            right_modes[index] * cosine + left_modes[index] * sine
        )

    return u @ evolved_left, vt.T @ evolved_right


def bilinear_modal_balance_invariants(
    route_matrix: Sequence[Sequence[float]],
    left: Sequence[float],
    right: Sequence[float],
) -> np.ndarray:
    """Per-singular-mode quadratic balance values ``xi_i^2-eta_i^2``.

    One value is returned for every singular value of the rectangular route
    matrix, including zero singular values. Every returned value is exactly
    conserved by the bilinear gradient flow.
    """
    singular_values, left_modes, right_modes = bilinear_route_modal_coordinates(
        route_matrix, left, right
    )
    count = len(singular_values)
    return left_modes[:count] ** 2 - right_modes[:count] ** 2


def bilinear_global_balance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Global bilinear balance ``||left||^2-||right||^2``."""
    x = np.asarray(tuple(left), dtype=float)
    y = np.asarray(tuple(right), dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise ValueError("left and right must be non-empty vectors")
    return float(np.dot(x, x) - np.dot(y, y))
