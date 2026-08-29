import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier.readout_prior import exact_distribution_gradient_snapshot
from memory_frontier.spectral import (
    collapsed_one_contrast_pressure_scale,
    observable_markov_source,
    predict_centered_collapsed_pressure,
)
from memory_frontier.surrogate import canonical_logits


def _walsh_fixture():
    # Orthonormal Walsh basis. The non-uniform predictive modes have distinct
    # eigenvalues 0.5, 0.2, and -0.1 while all transition probabilities stay > 0.
    vectors = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ],
        dtype=float,
    ).T / 2.0
    eigenvalues = np.array([1.0, 0.5, 0.2, -0.1])
    transition = vectors @ np.diag(eigenvalues) @ vectors.T
    return transition, vectors, eigenvalues


def _actual_centered_pressure(
    transition: np.ndarray,
    decoder_contrast: np.ndarray,
    *,
    horizon: int,
    margin: float,
    temperature: float,
) -> np.ndarray:
    q = transition.shape[0]
    source = observable_markov_source(transition)
    collapsed = np.zeros((q, q), dtype=int)
    transition_logits = canonical_logits(collapsed, margin=margin).detach().cpu().numpy()
    readout_logits = np.zeros((q, q), dtype=float)
    readout_logits[1] = decoder_contrast
    snapshot = exact_distribution_gradient_snapshot(
        source,
        transition_logits,
        readout_logits,
        horizon,
        initial_memory=0,
        temperature=temperature,
    )
    gradient = snapshot.transition_gradient
    pressure = np.array(
        [gradient[0, x, 0] - gradient[0, x, 1] for x in range(q)]
    )
    return pressure - pressure.mean()


def test_exact_predictive_operator_law_for_finite_decoder_contrast():
    transition, _, _ = _walsh_fixture()
    contrast = np.array([0.31, -0.17, 0.44, 0.08])
    kwargs = dict(horizon=17, margin=0.7, temperature=0.8)
    actual = _actual_centered_pressure(transition, contrast, **kwargs)
    predicted = predict_centered_collapsed_pressure(
        transition, contrast, **kwargs
    )
    np.testing.assert_allclose(actual, predicted, atol=1e-12, rtol=0.0)


def test_predictive_eigenmodes_set_exact_pressure_amplitude_and_sign():
    transition, vectors, eigenvalues = _walsh_fixture()
    horizon = 17
    margin = 0.7
    temperature = 0.8
    amplitude = 0.3
    scale = collapsed_one_contrast_pressure_scale(
        4, horizon, margin, temperature
    )

    for mode in range(1, 4):
        contrast = amplitude * vectors[:, mode]
        actual = _actual_centered_pressure(
            transition,
            contrast,
            horizon=horizon,
            margin=margin,
            temperature=temperature,
        )
        expected = scale * eigenvalues[mode] * contrast
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    # The negative predictive eigenvalue reverses the specialization pressure.
    negative_mode = amplitude * vectors[:, 3]
    negative_pressure = _actual_centered_pressure(
        transition,
        negative_mode,
        horizon=horizon,
        margin=margin,
        temperature=temperature,
    )
    assert float(np.dot(negative_pressure, negative_mode)) < 0.0
