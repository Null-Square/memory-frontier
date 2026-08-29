import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier.readout_prior import exact_distribution_gradient_snapshot
from memory_frontier.spectral import (
    collapsed_one_contrast_pressure_scale,
    collapsed_pressure_accessibility_margin,
    observable_markov_source,
    predict_centered_collapsed_pressure,
    predict_raw_collapsed_pressure,
    two_level_accessibility_cutoff,
)
from memory_frontier.surrogate import canonical_logits


def _walsh_fixture():
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


def _actual_pressure(
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
    return np.array(
        [gradient[0, x, 0] - gradient[0, x, 1] for x in range(q)]
    )


def test_exact_predictive_operator_law_for_finite_decoder_contrast():
    transition, _, _ = _walsh_fixture()
    contrast = np.array([0.31, -0.17, 0.44, 0.08])
    kwargs = dict(horizon=17, margin=0.7, temperature=0.8)
    actual = _actual_pressure(transition, contrast, **kwargs)
    actual -= actual.mean()
    predicted = predict_centered_collapsed_pressure(
        transition, contrast, **kwargs
    )
    np.testing.assert_allclose(actual, predicted, atol=1e-12, rtol=0.0)


def test_exact_centered_operator_law_also_holds_for_nonnormal_source():
    transition = np.array(
        [
            [0.5, 0.4, 0.1],
            [0.2, 0.4, 0.4],
            [0.3, 0.2, 0.5],
        ],
        dtype=float,
    )
    assert np.linalg.norm(transition @ transition.T - transition.T @ transition) > 0.05
    contrast = np.array([0.7, -0.2, 0.1])
    kwargs = dict(horizon=13, margin=0.8, temperature=0.9)
    actual = _actual_pressure(transition, contrast, **kwargs)
    actual -= actual.mean()
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
        actual = _actual_pressure(
            transition,
            contrast,
            horizon=horizon,
            margin=margin,
            temperature=temperature,
        )
        actual -= actual.mean()
        expected = scale * eigenvalues[mode] * contrast
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    negative_mode = amplitude * vectors[:, 3]
    negative_pressure = _actual_pressure(
        transition,
        negative_mode,
        horizon=horizon,
        margin=margin,
        temperature=temperature,
    )
    negative_pressure -= negative_pressure.mean()
    assert float(np.dot(negative_pressure, negative_mode)) < 0.0


def test_raw_pressure_exhibits_exact_finite_contrast_accessibility_barrier():
    transition, vectors, _ = _walsh_fixture()
    mode = vectors[:, 3]
    kwargs = dict(horizon=8, margin=0.3, temperature=0.8)

    below = 0.39 * mode
    above = 0.42 * mode

    for contrast in (below, above):
        actual = _actual_pressure(transition, contrast, **kwargs)
        predicted = predict_raw_collapsed_pressure(
            transition, contrast, **kwargs
        )
        np.testing.assert_allclose(actual, predicted, atol=1e-12, rtol=0.0)

    assert collapsed_pressure_accessibility_margin(transition, below) > 0.0
    assert collapsed_pressure_accessibility_margin(transition, above) < 0.0
    assert np.max(_actual_pressure(transition, below, **kwargs)) > 0.0
    assert np.max(_actual_pressure(transition, above, **kwargs)) < 0.0


def test_two_level_cutoff_predicts_full_mode_by_contrast_phase_grid():
    transition, vectors, eigenvalues = _walsh_fixture()
    kwargs = dict(horizon=8, margin=0.3, temperature=0.8)
    amplitudes = (0.2, 0.5, 0.9, 1.5, 2.5)

    for amplitude in amplitudes:
        cutoff = two_level_accessibility_cutoff(amplitude / 2.0)
        for mode in range(1, 4):
            contrast = amplitude * vectors[:, mode]
            predicted_accessible = abs(eigenvalues[mode]) > cutoff
            actual_accessible = np.max(
                _actual_pressure(transition, contrast, **kwargs)
            ) > 1e-12
            assert actual_accessible == predicted_accessible
