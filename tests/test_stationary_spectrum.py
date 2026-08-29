import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier import source_stationary_distribution
from memory_frontier.readout_prior import exact_distribution_gradient_snapshot
from memory_frontier.spectral import (
    collapsed_transition_pressure_scale,
    observable_markov_source,
    predict_stationary_centered_rescaled_pressure,
    predict_stationary_raw_collapsed_pressure,
)
from memory_frontier.surrogate import canonical_logits


def _actual_pressure(
    transition: np.ndarray,
    decoder_contrast: np.ndarray,
    *,
    horizon: int,
    margin: float,
    temperature: float,
):
    source = observable_markov_source(transition)
    pi = source_stationary_distribution(source)
    q = transition.shape[0]
    collapsed = np.zeros((q, q), dtype=int)
    transition_logits = canonical_logits(collapsed, margin=margin).detach().cpu().numpy()
    base = np.log(pi)
    readout_logits = np.tile(base, (q, 1))
    readout_logits[1] = base + decoder_contrast
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
    return pi, pressure


def test_general_stationary_raw_and_weighted_centered_operator_laws():
    transition = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ],
        dtype=float,
    )
    contrast = np.array([0.4, -0.3, 0.2])
    kwargs = dict(horizon=11, margin=0.6, temperature=0.7)
    pi, actual = _actual_pressure(transition, contrast, **kwargs)

    np.testing.assert_allclose(
        pi, np.array([7 / 24, 13 / 24, 1 / 6]), atol=1e-12
    )
    predicted_raw = predict_stationary_raw_collapsed_pressure(
        transition, contrast, **kwargs
    )
    np.testing.assert_allclose(actual, predicted_raw, atol=1e-12, rtol=0.0)

    rescaled = actual / pi
    weighted_centered = rescaled - float(pi @ rescaled)
    predicted_centered = predict_stationary_centered_rescaled_pressure(
        transition, contrast, **kwargs
    )
    np.testing.assert_allclose(
        weighted_centered, predicted_centered, atol=1e-12, rtol=0.0
    )


def test_nonuniform_reversible_source_has_exact_predictive_eigenmode_scaling():
    transition = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.1, 0.7, 0.2],
            [0.0, 0.2, 0.8],
        ],
        dtype=float,
    )
    source = observable_markov_source(transition)
    pi = source_stationary_distribution(source)
    np.testing.assert_allclose(pi, np.array([0.2, 0.4, 0.4]), atol=1e-12)
    np.testing.assert_allclose(
        pi[:, None] * transition,
        (pi[:, None] * transition).T,
        atol=1e-12,
    )

    horizon = 9
    margin = 0.5
    temperature = 0.9
    amplitude = 0.3
    scale = collapsed_transition_pressure_scale(
        3, horizon, margin, temperature
    )

    values, vectors = np.linalg.eig(transition)
    for target in (0.8, 0.5):
        index = int(np.argmin(np.abs(values - target)))
        mode = np.real(vectors[:, index])
        assert abs(float(pi @ mode)) < 1e-12
        contrast = amplitude * mode
        _, actual = _actual_pressure(
            transition,
            contrast,
            horizon=horizon,
            margin=margin,
            temperature=temperature,
        )
        actual = actual / pi
        actual -= float(pi @ actual)
        expected = scale * target * contrast
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)
