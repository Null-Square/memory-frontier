import numpy as np
import pytest

from memory_frontier import (
    build_hard_cell_oracle,
    delayed_repeat_source,
    exact_ste_gradient,
)


def test_hard_cell_matches_exact_gradient_random_logits():
    source = delayed_repeat_source(2, 2, 0.1)
    rng = np.random.default_rng(260836)
    for _ in range(12):
        k = 3
        transition_logits = rng.normal(size=(k, 2, k))
        readout_logits = rng.normal(size=(k, 2))
        horizon = int(rng.integers(3, 12))
        temperature = float(rng.uniform(0.4, 1.8))

        exact = exact_ste_gradient(
            source,
            transition_logits,
            readout_logits,
            horizon,
            temperature=temperature,
        )
        cell = build_hard_cell_oracle(
            source, transition_logits.argmax(axis=-1), horizon
        )
        cached = cell.evaluate(
            transition_logits,
            readout_logits,
            temperature=temperature,
        )

        assert abs(cached.loss - exact.loss) < 1e-12
        assert np.max(np.abs(cached.transition_tensor_gradient - exact.transition_tensor_gradient)) < 1e-12
        assert np.max(np.abs(cached.transition_logit_gradient - exact.transition_logit_gradient)) < 1e-12
        assert np.max(np.abs(cached.readout_logit_gradient - exact.readout_logit_gradient)) < 1e-12


def test_cached_training_matches_full_recomputation_across_cell_crossings():
    source = delayed_repeat_source(2, 2, 0.1)
    rng = np.random.default_rng(0)
    k = 3
    horizon = 10
    temperature = 0.8
    transition_learning_rate = 3.0
    readout_learning_rate = 0.4

    transition_full = rng.normal(scale=0.2, size=(k, 2, k))
    readout_full = rng.normal(scale=0.2, size=(k, 2))
    transition_cached = transition_full.copy()
    readout_cached = readout_full.copy()

    cell = build_hard_cell_oracle(source, transition_cached.argmax(axis=-1), horizon)
    crossings = 0
    for _ in range(50):
        full = exact_ste_gradient(
            source, transition_full, readout_full, horizon, temperature=temperature
        )
        cached = cell.evaluate(
            transition_cached, readout_cached, temperature=temperature
        )

        transition_full -= transition_learning_rate * full.transition_logit_gradient
        readout_full -= readout_learning_rate * full.readout_logit_gradient
        transition_cached -= transition_learning_rate * cached.transition_logit_gradient
        readout_cached -= readout_learning_rate * cached.readout_logit_gradient

        if not cell.contains(transition_cached):
            crossings += 1
            cell = build_hard_cell_oracle(
                source, transition_cached.argmax(axis=-1), horizon
            )

        assert np.max(np.abs(transition_full - transition_cached)) < 1e-12
        assert np.max(np.abs(readout_full - readout_cached)) < 1e-12

    assert crossings >= 4


def test_hard_cell_rejects_logits_from_another_cell():
    source = delayed_repeat_source(2, 2, 0.1)
    table = np.zeros((3, 2), dtype=int)
    cell = build_hard_cell_oracle(source, table, 8)
    logits = np.zeros((3, 2, 3), dtype=float)
    logits[..., 0] = 1.0
    logits[0, 0, 1] = 2.0

    assert not cell.contains(logits)
    with pytest.raises(ValueError, match="outside"):
        cell.evaluate(logits, np.zeros((3, 2)))
