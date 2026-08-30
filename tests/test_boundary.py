import numpy as np

from memory_frontier import (
    build_hard_cell_oracle,
    delayed_repeat_source,
    predict_boundary_race,
)


def test_boundary_race_orders_candidates_by_linearized_crossing_time():
    logits = np.zeros((2, 1, 2), dtype=float)
    logits[:, 0, 0] = np.array([0.4, 0.8])
    gradient = np.zeros_like(logits)
    gradient[0, 0] = np.array([0.3, -0.1])
    gradient[1, 0] = np.array([0.2, -0.2])

    prediction = predict_boundary_race(logits, gradient, 0.5)
    assert prediction.winner is not None
    assert prediction.winner.memory_state == 0
    assert prediction.winner.new_target == 1
    assert prediction.winner.estimated_steps == 2.0


def test_boundary_race_predicts_first_edit_in_exact_joint_sgd_fixture():
    source = delayed_repeat_source(2, 2, 0.1)
    rng = np.random.default_rng(90002)
    k = 3
    horizon = 20
    temperature = 0.8
    transition_learning_rate = 0.5
    readout_learning_rate = 0.4
    transition_logits = rng.normal(scale=0.2, size=(k, 2, k))
    readout_logits = rng.normal(scale=0.15, size=(k, 2))
    table = transition_logits.argmax(axis=-1)
    cell = build_hard_cell_oracle(source, table, horizon)

    initial = cell.evaluate(
        transition_logits, readout_logits, temperature=temperature
    )
    prediction = predict_boundary_race(
        transition_logits,
        initial.transition_logit_gradient,
        transition_learning_rate,
    )
    assert prediction.winner is not None
    expected = (
        prediction.winner.memory_state,
        prediction.winner.symbol,
        prediction.winner.new_target,
    )

    original = table.copy()
    for _ in range(100):
        gradient = cell.evaluate(
            transition_logits, readout_logits, temperature=temperature
        )
        transition_logits -= (
            transition_learning_rate * gradient.transition_logit_gradient
        )
        readout_logits -= readout_learning_rate * gradient.readout_logit_gradient
        new_table = transition_logits.argmax(axis=-1)
        if not np.array_equal(new_table, original):
            changed = np.argwhere(new_table != original)
            assert len(changed) == 1
            m, x = changed[0]
            actual = (int(m), int(x), int(new_table[m, x]))
            assert actual == expected
            return
    raise AssertionError("fixture did not cross a hard boundary")
