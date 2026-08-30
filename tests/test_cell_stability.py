import numpy as np

from memory_frontier import (
    controller_finite_horizon_log_loss,
    delayed_repeat_source,
    hard_cell_stability,
    one_edit_tables,
)


def test_collapsed_cell_can_be_surrogate_stable_but_not_hard_local_minimum():
    source = delayed_repeat_source(2, 2, 0.1)
    table = np.zeros((3, 2), dtype=int)
    stability = hard_cell_stability(source, table, 20)

    assert stability.is_stable()
    base = controller_finite_horizon_log_loss(source, table, 0, 20)
    neighbor_losses = [
        controller_finite_horizon_log_loss(source, other, 0, 20)
        for _, _, _, other in one_edit_tables(table)
    ]
    assert min(neighbor_losses) < base - 0.02


def test_hard_local_minimum_can_be_surrogate_unstable():
    source = delayed_repeat_source(2, 2, 0.1)
    table = np.array([[0, 2], [1, 1], [1, 2]], dtype=int)
    stability = hard_cell_stability(source, table, 20)

    base = controller_finite_horizon_log_loss(source, table, 0, 20)
    neighbor_losses = [
        controller_finite_horizon_log_loss(source, other, 0, 20)
        for _, _, _, other in one_edit_tables(table)
    ]
    assert min(neighbor_losses) >= base - 1e-12
    assert not stability.is_stable()
    assert stability.max_improvement_advantage > 0.0
