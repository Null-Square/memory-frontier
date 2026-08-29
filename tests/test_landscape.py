from memory_frontier import (
    finite_horizon_hard_landscape,
    four_state_aliasing_witness,
    horizon_switch_witness,
)


def test_binary_k2_hard_landscape_has_expected_size_and_witness_minima():
    landscape = finite_horizon_hard_landscape(
        four_state_aliasing_witness(), 2, 32, initial_memory=0
    )
    assert len(landscape.nodes) == 16
    assert len(landscape.edges) == 64
    assert [node.signature for node in landscape.global_minima()] == [(1, 0, 1, 0)]
    assert {node.signature for node in landscape.local_minima()} == {
        (0, 1, 0, 1),
        (1, 0, 1, 0),
    }
    assert abs(landscape.best_loss() - 0.6334255072575572) < 1e-12


def test_horizon_switch_reference_changes_exact_optimal_algorithm():
    source = horizon_switch_witness()
    short = finite_horizon_hard_landscape(source, 2, 4, initial_memory=0)
    long = finite_horizon_hard_landscape(source, 2, 32, initial_memory=0)
    assert [node.signature for node in short.global_minima()] == [(0, 1, 0, 1)]
    assert [node.signature for node in long.global_minima()] == [(0, 1, 0, 0)]
    assert abs(short.best_loss() - 0.627087323901246) < 1e-12
    assert abs(long.best_loss() - 0.6074525777599251) < 1e-12
