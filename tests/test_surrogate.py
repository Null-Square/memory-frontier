import numpy as np
import pytest

pytest.importorskip("torch")

from memory_frontier import four_state_aliasing_witness, horizon_switch_witness
from memory_frontier.surrogate import (
    canonical_ste_field,
    margin_sweep,
    train_exact_distribution_ste_batch,
)


def test_witness_optimum_has_two_persistent_canonical_sign_reversals():
    source = four_state_aliasing_witness()
    table = np.array([[1, 0], [1, 0]], dtype=int)
    fields = margin_sweep(source, table, 32, [0.05, 0.25, 1.0, 4.0])
    for field in fields:
        assert abs(field.hard_loss - 0.6334255072575572) < 1e-12
        wrong = {
            (edge.memory_state, edge.symbol)
            for edge in field.edges
            if not edge.agrees
        }
        assert wrong == {(0, 0), (1, 1)}


def test_switch_long_horizon_optimum_has_aligned_canonical_field():
    source = horizon_switch_witness()
    field = canonical_ste_field(
        source, np.array([[0, 1], [0, 0]], dtype=int), 32, margin=1.0
    )
    assert field.sign_fidelity() == 1.0
    assert field.surrogate_stable()
    assert field.fully_active()


def test_batched_exact_distribution_training_is_reproducible():
    source = horizon_switch_witness()
    kwargs = dict(
        k=2,
        horizon=4,
        seeds=[0, 1, 2],
        steps=8,
        learning_rate=0.05,
    )
    first = train_exact_distribution_ste_batch(source, **kwargs)
    second = train_exact_distribution_ste_batch(source, **kwargs)
    assert [
        (run.final_signature, run.final_oracle_loss) for run in first
    ] == [
        (run.final_signature, run.final_oracle_loss) for run in second
    ]
