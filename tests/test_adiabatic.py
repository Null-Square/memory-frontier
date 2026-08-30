import numpy as np

from memory_frontier import (
    adiabatic_orbit,
    adiabatic_successor,
    delayed_repeat_source,
)


def test_collapsed_controller_is_adiabatic_fixed_point():
    source = delayed_repeat_source(2, 2, 0.1)
    table = np.zeros((3, 2), dtype=int)
    step = adiabatic_successor(source, table, 20)

    assert step.is_fixed_point
    assert step.improvement_advantage == 0.0


def test_delay_two_has_exact_adiabatic_two_cycle():
    source = delayed_repeat_source(2, 2, 0.1)
    first = np.array([[0, 1], [0, 0], [2, 0]], dtype=int)
    second = np.array([[2, 1], [0, 0], [2, 0]], dtype=int)

    step_first = adiabatic_successor(source, first, 20)
    step_second = adiabatic_successor(source, second, 20)
    assert step_first.target_signature == tuple(map(int, second.ravel()))
    assert step_second.target_signature == tuple(map(int, first.ravel()))

    orbit = adiabatic_orbit(source, first, 20)
    assert not orbit.reaches_fixed_point
    assert len(orbit.cycle) == 2
    assert set(orbit.cycle) == {
        tuple(map(int, first.ravel())),
        tuple(map(int, second.ravel())),
    }
