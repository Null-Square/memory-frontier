import numpy as np

from memory_frontier.core import cesaro_limit_distribution, enumerate_deterministic_controllers


def test_cesaro_limit_handles_periodic_chain():
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    initial = np.array([1.0, 0.0])
    np.testing.assert_allclose(
        cesaro_limit_distribution(p, initial),
        np.array([0.5, 0.5]),
        atol=1e-12,
    )


def test_cesaro_limit_handles_multiple_recurrent_classes():
    # State 0 transitions equally to absorbing states 1 and 2.
    p = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    initial = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        cesaro_limit_distribution(p, initial),
        np.array([0.0, 0.5, 0.5]),
        atol=1e-12,
    )


def test_binary_two_state_controller_space_has_16_tables():
    controllers = list(enumerate_deterministic_controllers(2, 2))
    assert len(controllers) == 16
    assert len({tuple(f.ravel()) for f in controllers}) == 16
