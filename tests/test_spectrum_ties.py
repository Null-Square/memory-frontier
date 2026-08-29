import numpy as np

from memory_frontier.lab import ControllerClass, ControllerSpectrum


def _controller(signature, loss):
    return ControllerClass(
        signature=signature,
        loss=loss,
        transition_table=np.zeros((2, 2), dtype=int),
        initial_memory=0,
        occupancy=np.full((1, 2), 0.5),
        labeled_multiplicity=1,
    )


def test_optimum_uses_canonical_signature_inside_numerical_tie():
    # The lexicographically smaller algorithm is microscopically higher in raw
    # floating-point loss. It must still be the reported scientific
    # representative because the two losses are tied at the declared tolerance.
    canonical = _controller((0, 0, 0, 0, 0), 1.0 + 5e-13)
    raw_minimum = _controller((0, 1, 1, 1, 1), 1.0)
    spectrum = ControllerSpectrum(
        k=2,
        classes=(raw_minimum, canonical),
        labeled_count=2,
    )
    assert spectrum.optimum.signature == canonical.signature
    assert {c.signature for c in spectrum.optimal_classes()} == {
        canonical.signature,
        raw_minimum.signature,
    }


def test_runner_up_begins_outside_tie_tolerance():
    tied = _controller((0, 0, 0, 0, 0), 1.0 + 5e-13)
    best = _controller((0, 1, 1, 1, 1), 1.0)
    runner = _controller((1, 0, 0, 0, 0), 1.0 + 2e-12)
    spectrum = ControllerSpectrum(
        k=2,
        classes=(best, tied, runner),
        labeled_count=3,
    )
    assert spectrum.runner_up() is runner
