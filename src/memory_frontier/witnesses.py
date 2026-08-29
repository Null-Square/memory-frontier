from __future__ import annotations

import numpy as np

from .core import UnifilarSource


def four_state_aliasing_witness() -> UnifilarSource:
    """Four-state binary source exhibiting strict static/online/quotient gaps.

    State order is A, B, C, D. Symbol order is 0, 1.
    """
    emissions = np.array(
        [
            [0.2, 0.8],  # A
            [0.7, 0.3],  # B
            [0.9, 0.1],  # C
            [0.1, 0.9],  # D
        ],
        dtype=float,
    )
    transitions = np.array(
        [
            [3, 0],  # A: 0->D, 1->A
            [2, 0],  # B: 0->C, 1->A
            [1, 0],  # C: 0->B, 1->A
            [2, 2],  # D: 0->C, 1->C
        ],
        dtype=int,
    )
    return UnifilarSource(emissions, transitions)
