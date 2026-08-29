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


def horizon_switch_witness() -> UnifilarSource:
    """Frozen reference source whose K=2 finite-horizon optimum changes with T.

    It is the ``horizon_switch`` entry in ``data/reference_suite.json``. With
    controller reset state 0, the exact optimal hard table is [[0,1],[0,1]] at
    T=4 and [[0,1],[0,0]] at T=32.
    """
    emissions = np.array(
        [
            [0.1744377091152322, 0.8255622908847677],
            [0.7532082665358947, 0.2467917334641055],
            [0.570424639288503, 0.4295753607114971],
            [0.36152540181608567, 0.6384745981839145],
        ],
        dtype=float,
    )
    transitions = np.array(
        [
            [0, 1],
            [1, 3],
            [1, 0],
            [1, 2],
        ],
        dtype=int,
    )
    return UnifilarSource(emissions, transitions)
