from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .accessibility import gradient_accessibility_operator
from .core import UnifilarSource
from .exact_gradient import hard_value_gradient
from .landscape import one_edit_tables


def _softmax_vector(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1:
        raise ValueError("common_readout_logits must be one-dimensional")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / weights.sum()


def _symmetric_decoder_gradient(
    source: UnifilarSource,
    table: np.ndarray,
    common_readout_logits: np.ndarray,
    horizon: int,
    initial_memory: int,
) -> np.ndarray:
    k, alphabet_size = table.shape
    readout = _softmax_vector(common_readout_logits)
    hard = hard_value_gradient(
        source,
        table,
        np.tile(readout[None, :], (k, 1)),
        horizon,
        initial_memory,
    )
    mass = hard.readout_symbol_mass
    return mass.sum(axis=-1, keepdims=True) * readout[None, :] - mass


@dataclass(frozen=True)
class EditDirection:
    memory_state: int
    symbol: int
    old_target: int
    new_target: int


@dataclass(frozen=True)
class EditAlignmentOperators:
    """Linearized STE pressure and exact hard-edit gain for the same edits."""

    edges: tuple[EditDirection, ...]
    pressure_matrix: np.ndarray
    hard_gain_matrix: np.ndarray
    readout_shape: tuple[int, int]

    def _direction(self, readout_logit_direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(readout_logit_direction, dtype=float)
        if direction.shape != self.readout_shape:
            raise ValueError("readout_logit_direction has wrong shape")
        return direction.ravel()

    def directional_pressure(self, readout_logit_direction: np.ndarray) -> np.ndarray:
        return self.pressure_matrix @ self._direction(readout_logit_direction)

    def directional_hard_gain(self, readout_logit_direction: np.ndarray) -> np.ndarray:
        return self.hard_gain_matrix @ self._direction(readout_logit_direction)

    def projected_matrices(self, decoder_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        basis = np.asarray(decoder_basis, dtype=float)
        input_dim = int(np.prod(self.readout_shape))
        if basis.ndim != 2 or basis.shape[0] != input_dim:
            raise ValueError("decoder_basis must have shape (input_dim, subspace_dim)")
        return self.pressure_matrix @ basis, self.hard_gain_matrix @ basis

    def frobenius_alignment(self, decoder_basis: np.ndarray | None = None) -> float:
        if decoder_basis is None:
            pressure, gain = self.pressure_matrix, self.hard_gain_matrix
        else:
            pressure, gain = self.projected_matrices(decoder_basis)
        denom = float(np.linalg.norm(pressure) * np.linalg.norm(gain))
        if denom <= 1e-15:
            return float("nan")
        return float(np.sum(pressure * gain) / denom)

    def directional_sign_fidelity(
        self,
        readout_logit_direction: np.ndarray,
        *,
        atol: float = 1e-12,
    ) -> float:
        pressure = self.directional_pressure(readout_logit_direction)
        gain = self.directional_hard_gain(readout_logit_direction)
        mask = (np.abs(pressure) > atol) & (np.abs(gain) > atol)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.sign(pressure[mask]) == np.sign(gain[mask])))


def edit_alignment_operators(
    source: UnifilarSource,
    transition_logits: np.ndarray,
    common_readout_logits: np.ndarray,
    horizon: int,
    *,
    initial_memory: int = 0,
    temperature: float = 1.0,
) -> EditAlignmentOperators:
    """Construct first-order surrogate-pressure and exact-hard-gain maps.

    For decoder-logit perturbation ``D`` and one-edit alternative ``F_e``:

        pressure_e(eps) = eps * (P_F D)_e + O(eps^2)
        [L(F)-L(F_e)](eps) = eps * (H_F D)_e + O(eps^2)

    at a decoder-symmetric base point. ``P_F`` comes from the GAO. ``H_F`` is
    the exact derivative of the finite hard-edit loss gain.
    """
    logits = np.asarray(transition_logits, dtype=float)
    if logits.ndim != 3:
        raise ValueError("transition_logits must have shape (k, alphabet_size, k)")
    k, alphabet_size, k2 = logits.shape
    if k != k2 or alphabet_size != source.alphabet_size:
        raise ValueError("transition logit shape mismatch")
    common = np.asarray(common_readout_logits, dtype=float)
    if common.shape != (alphabet_size,):
        raise ValueError("common_readout_logits has wrong shape")

    table = logits.argmax(axis=-1)
    gao = gradient_accessibility_operator(
        source,
        logits,
        common,
        horizon,
        initial_memory=initial_memory,
        temperature=temperature,
    )
    base_decoder_gradient = _symmetric_decoder_gradient(
        source, table, common, horizon, initial_memory
    )

    edges: list[EditDirection] = []
    pressure_rows: list[np.ndarray] = []
    gain_rows: list[np.ndarray] = []
    output_shape = (k, alphabet_size, k)
    for m, x, new_target, other in one_edit_tables(table):
        old_target = int(table[m, x])
        old_index = np.ravel_multi_index((m, x, old_target), output_shape)
        new_index = np.ravel_multi_index((m, x, new_target), output_shape)
        pressure_rows.append(gao.matrix[old_index] - gao.matrix[new_index])
        other_decoder_gradient = _symmetric_decoder_gradient(
            source, other, common, horizon, initial_memory
        )
        gain_rows.append(
            (base_decoder_gradient - other_decoder_gradient).ravel()
        )
        edges.append(
            EditDirection(
                memory_state=m,
                symbol=x,
                old_target=old_target,
                new_target=new_target,
            )
        )

    return EditAlignmentOperators(
        edges=tuple(edges),
        pressure_matrix=np.stack(pressure_rows),
        hard_gain_matrix=np.stack(gain_rows),
        readout_shape=(k, alphabet_size),
    )
