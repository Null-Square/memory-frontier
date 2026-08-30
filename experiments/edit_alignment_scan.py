from __future__ import annotations

import argparse

import numpy as np

from memory_frontier import delayed_repeat_source, edit_alignment_operators


def canonical_logits(table: np.ndarray, margin: float) -> np.ndarray:
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def unreachable_contrast_basis(k: int) -> np.ndarray:
    basis = np.zeros((k - 1, k, 2), dtype=float)
    contrast = np.array([1.0, -1.0]) / np.sqrt(2.0)
    for m in range(1, k):
        basis[m - 1, m] = contrast
    return basis.reshape(k - 1, -1).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--topologies", type=int, default=180)
    parser.add_argument("--directions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=260835)
    args = parser.parse_args()

    if args.delay < 2:
        raise ValueError("delay must be at least 2")
    k = args.delay + 1
    source = delayed_repeat_source(2, args.delay, 0.1)
    decoder_basis = unreachable_contrast_basis(k)
    rng = np.random.default_rng(args.seed)

    alignments: list[float] = []
    fidelities: list[float] = []
    selected_useful: list[float] = []
    seen: set[tuple[int, ...]] = set()
    count = 0
    while count < args.topologies:
        table = np.zeros((k, 2), dtype=int)
        table[1:] = rng.integers(0, k, size=(k - 1, 2))
        signature = tuple(map(int, table[1:].ravel()))
        if signature in seen:
            continue
        seen.add(signature)

        operators = edit_alignment_operators(
            source,
            canonical_logits(table, 0.7),
            np.zeros(2),
            20,
            temperature=0.8,
        )
        alignment = operators.frobenius_alignment(decoder_basis)
        alignments.append(alignment)
        pressure_matrix, gain_matrix = operators.projected_matrices(decoder_basis)

        for direction_index in range(args.directions):
            local_rng = np.random.default_rng(
                600000 + count * 20 + direction_index
            )
            coefficients = local_rng.normal(size=k - 1)
            coefficients /= np.linalg.norm(coefficients)
            pressure = pressure_matrix @ coefficients
            gain = gain_matrix @ coefficients
            mask = (np.abs(pressure) > 1e-12) & (np.abs(gain) > 1e-12)
            if np.any(mask):
                fidelities.append(
                    float(np.mean(np.sign(pressure[mask]) == np.sign(gain[mask])))
                )
            selected_useful.append(float(gain[int(np.argmax(pressure))] > 0.0))
        count += 1

    values = np.asarray(alignments, dtype=float)
    print(f"topologies={len(values)}")
    print(f"median_alignment={np.nanmedian(values):.6f}")
    print(f"negative_alignment_fraction={np.nanmean(values < 0):.6f}")
    print(f"median_directional_sign_fidelity={np.nanmedian(fidelities):.6f}")
    print(f"top_pressure_edit_predicted_useful_fraction={np.mean(selected_useful):.6f}")


if __name__ == "__main__":
    main()
