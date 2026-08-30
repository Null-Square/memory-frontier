from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np

from memory_frontier import (
    build_hard_cell_oracle,
    controller_finite_horizon_log_loss,
    delayed_repeat_source,
    hard_cell_stability,
    one_edit_tables,
)


def _hard_loss_map(source, k: int, horizon: int) -> dict[tuple[int, ...], float]:
    out: dict[tuple[int, ...], float] = {}
    for flat in product(range(k), repeat=k * source.alphabet_size):
        table = np.asarray(flat, dtype=int).reshape(k, source.alphabet_size)
        out[tuple(flat)] = controller_finite_horizon_log_loss(
            source, table, 0, horizon
        )
    return out


def _local_minima(losses: dict[tuple[int, ...], float], k: int) -> set[tuple[int, ...]]:
    local: set[tuple[int, ...]] = set()
    for signature, base in losses.items():
        table = np.asarray(signature, dtype=int).reshape(k, -1)
        if all(
            losses[tuple(map(int, other.ravel()))] >= base - 1e-12
            for _, _, _, other in one_edit_tables(table)
        ):
            local.add(signature)
    return local


def main() -> None:
    source = delayed_repeat_source(2, 2, 0.1)
    k = 3
    horizon = 20
    steps = 180
    runs = 300
    learning_rate = 0.05
    temperature = 1.0
    init_scale = 0.25

    hard_losses = _hard_loss_map(source, k, horizon)
    best = min(hard_losses.values())
    optima = {s for s, loss in hard_losses.items() if abs(loss - best) < 1e-12}
    local_minima = _local_minima(hard_losses, k)
    stability = {
        s: hard_cell_stability(
            source, np.asarray(s, dtype=int).reshape(k, 2), horizon
        ).is_stable()
        for s in hard_losses
    }
    print(f"hard controllers={len(hard_losses)}")
    print(f"global optimum loss={best:.12f}")
    print(f"hard local minima={len(local_minima)}")
    print(f"surrogate-stable cells={sum(stability.values())}")
    print(f"intersection={sum(stability[s] and s in local_minima for s in hard_losses)}")

    cell_cache = {
        s: build_hard_cell_oracle(
            source, np.asarray(s, dtype=int).reshape(k, 2), horizon
        )
        for s in hard_losses
    }

    successes = 0
    revisits = 0
    harmful_paths = 0
    changes: list[int] = []
    failure_local = 0
    failure_stable = 0
    final_failures: Counter[tuple[int, ...]] = Counter()

    for seed in range(runs):
        rng = np.random.default_rng(10000 + seed)
        z = rng.normal(scale=init_scale, size=(k, 2, k))
        a = np.zeros((k, 2), dtype=float)
        mz = np.zeros_like(z)
        vz = np.zeros_like(z)
        ma = np.zeros_like(a)
        va = np.zeros_like(a)
        signature = tuple(map(int, z.argmax(axis=-1).ravel()))
        trajectory = [signature]

        for step in range(1, steps + 1):
            gradient = cell_cache[signature].evaluate(
                z, a, temperature=temperature
            )
            gz = gradient.transition_logit_gradient
            ga = gradient.readout_logit_gradient
            mz = 0.9 * mz + 0.1 * gz
            vz = 0.999 * vz + 0.001 * (gz * gz)
            ma = 0.9 * ma + 0.1 * ga
            va = 0.999 * va + 0.001 * (ga * ga)
            mz_hat = mz / (1.0 - 0.9**step)
            vz_hat = vz / (1.0 - 0.999**step)
            ma_hat = ma / (1.0 - 0.9**step)
            va_hat = va / (1.0 - 0.999**step)
            z -= learning_rate * mz_hat / (np.sqrt(vz_hat) + 1e-8)
            a -= learning_rate * ma_hat / (np.sqrt(va_hat) + 1e-8)

            new_signature = tuple(map(int, z.argmax(axis=-1).ravel()))
            if new_signature != signature:
                trajectory.append(new_signature)
                signature = new_signature

        successes += int(signature in optima)
        changes.append(len(trajectory) - 1)
        revisits += int(len(set(trajectory)) < len(trajectory))
        gains = [
            hard_losses[trajectory[i]] - hard_losses[trajectory[i + 1]]
            for i in range(len(trajectory) - 1)
        ]
        harmful_paths += int(any(gain < -1e-12 for gain in gains))
        if signature not in optima:
            final_failures[signature] += 1
            failure_local += int(signature in local_minima)
            failure_stable += int(stability[signature])

    failures = runs - successes
    print(f"recovery_rate={successes / runs:.6f}")
    print(f"mean_cell_changes={np.mean(changes):.6f}")
    print(f"revisit_fraction={revisits / runs:.6f}")
    print(f"harmful_path_fraction={harmful_paths / runs:.6f}")
    print(f"failure_local_min_fraction={failure_local / failures:.6f}")
    print(f"failure_surrogate_stable_fraction={failure_stable / failures:.6f}")
    print("most_common_failure_cells")
    for signature, count in final_failures.most_common(8):
        print(f"  count={count:3d} loss={hard_losses[signature]:.12f} table={signature}")


if __name__ == "__main__":
    main()
