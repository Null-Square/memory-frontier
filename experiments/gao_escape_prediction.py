from __future__ import annotations

import argparse

import numpy as np

from memory_frontier import (
    counterfactual_accessibility_operator,
    delayed_repeat_source,
    exact_ste_gradient,
    gradient_accessibility_operator,
    transition_logit_gradient_from_tensor_gradient,
)


def canonical_logits(table: np.ndarray, margin: float) -> np.ndarray:
    table = np.asarray(table, dtype=int)
    k, alphabet_size = table.shape
    logits = np.zeros((k, alphabet_size, k), dtype=float)
    for m in range(k):
        for x in range(alphabet_size):
            logits[m, x, table[m, x]] = margin
    return logits


def random_forward_equivalent_table(k: int, rng: np.random.Generator) -> np.ndarray:
    table = np.zeros((k, 2), dtype=int)
    table[1:] = rng.integers(0, k, size=(k - 1, 2))
    return table


def unreachable_contrast_basis(k: int) -> np.ndarray:
    basis = np.zeros((k - 1, k, 2), dtype=float)
    contrast = np.array([1.0, -1.0]) / np.sqrt(2.0)
    for m in range(1, k):
        basis[m - 1, m] = contrast
    return basis


def restricted_gao_matrix(operator, basis: np.ndarray) -> np.ndarray:
    columns = [operator.apply(direction).ravel() for direction in basis]
    return np.stack(columns, axis=1)


def directional_reachable_pressure(response: np.ndarray, current_target: int = 0) -> float:
    k, alphabet_size, _ = response.shape
    return max(
        float(response[0, x, current_target] - response[0, x, alternative])
        for x in range(alphabet_size)
        for alternative in range(k)
        if alternative != current_target
    )


def first_reachable_escape(
    source,
    table: np.ndarray,
    readout_logits: np.ndarray,
    *,
    horizon: int,
    margin: float,
    temperature: float,
    learning_rate: float,
    max_steps: int,
) -> tuple[int | None, float]:
    """Exact transition-only SGD until reachable row 0 changes hard target."""
    logits = canonical_logits(table, margin)
    initial = exact_ste_gradient(
        source,
        logits,
        readout_logits,
        horizon,
        temperature=temperature,
    )
    tensor_gradient = initial.transition_tensor_gradient
    initial_row = table[0].copy()
    for step in range(1, max_steps + 1):
        gradient = transition_logit_gradient_from_tensor_gradient(
            logits, tensor_gradient, temperature
        )
        logits[0] -= learning_rate * gradient[0]
        if not np.array_equal(logits[0].argmax(axis=-1), initial_row):
            final = exact_ste_gradient(
                source,
                logits,
                readout_logits,
                horizon,
                temperature=temperature,
            )
            return step, final.loss
    return None, initial.loss


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--topologies", type=int, default=320)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260830)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.delay < 2:
        raise ValueError("delay must be at least 2")
    k = args.delay + 1
    source = delayed_repeat_source(2, args.delay, 0.1)
    rng = np.random.default_rng(args.seed)
    basis = unreachable_contrast_basis(k)

    rows: list[tuple[float, ...]] = []
    seen: set[tuple[int, ...]] = set()
    topology_index = 0
    while topology_index < args.topologies:
        table = random_forward_equivalent_table(k, rng)
        signature = tuple(map(int, table[1:].ravel()))
        if signature in seen:
            continue
        seen.add(signature)

        logits = canonical_logits(table, args.margin)
        intrinsic = counterfactual_accessibility_operator(
            source, table, np.zeros(2), args.horizon
        )
        gao = gradient_accessibility_operator(
            source,
            logits,
            np.zeros(2),
            args.horizon,
            temperature=args.temperature,
        )
        restricted = restricted_gao_matrix(gao, basis)
        singular_values = np.linalg.svd(restricted, compute_uv=False)
        restricted_rank = int(
            np.count_nonzero(
                singular_values
                > max(
                    1e-12,
                    1e-9 * singular_values[0]
                    if len(singular_values)
                    else 0.0,
                )
            )
        )
        frobenius = float(np.linalg.norm(restricted, "fro"))
        leading = float(singular_values[0]) if len(singular_values) else 0.0

        for direction_index in range(args.directions):
            local_rng = np.random.default_rng(
                900000 + topology_index * 100 + direction_index
            )
            coefficients = local_rng.normal(size=k - 1)
            coefficients /= np.linalg.norm(coefficients)
            readout_direction = np.einsum("j,jmx->mx", coefficients, basis)
            predicted_response = gao.apply(readout_direction)
            predicted_pressure = directional_reachable_pressure(predicted_response)

            readout_logits = args.epsilon * readout_direction
            initial = exact_ste_gradient(
                source,
                logits,
                readout_logits,
                args.horizon,
                temperature=args.temperature,
            )
            actual_pressure = directional_reachable_pressure(
                initial.transition_logit_gradient
            ) / args.epsilon
            escape_step, final_loss = first_reachable_escape(
                source,
                table,
                readout_logits,
                horizon=args.horizon,
                margin=args.margin,
                temperature=args.temperature,
                learning_rate=args.learning_rate,
                max_steps=args.steps,
            )
            censored_step = args.steps + 1 if escape_step is None else escape_step
            rows.append(
                (
                    float(intrinsic.numerical_rank()),
                    float(restricted_rank),
                    frobenius,
                    leading,
                    predicted_pressure,
                    actual_pressure,
                    float(censored_step),
                    float(escape_step is not None),
                    final_loss,
                )
            )
        topology_index += 1

    data = np.asarray(rows)
    success = data[:, 7]
    reciprocal_censored_step = 1.0 / data[:, 6]
    print(f"runs={len(data)} escape_rate={success.mean():.6f}")
    print(
        "pressure_vs_reciprocal_censored_escape="
        f"{correlation(data[:, 4], reciprocal_censored_step):.6f}"
    )
    print(
        "pressure_vs_escape_indicator="
        f"{correlation(data[:, 4], success):.6f}"
    )
    print(
        "linear_vs_finite_epsilon_pressure="
        f"{correlation(data[:, 4], data[:, 5]):.6f}"
    )
    print(
        "frobenius_vs_reciprocal_censored_escape="
        f"{correlation(data[:, 2], reciprocal_censored_step):.6f}"
    )
    for rank in sorted(set(data[:, 1].astype(int))):
        subset = data[data[:, 1] == rank]
        escaped = subset[subset[:, 7] == 1]
        median_step = float(np.median(escaped[:, 6])) if len(escaped) else float("nan")
        print(
            f"restricted_rank={rank} runs={len(subset)} "
            f"escape_rate={subset[:, 7].mean():.6f} "
            f"median_escape_step={median_step:.3f}"
        )


if __name__ == "__main__":
    main()
