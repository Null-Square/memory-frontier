from __future__ import annotations

import argparse

import numpy as np

from memory_frontier import (
    adiabatic_successor,
    build_hard_cell_oracle,
    delayed_repeat_source,
    predict_boundary_race,
)


def edge_between(old: np.ndarray, new: np.ndarray):
    changed = np.argwhere(old != new)
    if len(changed) != 1:
        return None
    m, x = changed[0]
    return int(m), int(x), int(new[m, x])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=80000)
    args = parser.parse_args()

    delay = args.delay
    if delay < 2:
        raise ValueError("delay must be at least 2")
    source = delayed_repeat_source(2, delay, 0.1)
    k = delay + 1
    horizon = 20
    temperature = 0.8
    transition_learning_rate = 0.5
    readout_learning_rate = 0.4
    cache: dict[tuple[int, ...], object] = {}

    def cell_for(table: np.ndarray):
        signature = tuple(map(int, table.ravel()))
        if signature not in cache:
            cache[signature] = build_hard_cell_oracle(source, table, horizon)
        return cache[signature]

    single_events = 0
    covered = 0
    correct = 0
    adiabatic_correct = 0
    predicted_times: list[float] = []
    actual_times: list[float] = []

    for run in range(args.runs):
        rng = np.random.default_rng(args.seed + run)
        z = rng.normal(scale=0.2, size=(k, 2, k))
        a = np.zeros((k, 2), dtype=float)
        table = z.argmax(axis=-1)
        cell = cell_for(table)
        gradient = cell.evaluate(z, a, temperature=temperature)
        race = predict_boundary_race(
            z,
            gradient.transition_logit_gradient,
            transition_learning_rate,
        )
        winner = race.winner
        adiabatic = adiabatic_successor(source, table, horizon).target_signature
        entry_step = 0

        for step in range(1, args.steps + 1):
            gradient = cell.evaluate(z, a, temperature=temperature)
            z -= transition_learning_rate * gradient.transition_logit_gradient
            a -= readout_learning_rate * gradient.readout_logit_gradient
            new_table = z.argmax(axis=-1)
            if np.array_equal(new_table, table):
                continue

            actual = edge_between(table, new_table)
            if actual is not None:
                single_events += 1
                if winner is not None:
                    covered += 1
                    predicted = (
                        winner.memory_state,
                        winner.symbol,
                        winner.new_target,
                    )
                    if predicted == actual:
                        correct += 1
                        predicted_times.append(winner.estimated_steps)
                        actual_times.append(step - entry_step)
                adiabatic_correct += int(
                    tuple(map(int, new_table.ravel())) == adiabatic
                )

            table = new_table.copy()
            cell = cell_for(table)
            gradient = cell.evaluate(z, a, temperature=temperature)
            race = predict_boundary_race(
                z,
                gradient.transition_logit_gradient,
                transition_learning_rate,
            )
            winner = race.winner
            adiabatic = adiabatic_successor(
                source, table, horizon
            ).target_signature
            entry_step = step

    print(f"delay={delay} K={k}")
    print(f"single_edit_events={single_events}")
    print(f"coverage={covered / single_events:.6f}")
    print(f"conditional_edge_accuracy={correct / covered:.6f}")
    print(f"overall_edge_accuracy={correct / single_events:.6f}")
    print(f"adiabatic_edge_accuracy={adiabatic_correct / single_events:.6f}")
    if len(actual_times) > 1:
        log_corr = np.corrcoef(
            np.log(np.asarray(actual_times, dtype=float)),
            np.log(np.asarray(predicted_times, dtype=float)),
        )[0, 1]
        ratio = np.median(
            np.asarray(actual_times, dtype=float)
            / np.asarray(predicted_times, dtype=float)
        )
        print(f"log_residence_time_correlation={log_corr:.6f}")
        print(f"median_actual_over_predicted_time={ratio:.6f}")
    print(f"unique_cells_built={len(cache)}")


if __name__ == "__main__":
    main()
