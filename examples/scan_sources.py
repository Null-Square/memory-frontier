from __future__ import annotations

import argparse
import math

import numpy as np

from memory_frontier.lab import random_synchronizing_source, theory_card


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan exact finite-memory frontiers on random synchronized sources")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--memory", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cards = []
    for _ in range(args.count):
        source = random_synchronizing_source(args.states, rng=rng)
        cards.append(theory_card(source, args.memory))

    tau = np.asarray([c["static_to_deterministic_gap"] for c in cards])
    alpha = np.asarray([c["deterministic_to_quotient_gap"] for c in cards])
    alias = np.asarray([c["deterministic_alias_entropy"] for c in cards])
    distinct_gap = np.asarray([c["distinct_loss_gap"] or math.inf for c in cards])

    print(f"sources={args.count} states={args.states} memory={args.memory} seed={args.seed}")
    print(f"median realizability gap: {np.median(tau):.6f}")
    print(f"median aliasing advantage: {np.median(alpha):.6f}")
    print(f"fraction tau > .02: {np.mean(tau > .02):.3f}")
    print(f"fraction alpha > .02: {np.mean(alpha > .02):.3f}")
    print(f"fraction both > .02: {np.mean((tau > .02) & (alpha > .02)):.3f}")
    print(f"fraction alias entropy > 1e-8: {np.mean(alias > 1e-8):.3f}")
    print(f"median distinct-loss gap: {np.median(distinct_gap[np.isfinite(distinct_gap)]):.6f}")

    if np.any((alpha > 1e-8) & (alias <= 1e-8)):
        raise RuntimeError("found positive quotient advantage without history-dependent aliasing")


if __name__ == "__main__":
    main()
