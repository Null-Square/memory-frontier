from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from memory_frontier import UnifilarSource, finite_horizon_hard_landscape
from memory_frontier.surrogate import train_exact_distribution_ste_batch


SEEDS = list(range(40))
HORIZON = 32
STEPS = 180
LEARNING_RATE = 0.05


manifest = json.loads(Path("data/reference_suite.json").read_text())
print(
    f"seeds={len(SEEDS)} horizon={HORIZON} steps={STEPS} "
    f"learning_rate={LEARNING_RATE}"
)

for name, entry in manifest["sources"].items():
    source = UnifilarSource(
        np.asarray(entry["emissions"], dtype=float),
        np.asarray(entry["transitions"], dtype=int),
    )
    landscape = finite_horizon_hard_landscape(
        source, 2, HORIZON, initial_memory=0
    )
    targets = {node.signature for node in landscape.global_minima()}
    runs = train_exact_distribution_ste_batch(
        source,
        2,
        HORIZON,
        SEEDS,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
        initial_memory=0,
    )
    initial_hits = sum(run.initial_signature in targets for run in runs)
    final_hits = sum(run.final_signature in targets for run in runs)
    discoveries = sum(
        run.initial_signature not in targets and run.final_signature in targets
        for run in runs
    )
    final_counts = Counter(run.final_signature for run in runs)
    print(
        f"{name:14s} target={sorted(targets)} "
        f"initial={initial_hits:2d}/40 final={final_hits:2d}/40 "
        f"new={discoveries:2d} finals={dict(final_counts)}"
    )
