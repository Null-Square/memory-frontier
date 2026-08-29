from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from memory_frontier import UnifilarSource, finite_horizon_hard_landscape
from memory_frontier.readout_prior import (
    train_exact_distribution_ste_with_readout_prior,
)


SEEDS = list(range(40))
HORIZON = 32
STEPS = 120
LEARNING_RATE = 0.05
MODES = ("uniform", "source_marginal", "random")

# These experiments consist of many tiny tensor operations. A single CPU thread
# is substantially faster and does not alter the mathematical objective.
torch.set_num_threads(1)

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
    print(f"\n{name}: targets={sorted(targets)}")
    for mode in MODES:
        runs = train_exact_distribution_ste_with_readout_prior(
            source,
            2,
            HORIZON,
            SEEDS,
            readout_initialization=mode,
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
        collapsed = [
            run for run in runs if len(set(run.initial_signature)) == 1
        ]
        collapsed_hits = sum(
            run.final_signature in targets for run in collapsed
        )
        final_counts = Counter(run.final_signature for run in runs)
        print(
            f"  {mode:15s} initial={initial_hits:2d}/{len(runs)} "
            f"final={final_hits:2d}/{len(runs)} new={discoveries:2d} "
            f"constant-target={collapsed_hits:2d}/{len(collapsed):2d} "
            f"finals={dict(final_counts)}"
        )
