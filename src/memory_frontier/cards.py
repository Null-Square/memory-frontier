from __future__ import annotations

from hashlib import sha256
import json
from math import inf

from .core import UnifilarSource
from .lab import deterministic_controller_spectrum, source_fingerprint


def finite_horizon_card(
    source: UnifilarSource,
    k: int,
    horizons: list[int] | tuple[int, ...],
) -> dict[str, object]:
    """Hashable exact reset-sequence oracle predictions for selected horizons."""
    unique_horizons = sorted(set(int(t) for t in horizons))
    if not unique_horizons or unique_horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers")

    rows: list[dict[str, object]] = []
    for horizon in unique_horizons:
        spectrum = deterministic_controller_spectrum(source, k, horizon=horizon)
        optimum = spectrum.optimum
        gap = spectrum.distinct_loss_gap()
        rows.append(
            {
                "horizon": horizon,
                "loss": optimum.loss,
                "transition_table": optimum.transition_table.tolist(),
                "initial_memory": optimum.initial_memory,
                "canonical_signature": list(optimum.signature),
                "alias_entropy": optimum.alias_entropy,
                "distinct_loss_gap": None if gap == inf else gap,
                "optimal_relabeling_class_count": len(spectrum.optimal_classes()),
            }
        )

    card: dict[str, object] = {
        "schema_version": 1,
        "source_sha256": source_fingerprint(source),
        "memory_states": k,
        "horizons": rows,
    }
    canonical = json.dumps(card, sort_keys=True, separators=(",", ":")).encode()
    card["finite_horizon_card_sha256"] = sha256(canonical).hexdigest()
    return card
