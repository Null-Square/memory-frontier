import json
from pathlib import Path

import numpy as np

from memory_frontier.cards import finite_horizon_card
from memory_frontier.core import UnifilarSource
from memory_frontier.lab import theory_card


def test_frozen_reference_suite_recomputes_exact_hashes():
    manifest = json.loads(Path("data/reference_suite.json").read_text())
    for name, entry in manifest["sources"].items():
        source = UnifilarSource(
            np.asarray(entry["emissions"], dtype=float),
            np.asarray(entry["transitions"], dtype=int),
        )
        card = theory_card(source, 2)
        assert card["source_sha256"] == entry["source_sha256"], name
        assert card["theory_card_sha256"] == entry["theory_card_sha256"], name

        if name == "horizon_switch":
            finite = finite_horizon_card(source, 2, list(range(1, 33)))
            assert finite["finite_horizon_card_sha256"] == entry["finite_horizon_card_sha256"]
            rows = {row["horizon"]: row for row in finite["horizons"]}
            assert rows[4]["canonical_signature"] != rows[32]["canonical_signature"]
            assert rows[4]["distinct_loss_gap"] >= 0.003
            assert rows[32]["distinct_loss_gap"] >= 0.003
