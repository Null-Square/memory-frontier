import numpy as np

from memory_frontier.lab import (
    canonical_controller_signature,
    deterministic_controller_spectrum,
    find_synchronizing_word,
    is_strongly_connected,
    random_synchronizing_source,
    relabel_controller,
    theory_card,
)
from memory_frontier.witnesses import four_state_aliasing_witness


def test_witness_has_short_reset_word():
    source = four_state_aliasing_witness()
    assert find_synchronizing_word(source) == (1, 1)


def test_random_source_is_valid_and_reproducible():
    a = random_synchronizing_source(4, rng=123)
    b = random_synchronizing_source(4, rng=123)
    np.testing.assert_allclose(a.emissions, b.emissions)
    np.testing.assert_array_equal(a.transitions, b.transitions)
    assert is_strongly_connected(a)
    assert find_synchronizing_word(a) is not None
    assert np.all(a.emissions > 0)


def test_controller_signature_is_invariant_to_memory_relabeling():
    f = np.array([[0, 1], [0, 1]])
    sig = canonical_controller_signature(f, 0)
    renamed, m0 = relabel_controller(f, 0, (1, 0))
    assert canonical_controller_signature(renamed, m0) == sig


def test_witness_spectrum_reduces_labeled_controllers_to_relabeling_classes():
    source = four_state_aliasing_witness()
    spectrum = deterministic_controller_spectrum(source, 2)
    assert spectrum.labeled_count == 32  # 16 tables x 2 reset states
    assert len(spectrum.classes) < 32
    assert abs(spectrum.optimum.loss - 0.6297909185) < 1e-9
    assert spectrum.optimum.alias_entropy > 0
    assert spectrum.distinct_loss_gap() > 0.05


def test_theory_card_records_the_pretraining_oracle():
    card = theory_card(four_state_aliasing_witness(), 2)
    assert card["synchronizing_word"] == [1, 1]
    assert abs(card["deterministic_loss"] - 0.6297909185) < 1e-9
    assert card["static_to_deterministic_gap"] > 0
    assert card["deterministic_to_quotient_gap"] > 0
    assert card["optimal_relabeling_class_count"] >= 1


def test_synchronizing_sources_obey_expected_frontier_order_on_fixed_scan():
    # This is not the proof; it is a regression check for the theorem's domain.
    for seed in range(12):
        source = random_synchronizing_source(4, rng=1000 + seed)
        card = theory_card(source, 2)
        assert card["static_loss"] <= card["deterministic_loss"] + 1e-10
        assert card["deterministic_loss"] <= card["quotient_loss"] + 1e-10


def test_distinct_loss_gap_ignores_exact_ties():
    spectrum = deterministic_controller_spectrum(four_state_aliasing_witness(), 2)
    levels = spectrum.distinct_loss_levels()
    assert len(levels) >= 2
    assert abs(levels[0] - 0.6297909185) < 1e-9
    assert spectrum.distinct_loss_gap() > 0.05


def test_theory_card_hash_is_reproducible():
    a = theory_card(four_state_aliasing_witness(), 2)
    b = theory_card(four_state_aliasing_witness(), 2)
    assert a["theory_card_sha256"] == b["theory_card_sha256"]
    assert len(a["theory_card_sha256"]) == 64
