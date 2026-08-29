from memory_frontier.hashing import scientific_sha256


def test_scientific_hash_ignores_sub_precision_float_noise():
    first = {
        "source_sha256": "exact-source-id",
        "loss": 0.12345678901234,
        "algorithm": [0, 1, 0, 0],
    }
    second = {
        "source_sha256": "exact-source-id",
        "loss": 0.12345678901236,
        "algorithm": [0, 1, 0, 0],
    }
    assert scientific_sha256(first) == scientific_sha256(second)


def test_scientific_hash_changes_for_scientifically_resolved_float_change():
    first = {"source_sha256": "same", "loss": 0.123456789012}
    second = {"source_sha256": "same", "loss": 0.123456799012}
    assert scientific_sha256(first) != scientific_sha256(second)


def test_scientific_hash_keeps_discrete_algorithm_structure_exact():
    first = {"source_sha256": "same", "algorithm": [0, 1, 0, 0]}
    second = {"source_sha256": "same", "algorithm": [0, 1, 0, 1]}
    assert scientific_sha256(first) != scientific_sha256(second)
