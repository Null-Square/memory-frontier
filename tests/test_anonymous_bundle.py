from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "build_anonymous_bundle.py"


def _module():
    spec = importlib.util.spec_from_file_location("anonymous_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_manifest_is_whitelist_based_and_excludes_development_material():
    module = _module()
    files = module.collect_export_files(ROOT)
    names = {path.as_posix() for path in files}

    assert "pyproject.toml" in names
    assert "experiments/forward_equivalence_order_census.py" in names
    assert "experiments/linear_ssm_validation.py" in names
    assert "experiments/paper_figures.py" in names
    assert any(name.startswith("src/memory_frontier/") for name in names)
    assert any(name.startswith("tests/") for name in names)
    assert "data/reference_suite.json" in names

    assert not any(name.startswith("paper/") for name in names)
    assert not any(name.startswith("docs/") for name in names)
    assert not any(name.startswith(".git") for name in names)
    assert not any(".egg-info/" in name for name in names)
    assert "README.md" not in names
    assert "experiments/build_anonymous_bundle.py" not in names
    assert "tests/test_anonymous_bundle.py" not in names


def test_identity_scanner_detects_generic_and_custom_markers():
    module = _module()

    assert "email address" in module.identity_leaks("Contact person@example.org")
    assert "public GitHub URL" in module.identity_leaks(
        "See https://github.com/example/project"
    )
    assert "GitHub SSH URL" in module.identity_leaks("git@github.com:example/x.git")
    assert "local Unix home path" in module.identity_leaks("/home/alice/project/file")

    findings = module.identity_leaks(
        "The hidden-owner marker occurs here.", deny_markers=("Hidden-Owner",)
    )
    assert any("deny marker" in finding for finding in findings)


def test_current_whitelisted_repository_payload_passes_generic_identity_audit():
    module = _module()
    files = module.collect_export_files(ROOT)
    module.audit_export(ROOT, files, deny_markers=())


def test_bundle_is_deterministic_and_contains_verified_manifest(tmp_path: Path):
    module = _module()
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    _, files_first = module.build_bundle(ROOT, first)
    _, files_second = module.build_bundle(ROOT, second)

    assert files_first == files_second
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names[:-1]) + ["submission_code/MANIFEST.sha256"]
        assert "submission_code/README.md" in names
        assert "submission_code/pyproject.toml" in names
        assert "submission_code/MANIFEST.sha256" in names
        assert not any("paper/" in name for name in names)
        assert not any("docs/" in name for name in names)
        assert not any(".egg-info/" in name for name in names)
        assert "submission_code/tests/test_anonymous_bundle.py" not in names

        manifest = archive.read("submission_code/MANIFEST.sha256").decode("utf-8")
        for line in manifest.strip().splitlines():
            expected_digest, archived_name = line.split("  ", 1)
            payload = archive.read(archived_name)
            assert hashlib.sha256(payload).hexdigest() == expected_digest

        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_bundle_refuses_custom_identity_marker(tmp_path: Path):
    module = _module()
    marker = "from __future__ import annotations"
    with pytest.raises(RuntimeError, match="deny marker"):
        module.build_bundle(ROOT, tmp_path / "blocked.zip", deny_markers=(marker,))
