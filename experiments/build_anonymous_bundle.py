"""Build the anonymous supplementary-code ZIP used for double-blind review.

The exporter is deliberately whitelist-based.  It includes the installable
package, exact regression suite, reference data, and the three presentation/
validation experiments needed by the paper.  Development history, paper sources,
docs, Git metadata, and repository README files are excluded.

The output is deterministic: files are sorted, ZIP timestamps/permissions are
fixed, and a SHA-256 manifest is embedded.  Before writing the archive, every
text file is scanned for obvious identity leaks (emails, public GitHub links,
local absolute home paths) plus any caller-provided deny markers.

Usage
-----
python experiments/build_anonymous_bundle.py \
    --output dist/anonymous_submission_code.zip \
    --deny-marker "<project-owner-or-author-marker>"

For a private local release, additional markers can be supplied through the
``ANONYMITY_DENYLIST`` environment variable separated by ``os.pathsep``.  The
marker values are never written into the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import zipfile


ARCHIVE_ROOT = "submission_code"
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".json",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
}

# Keep this list narrow and paper-facing.  Exhaustive discovery scripts and
# manuscript-development material are intentionally absent.
INCLUDE_FILES = (
    Path("pyproject.toml"),
    Path("experiments/forward_equivalence_order_census.py"),
    Path("experiments/linear_ssm_validation.py"),
    Path("experiments/paper_figures.py"),
)
INCLUDE_TREES = (
    Path("src"),
    Path("tests"),
    Path("data"),
)

GENERIC_IDENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email address",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    (
        "public GitHub URL",
        re.compile(r"https?://(?:www\.)?github\.com/", re.IGNORECASE),
    ),
    (
        "GitHub SSH URL",
        re.compile(r"git@github\.com:", re.IGNORECASE),
    ),
    (
        "local Unix home path",
        re.compile(r"/(?:home|Users)/[^/\s]+/"),
    ),
)

ANONYMOUS_README = """# Anonymous supplementary code

This archive accompanies a double-blind research submission.  It contains the
installable implementation, exact regression tests, the fixed forward-equivalence
census, the independent linear state-space validation, and the deterministic
paper-figure generator.

## Install and test

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,optimization]'
pytest -q
```

## Main evidence programs

```bash
python experiments/forward_equivalence_order_census.py
python experiments/linear_ssm_validation.py
python experiments/paper_figures.py --outdir generated_figures
```

The manuscript gives the mathematical assumptions and distinguishes exact
regressions from outside-CI breadth/optimizer evidence.  This archive intentionally
contains no repository history, author metadata, acknowledgments, or public
project links.
"""


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def collect_export_files(root: Path) -> tuple[Path, ...]:
    """Return the sorted whitelist of repository files to export."""
    root = root.resolve()
    selected: set[Path] = set()

    for relative in INCLUDE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required export file is missing: {relative}")
        selected.add(relative)

    for relative_tree in INCLUDE_TREES:
        tree = root / relative_tree
        if not tree.is_dir():
            raise FileNotFoundError(f"required export tree is missing: {relative_tree}")
        for path in tree.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache"} for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            selected.add(relative)

    return tuple(sorted(selected, key=lambda path: path.as_posix()))


def _custom_markers(cli_markers: tuple[str, ...]) -> tuple[str, ...]:
    environment = os.environ.get("ANONYMITY_DENYLIST", "")
    environment_markers = tuple(
        marker for marker in environment.split(os.pathsep) if marker
    )
    return tuple(marker for marker in (*cli_markers, *environment_markers) if marker)


def identity_leaks(text: str, *, deny_markers: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return human-readable identity-leak matches for one text payload."""
    findings: list[str] = []
    for label, pattern in GENERIC_IDENTITY_PATTERNS:
        if pattern.search(text):
            findings.append(label)
    lowered = text.casefold()
    for marker in deny_markers:
        if marker.casefold() in lowered:
            findings.append(f"deny marker {marker!r}")
    return tuple(findings)


def audit_export(root: Path, files: tuple[Path, ...], *, deny_markers: tuple[str, ...]) -> None:
    """Fail if any exported textual payload contains an identity marker."""
    failures: list[str] = []
    for relative in files:
        path = root / relative
        if not _is_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"{relative}: not valid UTF-8 ({exc})")
            continue
        findings = identity_leaks(text, deny_markers=deny_markers)
        if findings:
            failures.append(f"{relative}: {', '.join(findings)}")

    readme_findings = identity_leaks(ANONYMOUS_README, deny_markers=deny_markers)
    if readme_findings:
        failures.append(f"generated README: {', '.join(readme_findings)}")

    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"anonymous bundle identity audit failed:\n  - {joined}")


def _zip_info(archive_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    # Regular file, rw-r--r--.  Fixed permissions avoid host-dependent ZIP bytes.
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _manifest(entries: list[tuple[str, bytes]]) -> bytes:
    lines = [
        f"{hashlib.sha256(payload).hexdigest()}  {name}"
        for name, payload in entries
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_bundle(
    root: Path,
    output: Path,
    *,
    deny_markers: tuple[str, ...] = (),
) -> tuple[Path, tuple[str, ...]]:
    """Build and return ``(output_path, archived_repository_files)``."""
    root = root.resolve()
    output = output.resolve()
    files = collect_export_files(root)
    markers = _custom_markers(deny_markers)
    audit_export(root, files, deny_markers=markers)

    entries: list[tuple[str, bytes]] = [
        (f"{ARCHIVE_ROOT}/README.md", ANONYMOUS_README.encode("utf-8"))
    ]
    for relative in files:
        entries.append(
            (f"{ARCHIVE_ROOT}/{relative.as_posix()}", (root / relative).read_bytes())
        )
    entries.sort(key=lambda item: item[0])
    manifest_payload = _manifest(entries)
    entries.append((f"{ARCHIVE_ROOT}/MANIFEST.sha256", manifest_payload))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in entries:
            archive.writestr(_zip_info(name), payload)

    return output, tuple(relative.as_posix() for relative in files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/anonymous_submission_code.zip"),
    )
    parser.add_argument(
        "--deny-marker",
        action="append",
        default=[],
        help="case-insensitive identity/project marker that must not occur in exported text",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output, files = build_bundle(
        root,
        args.output,
        deny_markers=tuple(args.deny_marker),
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"anonymous bundle: {output}")
    print(f"repository files: {len(files)}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
