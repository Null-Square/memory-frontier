# Anonymous code and artifact checklist

ICLR 2027 is double blind. The research repository is public and currently contains identity-bearing Git history, account names, commit metadata, and pull-request links. Do **not** submit a direct link to this repository during anonymous review.

## Preferred anonymous artifact

Prepare a clean snapshot from the frozen release commit into a new directory or anonymous repository with **no Git history**. Include only files needed to reproduce the paper:

- `src/memory_frontier/`
- `tests/`
- selected `experiments/` used by the paper
- `pyproject.toml`
- a neutral reproduction README
- frozen paper figures or the figure-generation script
- optional machine-readable release metadata with package versions

Do not include internal research-chat transcripts, scratch files, pull-request metadata, or author-facing submission notes.

## Identity scrub

Before release, search the anonymous snapshot for:

- author names;
- GitHub usernames or organization names;
- personal email addresses;
- direct URLs to the public development repository, PRs, issues, commits, Actions runs, or user profiles;
- badges whose target reveals the development repository;
- package metadata that names an author or maintainer;
- local absolute paths or home-directory names;
- comments/docstrings that identify an individual or institution;
- generated PDF metadata containing author identity.

The exact release commit SHA may be stored privately in submission/release notes for provenance, but should not be exposed in the anonymous artifact if it trivially resolves to the identity-bearing public repository.

## Git/history scrub

For a zip artifact:

1. export/copy files from the release commit;
2. ensure `.git/` is absent;
3. remove `.github/` unless needed for anonymous reproduction;
4. remove development-only paper/submission notes that contain identity-bearing links;
5. inspect `git grep`-equivalent text search over the exported directory;
6. inspect archive filenames and top-level directory name for identity.

For an anonymous repository:

1. initialize a fresh repository from the exported snapshot;
2. use a neutral project/repository name;
3. make a single anonymous initial commit or otherwise avoid preserving development authorship/history;
4. verify the hosting account/organization does not reveal the authors;
5. do not cross-link the anonymous repository to the public development repository during review.

## Reproduction content that should remain

Anonymity must not remove scientific provenance inside the artifact. Keep:

- deterministic seeds and fixed experiment configurations;
- exact expected census counts;
- test names and mathematical fixture descriptions;
- software-version/install instructions;
- figure-generation commands;
- clear labels separating theorem regressions from illustrative numerical evidence.

## Final anonymous-artifact gate

Before entering a URL in OpenReview, verify from a logged-out/private browsing session that:

- the artifact itself does not identify the authors;
- the landing page and account name do not identify the authors;
- no obvious link resolves back to the public development repository;
- installation and the core regression suite work from the anonymous snapshot;
- the paper cites the artifact anonymously and does not mention development commit/PR numbers.

After the double-blind period ends, the anonymous artifact can be replaced or redirected to the public canonical repository as permitted by the venue.
