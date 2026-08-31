# JMLR submission workspace

This directory is the primary journal-submission workspace for **Journal of Machine Learning Research (JMLR)**.

## Why JMLR

The paper is primarily a theoretical/analytical study of learning-system behavior: it introduces an exact construction-order framework, proves forward-equivalence and accessibility results for finite-memory predictors, and validates the mechanism in a differentiable recurrent state-space model. JMLR explicitly welcomes theoretical studies that provide new insight into the design or performance of learning methods and new analytical frameworks for understanding learning behavior.

## Build

The authoritative build is `.github/workflows/jmlr-build.yml`. It:

1. checks out a clean repository state;
2. installs Python 3.11 dependencies and runs the full regression suite;
3. rebuilds and retests the deterministic reviewer-code ZIP;
4. generates the four manuscript figures;
5. fetches the official `jmlr2e.sty` from the pinned JMLR style repository commit;
6. compiles LaTeX/BibTeX;
7. fails on undefined references/citations or overfull boxes;
8. enforces the JMLR 5 MB PDF limit; and
9. uploads the PDF, source package, code package, and a build report.

The pinned official style commit is `f413f638b407af76074813f8f88a82a7a5a81e9d` from `JmlrOrg/jmlr-style-file`.

## Human-only submission fields

Before actual submission, complete `SUBMISSION_FIELDS.md` and replace the placeholders in `author_metadata.tex` and `disclosures.tex`. The repository deliberately does not infer affiliation, postal address, funding, conflicts of interest, or reviewer/editor suggestions.

## Reproduction

See the repository-root `REPRODUCIBILITY.md` for the shortest reviewer path from a clean Python environment to the exact theorem regressions, frozen breadth evidence, figures, and paper build.
