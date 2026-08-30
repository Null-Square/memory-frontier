# Submission release checklist

This checklist defines the final paper-release procedure. It is deliberately
separate from the scientific discovery workflow: once these items are satisfied,
new theory should enter only in response to a concrete correctness or reviewer
objection.

## 1. Freeze an authoritative release commit

Before generating submission artifacts:

1. confirm `main` contains the canonical `paper/manuscript.md` and
   `paper/references.bib`;
2. confirm no temporary manuscript or bibliography variants remain;
3. record the exact release commit SHA in the submission notes;
4. do not regenerate reported constants from scratch notes after this point.

The release SHA is the provenance anchor for every table, plot, appendix, and
artifact.

## 2. Clean environment regression pass

From a fresh checkout of the release commit:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,optimization]'
pytest -q
```

Required outcome: all tests pass with no local uncommitted source/test edits.

The exact theorem suite must include at least:

- `tests/test_general_construction_order.py`;
- `tests/test_dormant_forward_equivalence.py`;
- `tests/test_construction_time.py`;
- `tests/test_parameterization_order.py`;
- `tests/test_preconditioned_construction.py`;
- `tests/test_softmax_boundary.py`;
- `tests/test_linear_ssm_validation.py`;
- `tests/test_near_tie_route_reversal.py`;
- `tests/test_spectral_gap_reversal.py`.

## 3. Reproduce main numerical evidence

Run the two main outside-CI evidence programs from the same release commit:

```bash
python experiments/forward_equivalence_order_census.py
python experiments/linear_ssm_validation.py
```

Record stdout in the release notes or supplementary artifact bundle.

Expected frozen headline values:

### Fixed forward-equivalence-class census

| exact order | count |
|---:|---:|
| 1 | 235 |
| 2 | 282 |
| 3 | 244 |
| 4 | 155 |
| 5 | 84 |

Total: 1,000 controllers; zero hierarchy violations.

### Linear-SSM illustrative SGD audit

For the frozen optimizer configuration used by the experiment, expected threshold
steps are approximately:

| construction order | steps |
|---:|---:|
| 1 | 4 |
| 2 | 43 |
| 3 | 361 |
| 4 | 3965 |
| 5 | 53327 |

These step counts are illustrative evidence only and must not be described as a
universal theorem.

## 4. Generate paper figures

Install matplotlib in the release environment and run:

```bash
python -m pip install matplotlib
python experiments/paper_figures.py --outdir paper/generated_figures
```

Expected main figures:

1. `figure1_forward_equivalence.pdf` — same current forward process, different
   dormant construction orders;
2. `figure2_order_hierarchy.pdf` — support/operator/loss hierarchy and the two
   cancellation mechanisms;
3. `figure3_forward_class_census.pdf` — fixed-class random dormant-topology
   census;
4. `figure4_order_to_time_geometry.pdf` — affine-probability versus rare-edge
   softmax order-to-time maps.

Before submission, visually inspect every figure at single-column and full-page
sizes. Check mathematical subscripts, line styles, legend readability, and that
no figure implies an optimizer-invariant time law.

## 5. Manuscript semantic checks

Search the final manuscript for wording that can overstate the result.

### Required distinctions

The paper must distinguish:

- predictive capacity;
- current forward behavior;
- scalar visibility/construction order;
- beneficial descent order on the admissible cone;
- optimizer/parameterization-specific bootstrap time.

The preferred causal chain is

\[
\boxed{
\text{topology}
\to
\text{construction order/local homogeneity}
\to
\text{optimizer geometry}
\to
\text{bootstrap time}.
}
\]

### Forbidden overclaims

Remove or qualify any sentence implying that we discovered:

- vanishing memory gradients;
- polynomial finite-horizon FSC objectives;
- same-function models having different optimization landscapes;
- dormant-feature activation in general;
- vanishing order / singular directions in general;
- balancedness or deep-linear singular-mode dynamics;
- the `d-2` homogeneous saddle-escape exponent;
- generic softmax slowness;
- optimizer-invariant construction times;
- a global route winner from leading-order coefficients alone.

## 6. Beneficial-sign audit

Every main-text sentence that calls a construction “useful,” “beneficial,” or
“learnable” must be supported by a descent sign condition, not merely a nonzero
Taylor coefficient.

For an admissible cone `K` and leading homogeneous form `P_d`, verify either
explicitly or by cited regression that

\[
\exists v\in K:\quad P_d(v)<0.
\]

The isolated scaffold witnesses satisfy this through the leading form

\[
-C\prod_i x_i,
\qquad C>0,
\]

on the positive construction cone.

Use `docs/descent_visibility_boundary.md` as the terminology source of truth.

## 7. Bibliography audit

For each bibliography entry used in the main text:

1. verify title, author list, year, venue/journal, pages, DOI/arXiv identifier
   against a primary publisher/proceedings/arXiv record;
2. confirm the cited sentence does not attribute a stronger result than the
   source establishes;
3. prefer the published version over an arXiv version when both exist and the
   bibliographic metadata are stable;
4. keep contemporary preprints clearly labeled as preprints.

Especially re-check the nearest novelty-boundary references:

- Aberdeen & Baxter (2002);
- Boularias & Chaib-draa (2009);
- Fukumizu et al. (2019);
- Saxe et al. (2014);
- Kunin et al. (2025);
- Shirodkar (2026);
- Li et al. (2023);
- Rawal & DeWeese (2026).

## 8. Main-text / appendix allocation

Follow `paper/appendix_map.md`.

Main text should explain only what is required to understand the central causal
story. In particular:

- retain one cancellation fixture;
- retain one same-forward dormant spectrum/census;
- retain the structural-versus-metric dynamics distinction;
- retain the smooth linear-SSM validation;
- retain at most one concise near-tie failure example.

Balance laws, detailed route geometry, spectral variants, hard-forward
counterfactual machinery, and auxiliary exact oracles should support the paper in
appendices rather than compete for headline status.

## 9. Claim-to-code audit

Walk through every row of `paper/reproducibility_matrix.md`.

For each quantitative claim in the final paper, verify that the cited test or
experiment still exists at the release SHA and produces the stated object.

No number should survive in the paper merely because it appeared in a research
chat or scratch notebook.

## 10. Submission build

Once a venue is chosen:

1. copy the venue's official LaTeX/template files without changing the scientific
   wording first;
2. port `paper/manuscript.md` section by section;
3. preserve theorem assumptions and scope qualifiers during compression;
4. insert generated vector figures;
5. build references from canonical `paper/references.bib`;
6. compile from a clean directory;
7. inspect every warning related to undefined references, citations, overfull
   boxes, missing fonts/images, or duplicate labels.

Do not let page-limit compression remove:

- the construction-origin assumption;
- the decoder-equality quotient definition;
- the distinction between visibility and beneficial descent;
- the finite-horizon scope;
- the parameterization-dependent time caveat;
- the novelty boundary against FSC polynomial-degree and same-function embedding
  work.

## 11. Reproducibility bundle

A minimal public artifact should include:

- the exact repository release SHA;
- installation command;
- full pytest command;
- the fixed-class census command;
- the linear-SSM validation command;
- the figure-generation command;
- a short mapping from paper figures/tables to scripts;
- software versions used for the release run.

Do not require an exhaustive experiment to verify a theorem whose exact regression
already exists; separate algebraic correctness from breadth evidence.

## 12. Final stop rule

The research should be considered ready for submission once:

- the clean test suite passes;
- the two main outside-CI audits reproduce;
- the four main figures are visually approved;
- the bibliography is verified;
- theorem proofs are numbered/cross-referenced;
- the final venue template compiles cleanly;
- the claim-to-code matrix has no orphaned headline statement.

After this point, add new science only for a specific reviewer/editorial concern or
a demonstrated correctness gap. Infinite-horizon theory, adaptive-optimizer
generalization, and large neural benchmarks are follow-up projects rather than
prerequisites for the present paper.
