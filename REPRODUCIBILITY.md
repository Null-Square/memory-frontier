# Reproducibility guide

This file is the shortest reviewer path from a clean environment to the headline claims.

## Environment and exact test suite

Reference environment: Python 3.11.

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev,optimization]'
pytest -q
```

The ordinary GitHub Actions test workflow performs a clean checkout, installs these dependencies, and runs the complete regression suite.

## Claim-to-code map

| Paper claim | Primary regression / evidence |
|---|---|
| support/operator/loss hierarchy and coefficient reconstruction | `tests/test_general_construction_order.py` |
| dormant forward equivalence | `tests/test_dormant_forward_equivalence.py` |
| affine construction-time law | `tests/test_construction_time.py` |
| regular versus singular reparameterization | `tests/test_parameterization_order.py` |
| positive diagonal preconditioning | `tests/test_preconditioned_construction.py` |
| binary/full-softmax boundary law | `tests/test_softmax_boundary.py` |
| independent linear state-space validation | `tests/test_linear_ssm_validation.py` |
| scalar near-tie route reversal | `tests/test_near_tie_route_reversal.py` |
| bilinear construction spectrum | `tests/test_bilinear_route_spectrum.py` |
| spectral-gap reversal | `tests/test_spectral_gap_reversal.py` |
| figure rendering | `tests/test_paper_figures.py` |

Supporting exact regressions in `tests/` cover the finite-horizon oracle, accessibility operators, higher-order blindness, dormant scaffolds, invariant geometry, route races, and hard/surrogate comparisons.

## Frozen outside-CI evidence

These programs are intentionally not added to CI because they are breadth/optimizer evidence rather than correctness gates.

### Same-forward-equivalence-class census

```bash
python experiments/forward_equivalence_order_census.py
```

Frozen 1,000-controller result:

```text
order 1: 235
order 2: 282
order 3: 244
order 4: 155
order 5: 84
hierarchy violations: 0
```

Only dormant zero-cost wiring varies; the source, architecture, decoder, current reachable dynamics, trainable directions, and horizon remain fixed.

### Linear state-space validation and optimizer audit

```bash
python experiments/linear_ssm_validation.py
```

The exact/autograd loss slopes are approximately 1,2,3,4,5 and missing-factor gradient slopes approximately 0,1,2,3,4. The frozen illustrative SGD gain-threshold steps are:

```text
4, 43, 361, 3965, 53327
```

These step counts are fixture/optimizer specific and are not universal constants.

## Figures

```bash
python experiments/paper_figures.py --outdir paper/generated_figures
```

This creates PDF and PNG versions of the four main figures:

1. forward-equivalent dormant scaffolds;
2. support/operator/loss order hierarchy;
3. 1,000-controller fixed-class census;
4. probability-coordinate versus softmax order-to-time map.

## JMLR paper build

The authoritative journal build is `.github/workflows/jmlr-build.yml`. It performs a clean install/test, retests the deterministic reviewer-code package, generates figures, fetches the pinned official JMLR style, compiles LaTeX/BibTeX, checks references/layout/PDF size, and uploads the PDF/source/code artifacts.

The human-only fields listed in `paper/jmlr/SUBMISSION_FIELDS.md` must be completed before actual journal submission.

## Evidence discipline

- Algebraic/theorem claims are exact and regression backed.
- Numerical finite audits are labeled as such.
- Exhaustive/randomized breadth programs stay outside CI.
- Optimization-time claims state their parameterization/metric assumptions.
- Leading route/mode predictions are not treated as reliable at near-degeneracy; explicit counterexamples are frozen in the test suite.
