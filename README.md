<p align="center">
  <img src="assets/memory-frontier-cover.png" alt="Memory Frontier — Same predictor. Different learnability." width="100%">
</p>

<p align="center">
  <a href="https://github.com/Null-Square/memory-frontier/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/Null-Square/memory-frontier/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/Null-Square/memory-frontier/actions/workflows/jmlr-build.yml"><img alt="JMLR build" src="https://github.com/Null-Square/memory-frontier/actions/workflows/jmlr-build.yml/badge.svg"></a>
</p>

# Memory Frontier

**Same Predictor, Different Learnability: Construction Order in Finite-State Memory**

Memory Frontier is the research and reproducibility package for studying a precise failure of the usual representability intuition: a memory system can already represent a useful computation while gradient-based learning still has very different difficulty constructing it.

The repository contains exact finite-memory oracles, theorem-backed regression tests, frozen breadth experiments, independent differentiable validation, paper figures, and the journal/reviewer build pipeline.

## Why this matters

Two controllers can implement the **same current finite-horizon predictor** yet expose different learning paths once dormant memory transitions become reachable. In this setting, present behavior alone does not determine local learnability.

For the finite-horizon affine controller family, the first local degree at which a latent computation becomes visible obeys

$$
d_{\mathrm{support}} \le d_{\mathrm{operator}} \le d_{\mathrm{loss}}.
$$

- **`d_support`** — minimum source-valid perturbative construction cost to reach a distinct predictive readout class.
- **`d_operator`** — first nonzero quotient occupancy construction operator.
- **`d_loss`** — first nonconstant scalar-loss degree after decoder cancellation.

A second theorem shows that rewiring transition rows never exercised by the current source-memory process leaves the entire current finite-horizon predictor unchanged, while potentially changing construction order once learning makes those dormant rows reachable.

## Evidence at a glance

| Evidence | Frozen result | Reproduce |
|---|---:|---|
| Same forward-equivalence-class census | orders 1–5 across 1,000 controllers | `python experiments/forward_equivalence_order_census.py` |
| Order counts | `235, 282, 244, 155, 84` | same command |
| Hierarchy violations | `0` | same command |
| Independent linear state-space validation | loss orders 1–5 with ordinary autograd | `python experiments/linear_ssm_validation.py` |
| Main paper figures | 4 deterministic figures | `python experiments/paper_figures.py --outdir paper/generated_figures` |

Only dormant zero-cost wiring varies in the 1,000-controller census; the source, architecture, decoder, current reachable dynamics, trainable directions, and horizon remain fixed.

## Quick start

Python **3.11+** is required.

```bash
git clone https://github.com/Null-Square/memory-frontier.git
cd memory-frontier

python -m pip install --upgrade pip
python -m pip install -e '.[dev,optimization]'
pytest -q
```

For the lightweight package without research/development extras:

```bash
python -m pip install -e .
```

## Reproduce the headline results

Run the two frozen outside-CI evidence programs:

```bash
python experiments/forward_equivalence_order_census.py
python experiments/linear_ssm_validation.py
```

Generate the paper figures:

```bash
python experiments/paper_figures.py --outdir paper/generated_figures
```

For exact expected outputs, claim-to-code provenance, and reviewer instructions, use [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository structure

| Path | Purpose |
|---|---|
| [`src/memory_frontier/`](src/memory_frontier/) | Exact finite-memory oracles and construction/accessibility utilities |
| [`tests/`](tests/) | Exact theorem fixtures, regression tests, and adversarial cases |
| [`experiments/`](experiments/) | Frozen breadth evidence, optimizer audits, and figure generation |
| [`examples/`](examples/) | Small executable reports and source/witness scans |
| [`data/`](data/) | Frozen reference data |
| [`docs/`](docs/) | Supporting theory and research notes |
| [`paper/`](paper/) | Manuscript, appendices, bibliography, venue workspaces, and release material |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Reviewer path from clean environment to headline claims |

## Paper and reviewer workflow

The primary journal workspace is [`paper/jmlr/`](paper/jmlr/). A secondary conference-formatted workspace is retained in [`paper/iclr2027/`](paper/iclr2027/) for portability, with the canonical bibliography in [`paper/references.bib`](paper/references.bib) and long-form manuscript in [`paper/manuscript.md`](paper/manuscript.md).

The repository includes three GitHub Actions workflows:

- **tests** — clean checkout, dependency installation, and complete regression suite;
- **JMLR build** — clean install/test, reviewer-code retest, deterministic figure generation, pinned JMLR style retrieval, LaTeX/BibTeX build, validation, and artifact upload;
- **submission build** — venue/release packaging workflow retained with the paper workspace.

## What the paper establishes

Beyond the support/operator/loss hierarchy and dormant-topology intervention, the package regression-tests or freezes evidence for the following statements:

- construction order is preserved by regular local reparameterizations;
- singular charts can change the order;
- positive diagonal preconditioning changes constants/balance geometry but not the isolated degree-controlled escape class;
- rare-edge Euclidean softmax converts degree `d` to a `Theta(delta^-d)` bootstrap-time exponent;
- an independent differentiable linear state-space delay model reproduces loss orders 1–5 using ordinary autograd;
- exact near-degeneracy counterexamples show where leading route/mode predictions can fail.

## Claim discipline

This project does **not** claim that vanishing gradients, finite-state-controller polynomial objectives, deep-linear balancedness, singular/dead directions, same-function optimization differences, or softmax slowness are new.

The narrow contribution is the **source-aware local construction-order framework** and the **dormant-topology intervention within a fixed current forward-equivalence class**.

Exact algebraic claims are regression tested. The 1,000-controller census and long optimizer audits remain outside CI by design and are reported as breadth/dynamics evidence rather than universal theorems.
