![Memory Frontier — Same predictor. Different learnability.](./assets/memory-frontier-cover.png)

<p align="center">
  <a href="https://github.com/Null-Square/memory-frontier/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/Null-Square/memory-frontier/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/Null-Square/memory-frontier/actions/workflows/jmlr-build.yml"><img alt="JMLR build" src="https://github.com/Null-Square/memory-frontier/actions/workflows/jmlr-build.yml/badge.svg"></a>
</p>

# Memory Frontier

**Same Predictor, Different Learnability: Construction Order in Finite-State Memory**

Memory Frontier is the research and reproducibility package for a narrow question about learned memory:

> If two finite-state memory systems make the **same predictions now**, can gradient learning still have very different difficulty constructing the computation that matters next?

**Yes.** Present behavior does not determine local learnability. Dormant memory topology can leave the current finite-horizon predictor exactly unchanged while changing the order at which useful latent computation becomes accessible to gradient-based learning.

## Findings at a glance

| Finding | Result |
|---|---|
| **Same predictor, different learnability** | Rewiring transition rows that are dormant under the current source-memory process preserves the entire current finite-horizon predictor, yet can change construction order once those rows become reachable. |
| **Construction-order hierarchy** | For the finite-horizon affine controller family, `d_support <= d_operator <= d_loss`. |
| **Forward-equivalence census** | Across **1,000** controllers in one exact forward-equivalence class, construction orders **1–5** all occur, with counts **235, 282, 244, 155, 84**. |
| **Hierarchy violations** | **0** in the frozen 1,000-controller census. |
| **Independent validation** | A differentiable linear state-space delay model reproduces loss orders **1–5** using ordinary autograd. |

Only dormant zero-cost wiring varies in the census; the source, architecture, decoder, current reachable dynamics, trainable directions, and horizon remain fixed.

## The core idea

The first local degree at which a latent computation becomes visible is separated into three quantities:

$$
d_{\mathrm{support}} \le d_{\mathrm{operator}} \le d_{\mathrm{loss}}.
$$

- **`d_support`** — minimum source-valid perturbative construction cost to reach a distinct predictive readout class.
- **`d_operator`** — first nonzero quotient occupancy construction operator.
- **`d_loss`** — first nonconstant scalar-loss degree after decoder cancellation.

The distinction matters because representability alone is not enough: a useful computation may already exist in the model class while the local learning path needed to construct it appears only at a higher order.

For the exact dormant-topology intervention, see [`docs/dormant_forward_equivalence.md`](docs/dormant_forward_equivalence.md). For the frozen breadth result, see [`docs/forward_equivalence_order_census.md`](docs/forward_equivalence_order_census.md).

## Reproduce the headline results

Python **3.11+** is required.

```bash
git clone https://github.com/Null-Square/memory-frontier.git
cd memory-frontier

python -m pip install --upgrade pip
python -m pip install -e '.[dev,optimization]'
pytest -q
```

Run the two frozen evidence programs:

```bash
python experiments/forward_equivalence_order_census.py
python experiments/linear_ssm_validation.py
```

Generate the deterministic paper figures:

```bash
python experiments/paper_figures.py --outdir paper/generated_figures
```

For exact expected outputs, claim-to-code provenance, and reviewer instructions, use [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## What's in this repository

| Path | Purpose |
|---|---|
| [`src/memory_frontier/`](src/memory_frontier/) | Exact finite-memory oracles and construction/accessibility utilities |
| [`tests/`](tests/) | Exact theorem fixtures, regression tests, and adversarial cases |
| [`experiments/`](experiments/) | Frozen breadth evidence, optimizer audits, independent validation, and figure generation |
| [`docs/`](docs/) | Theory notes and supporting derivations |
| [`examples/`](examples/) | Small executable reports and source/witness scans |
| [`data/`](data/) | Frozen reference data |
| [`paper/`](paper/) | Manuscript, appendices, bibliography, venue workspaces, and release material |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Reviewer path from a clean environment to the headline claims |

## Additional validated results

The package also regression-tests or freezes evidence that:

- construction order is preserved by regular local reparameterizations, while singular charts can change it;
- positive diagonal preconditioning changes constants and balance geometry but not the isolated degree-controlled escape class;
- rare-edge Euclidean softmax converts degree `d` to a `Theta(delta^-d)` bootstrap-time exponent;
- exact near-degeneracy counterexamples mark where leading route/mode predictions can fail.

## Paper and reviewer path

The primary journal workspace is [`paper/jmlr/`](paper/jmlr/). A secondary conference-formatted workspace is retained in [`paper/iclr2027/`](paper/iclr2027/) for portability. The canonical bibliography is [`paper/references.bib`](paper/references.bib), and the long-form manuscript is [`paper/manuscript.md`](paper/manuscript.md).

The repository includes GitHub Actions for the complete test suite, the JMLR/reviewer build, and the retained submission build pipeline.

## Scope and claim discipline

This project does **not** claim that vanishing gradients, finite-state-controller polynomial objectives, deep-linear balancedness, singular/dead directions, same-function optimization differences, or softmax slowness are new.

The narrow contribution is the **source-aware local construction-order framework** and the **dormant-topology intervention within a fixed current forward-equivalence class**. Exact algebraic claims are regression tested; the 1,000-controller census and long optimizer audits are reported as breadth/dynamics evidence rather than universal theorems.
