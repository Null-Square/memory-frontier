# Construction Order in Finite-State Memory

This repository is the research and reproducibility package for the paper **“Same Predictor, Different Learnability: Construction Order in Finite-State Memory.”**

## Main result

A memory system's ability to represent a useful computation does not determine how easily gradient-based learning can construct that computation.

For a finite-horizon affine controller family, the first local degree at which a latent computation becomes visible obeys

\[
d_{\mathrm{support}}\le d_{\mathrm{operator}}\le d_{\mathrm{loss}}.
\]

- `d_support`: minimum source-valid perturbative construction cost to reach a distinct predictive readout class.
- `d_operator`: first nonzero quotient occupancy construction operator.
- `d_loss`: first nonconstant scalar-loss degree after decoder cancellation.

The coefficient factorization separates source/path cancellation from decoder cancellation. A second theorem shows that rewiring transition rows never exercised by the current source-memory process leaves the entire current finite-horizon predictor unchanged, yet can change construction order once learning makes those dormant rows reachable.

## Headline evidence

Inside one exact forward-equivalence class, only dormant zero-cost wiring is randomized. The frozen 1,000-controller census finds exact orders 1–5 with counts

```text
order:   1    2    3    4   5
count: 235  282  244  155  84
```

with zero hierarchy violations.

The paper also establishes:

- construction order is preserved by regular local reparameterizations;
- singular charts can change the order;
- positive diagonal preconditioning changes constants/balance geometry but not the isolated degree-controlled escape class;
- rare-edge Euclidean softmax converts degree `d` to a `Theta(delta^-d)` bootstrap-time exponent;
- an independent differentiable linear state-space delay model reproduces loss orders 1–5 using ordinary autograd;
- exact near-degeneracy counterexamples show where leading route/mode predictions can fail.

## Reviewer quick start

Python 3.11 is the reference environment.

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev,optimization]'
pytest -q
```

Reproduce the two frozen outside-CI evidence programs:

```bash
python experiments/forward_equivalence_order_census.py
python experiments/linear_ssm_validation.py
```

Generate the four paper figures:

```bash
python experiments/paper_figures.py --outdir paper/generated_figures
```

For claim-to-code provenance, exact expected outputs, and the journal build, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Paper workspaces

- `paper/jmlr/` — primary journal submission workspace.
- `paper/iclr2027/` — secondary conference-formatted workspace retained for portability.
- `paper/references.bib` — canonical bibliography.
- `paper/manuscript.md` — long-form canonical prose manuscript.

The JMLR workspace uses the official `jmlr2e` template and has its own CI build. It produces a PDF, source bundle, reviewer code bundle, and build report.

## Repository map

- `src/memory_frontier/` — exact finite-memory oracles and construction/accessibility utilities.
- `tests/` — exact theorem fixtures and adversarial regressions.
- `experiments/` — frozen breadth evidence, optimizer audits, and figure generation.
- `data/` — frozen reference data.
- `docs/` — supporting theory and research notes.
- `paper/` — manuscript, appendices, bibliography, venue workspaces, and release documentation.

## Claim discipline

The paper does **not** claim that vanishing gradients, finite-state-controller polynomial objectives, deep-linear balancedness, singular/dead directions, same-function optimization differences, or softmax slowness are new. The paper's narrow contribution is the source-aware local construction-order framework and dormant-topology intervention within a fixed current forward-equivalence class.

Exact algebraic claims are regression tested. The 1,000-controller census and long optimizer audits remain outside CI by design and are reported as breadth/dynamics evidence rather than universal theorems.
