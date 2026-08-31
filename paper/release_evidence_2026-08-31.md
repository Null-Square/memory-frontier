# Release evidence audit — 2026-08-31

This audit records an independent local rerun of the two headline programs that are intentionally kept outside CI. It is **not** a substitute for the final clean-checkout installation test: the execution environment for this audit could not clone the public repository directly, so the relevant current-main source/dependencies were reconstructed from repository file contents and executed locally.

The purpose of this audit is narrower: verify that the frozen manuscript outputs still follow from the current source logic after the submission-packaging changes.

## 1. Same-forward dormant-topology census

Source program: `experiments/forward_equivalence_order_census.py`.

Reconstructed current-main dependencies:

- `src/memory_frontier/core.py`
- `src/memory_frontier/construction_order.py`
- `src/memory_frontier/forward_equivalence.py`

Frozen configuration:

- seed: `20260830`
- samples: `1000`
- depth: `5`
- horizon: `10`
- dormant skip probability: `0.45`
- dormant edge strength: `0.5`

Observed rerun:

- non-dormant rewires: **0**
- hierarchy violations: **0**
- exact `(1,1,1)`: **235**
- exact `(2,2,2)`: **282**
- exact `(3,3,3)`: **244**
- exact `(4,4,4)`: **155**
- exact `(5,5,5)`: **84**

The counts exactly match the frozen manuscript/reproducibility-matrix values.

## 2. Independent linear state-space validation

Source program: `experiments/linear_ssm_validation.py`.

Environment used NumPy and PyTorch with the program's fixed seed and scales.

### Closed-form/autograd check

The recurrent loss and closed-form objective agreed, and the missing-link gradient norm agreed with

\[
\|\nabla_{1:r}L\|_2
=\sqrt r\,(1-s^r)s^{r-1}
\]

within the program's strict tolerances.

### Fitted scaling slopes

| order | loss slope | gradient slope | predicted |
|---:|---:|---:|---:|
| 1 | 0.993789 | -0.012515 | (1, 0) |
| 2 | 1.999820 | 0.999640 | (2, 1) |
| 3 | 2.999995 | 1.999991 | (3, 2) |
| 4 | 4.000000 | 3.000000 | (4, 3) |
| 5 | 5.000009 | 4.000000 | (5, 4) |

The small finite-scale deviation in the order-1 gradient fit is expected from the exact factor `(1-s)`; the regression claim is the leading order and the exact formula check passes.

### Fixed-step SGD reference

Configuration: initialization `0.05`, learning rate `0.05`, gain threshold `0.2`.

| order | threshold step | gain at crossing |
|---:|---:|---:|
| 1 | 4 | 0.248822827 |
| 2 | 43 | 0.212748591 |
| 3 | 361 | 0.212640548 |
| 4 | 3965 | 0.207755313 |
| 5 | 53327 | 0.205800049 |

The threshold-step counts exactly match the frozen manuscript values.

## Status after this audit

Closed:

- headline same-forward census outputs reproduce;
- independent smooth recurrent analytic/autograd outputs reproduce;
- illustrative SGD threshold counts reproduce.

Still required before submission:

1. clean checkout/export from the final release SHA in a normal networked environment;
2. `python -m pip install -e '.[dev,optimization]'` from that clean source tree;
3. full `pytest -q` at the final release SHA;
4. rerun the two programs directly from that installed clean tree and capture the console logs;
5. generate the four main figures from that same clean tree;
6. build the final anonymous PDF with the official ICLR 2027 style package.
