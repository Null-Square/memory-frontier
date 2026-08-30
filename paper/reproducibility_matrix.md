# Claim-to-code reproducibility matrix

This table maps the manuscript's main scientific claims to the exact repository
artifacts that support them. It separates theorem/regression evidence from
outside-CI numerical audits so that numerical choices cannot silently become part
of an algebraic claim.

| manuscript claim | status | implementation / derivation | CI regression | outside-CI evidence | wording boundary |
|---|---|---|---|---|---|
| Finite-horizon affine controller loss has an exact local polynomial expansion and source-valid construction coefficients | exact algebra | `src/memory_frontier/perturbative.py`, `src/memory_frontier/construction_order.py` | `tests/test_perturbative.py`, `tests/test_general_construction_order.py` | `experiments/general_construction_order_census.py` | Polynomiality/maximal degree itself has prior FSC/PSR literature; claim the local minimum construction degree and factorization, not polynomiality. |
| Construction-order hierarchy \(d_{support}\le d_{operator}\le d_{loss}\) | theorem | `src/memory_frontier/construction_order.py` | `tests/test_general_construction_order.py` | `experiments/general_construction_order_census.py` | Construction-origin regime; support cost zero is not itself a derivative-order statement. |
| Scalar coefficient factorization \(c_\alpha=-\langle G_\alpha,\log q\rangle\) | exact identity | `src/memory_frontier/construction_order.py` | `tests/test_general_construction_order.py` | — | Decoder values enter after the source/transition quotient operator; do not identify graph distance with scalar order. |
| Strict decoder cancellation can realize \((1,1,2)\) | exact counterexample | `src/memory_frontier/construction_order.py` | `tests/test_general_construction_order.py` | — | Shows the operator layer is necessary; exact cancellation is nongeneric, not impossible. |
| Generic decoder values satisfy \(d_{loss}=d_{operator}\) almost surely under fixed equality partition | analytic corollary | `docs/construction_order_genericity.md` | deterministic coefficient machinery in `tests/test_general_construction_order.py` | `experiments/general_construction_order_census.py` | Requires alphabet size at least two, interior decoder probabilities, fixed equality partition, finite operator order. |
| Rewiring only source/horizon forward-dormant transition rows preserves the full current occupancy process | theorem | `src/memory_frontier/forward_equivalence.py` | `tests/test_dormant_forward_equivalence.py` | — | Finite-horizon/source-aware statement; stronger than equality of one scalar loss. |
| Forward-equivalent scaffold family can realize orders \(1,\ldots,R\) | exact existence construction | `src/memory_frontier/forward_equivalence.py`, `src/memory_frontier/construction_order.py` | `tests/test_dormant_forward_equivalence.py` | — | The deterministic existence family chooses the perturbation-direction prefix for each desired order; do not call its directions fixed across all members. |
| One fixed forward-equivalence class with fixed trainable directions realizes orders 1–5 when only dormant zero-cost wiring varies | exact randomized census | same construction oracle as above | base theorem regressions above | `experiments/forward_equivalence_order_census.py` | 1,000 frozen samples; counts 235, 282, 244, 155, 84; experiment is breadth evidence, not a probability theorem. |
| First nonzero scalar degree is visibility order, while beneficial learnability additionally requires an admissible negative leading ray | theorem-level semantic refinement | `docs/descent_visibility_boundary.md` | main scaffold sign checks are frozen in `tests/test_order_barrier.py`, `tests/test_scaffold_escape.py`, and route-specific regressions | — | Use “construction/loss order” for visibility. Use “beneficial construction” or “useful gradient accessibility” only when the admissible sign condition is satisfied. |
| Beneficial isolated degree-\(d\) monomial under affine probability-coordinate flow has finite/log/\(\delta^{-(d-2)}\) bootstrap classes | exact isolated-flow theorem | `src/memory_frontier/construction_time.py` | `tests/test_construction_time.py`, `tests/test_scaffold_escape.py` | `experiments/construction_time_scaling.py`, `experiments/dormant_scaffold_escape.py` | Homogeneous escape exponent is not claimed as new general mathematics; finite-memory contribution is the topology/order assignment. |
| Positive diagonal conditioning changes metric/prefactor but not the isolated probability-coordinate degree class | exact monomial extension | `src/memory_frontier/preconditioned_construction.py` | `tests/test_preconditioned_construction.py` | — | Positive diagonal metric only; no general adaptive-optimizer invariance claim. |
| Smooth local reparameterizations with nonsingular Jacobian preserve scalar loss-germ order | classical local theorem applied here | `src/memory_frontier/parameterization.py` | `tests/test_parameterization_order.py` | — | Scalar loss order only. Support/operator orders are defined in the affine transition family; Euclidean trajectories are not coordinate invariant. |
| Singular power charts transform ordinary degree into weighted degree | exact pullback identity | `src/memory_frontier/parameterization.py` | `tests/test_parameterization_order.py` | — | Used as boundary/scope clarification, not as a novelty claim about singularity theory. |
| Binary rare-edge logit flow for isolated degree \(d\) obeys \(\dot p=Cp^{d+1}(1-p)^2\) and \(\tau\sim (Cd)^{-1}\delta^{-d}\) | exact theorem | `src/memory_frontier/softmax_boundary.py` | `tests/test_softmax_boundary.py` | — | Exact closed form is binary/symmetric/isolated. Softmax slowness itself is prior art. |
| Full \(K\)-way softmax target-edge Jacobian preserves the rare-edge \(\delta^{-d}\) exponent up to constants | exact Jacobian bound + asymptotic corollary | `src/memory_frontier/softmax_boundary.py`, `docs/softmax_boundary_bootstrap.md` | `tests/test_softmax_boundary.py` | — | Exponent/bounds only; no arbitrary coupled-row exact prefactor theorem. |
| Five-link linear recurrent memory reproduces loss orders 1–5 and gradient orders 0–4 while all base initializations have identical zero predictor | exact smooth-model identity + autograd regression | recurrent formulas embedded in `tests/test_linear_ssm_validation.py` | `tests/test_linear_ssm_validation.py` | `experiments/linear_ssm_validation.py` | Validation outside finite-state-controller formalism; multiplication-chain/deep-linear dynamics are not claimed as novel. |
| Fixed-step SGD threshold counts 4, 43, 361, 3965, 53327 for the frozen linear-SSM setup | numerical audit | — | — | `experiments/linear_ssm_validation.py` | Illustrative optimizer evidence only; not universal scaling constants. |
| Leading-route labels can reverse when the leading margin is comparable to the first neglected order | exact finite-memory counterexample | `src/memory_frontier/near_tie.py` | `tests/test_near_tie_route_reversal.py` | route-race experiment scripts under `experiments/` | Scope boundary for leading-order theory, not a universal route-selection theorem. |
| Bilinear leading construction dynamics decompose by SVD; near spectral degeneracy higher-order terms can rotate/reverse the dominant local mode | exact leading solver + exact local counterexample | `src/memory_frontier/bilinear_routes.py`, `src/memory_frontier/spectral.py` | `tests/test_bilinear_route_spectrum.py`, `tests/test_spectral_gap_reversal.py` | `experiments/bilinear_route_spectrum.py` | SVD/deep-linear and perturbation mathematics are prior art; finite-memory realization is supporting scope evidence. |
| Main manuscript figures are reproducible from frozen theorem/census values | deterministic plotting script | — | — | `experiments/paper_figures.py` | Figure script is presentation infrastructure and intentionally outside CI; generated binaries need not be committed. |

## Main-paper numerical statements

The following numbers should not appear in the main paper without the associated
source artifact:

- **one-class census:** 235, 282, 244, 155, 84 —
  `experiments/forward_equivalence_order_census.py`;
- **linear-SSM fixed-step SGD crossings:** 4, 43, 361, 3965, 53327 —
  `experiments/linear_ssm_validation.py`;
- **near-tie and spectral reversal constants:** use the exact regression fixtures
  in `tests/test_near_tie_route_reversal.py` and
  `tests/test_spectral_gap_reversal.py` rather than copying scratch values.

## Release reproduction sequence

For a clean release candidate, run:

```bash
python -m pip install -e '.[dev,optimization]'
pytest -q
python experiments/forward_equivalence_order_census.py
python experiments/linear_ssm_validation.py
```

For the main figures, additionally install matplotlib and run:

```bash
python experiments/paper_figures.py --outdir paper/generated_figures
```

The exhaustive/random experiments remain outside CI by repository policy. Their
role is breadth and numerical validation; theorem correctness is frozen in the
unit/regression suite.
