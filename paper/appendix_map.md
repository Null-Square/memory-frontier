# Appendix map for the construction-order manuscript

The main paper should remain centered on the support/operator/loss hierarchy,
dormant forward equivalence, the same-forward order spectrum, and the structural
order versus optimizer-geometry distinction. This map assigns the existing exact
results to appendices so they support rather than compete with that story.

## Appendix A — Exact finite-memory setup and polynomial machinery

Include:

- finite-horizon source-memory occupancy recursion;
- affine transition-direction validity conditions;
- exact multivariate coefficient recursion;
- proof that the source marginal of every nonconstant occupancy coefficient sums
  to zero over controller memory;
- relation between the exact polynomial oracle and the paper notation.

Primary repository sources:

- `src/memory_frontier/finite.py`
- `src/memory_frontier/perturbative.py`
- `tests/test_finite_horizon.py`
- `tests/test_perturbative.py`

## Appendix B — Construction-order theorem and genericity

Include the full proof of

\[
d_{\rm support}\le d_{\rm operator}\le d_{\rm loss},
\]

followed by:

- the exact coefficient factorization;
- the `(1,1,2)` decoder-cancellation witness;
- the real-analytic genericity argument for operator-to-loss equality;
- the conditional support-to-operator genericity statement;
- explicit statement of the construction-origin assumption.

Primary sources:

- `src/memory_frontier/construction_order.py`
- `tests/test_general_construction_order.py`
- `docs/construction_order_genericity.md`
- the merged paper-facing construction-order proof note.

## Appendix C — Dormant forward equivalence and accessibility spectra

Include:

- source-aware definition of an active transition row;
- induction proof of complete occupancy equivalence;
- example where a row of a reachable memory state is dormant because its symbol
  is source-impossible;
- deterministic arbitrary-order scaffold family;
- full protocol for the one-class randomized dormant-wiring census.

Primary sources:

- `src/memory_frontier/forward_equivalence.py`
- `tests/test_dormant_forward_equivalence.py`
- `experiments/forward_equivalence_order_census.py`

## Appendix D — Visibility, beneficial descent, and admissible cones

Include the semantic refinement that scalar loss order is a **visibility** order.
For an admissible local perturbation cone `K`, define the beneficial ray order and
state

\[
d_\downarrow\ge d_{\rm vis},
\]

with equality precisely when the degree-leading homogeneous form is negative on
an admissible ray.

Keep the simple example

\[
L(x,y)-L(0)=x-y^2,
\qquad K=\mathbb R_+^2,
\]

for which `d_vis=1` but `d_down=2`.

Then verify that the manuscript's bootstrap fixtures have leading term

\[
-C\prod_i x_i
\]

on the positive cone and therefore satisfy the beneficial sign condition.

Primary source:

- `docs/descent_visibility_boundary.md`

## Appendix E — Isolated construction dynamics in affine coordinates

Include:

- square-free monomial flow;
- exponent-vector/repeated-parameter extension;
- weighted balancedness identities;
- exact completion-time formulas;
- positive diagonal preconditioning extension;
- full-polynomial finite-memory audits as numerical validation rather than theorem
  premises.

Primary sources:

- `src/memory_frontier/construction_time.py`
- `src/memory_frontier/preconditioned_construction.py`
- `tests/test_construction_time.py`
- `tests/test_scaffold_escape.py`
- `tests/test_preconditioned_construction.py`
- `experiments/construction_time_scaling.py`
- `experiments/dormant_scaffold_escape.py`

## Appendix F — Parameterization and simplex-boundary geometry

Separate two regimes explicitly.

### F.1 Regular local charts

Include:

- invariance of scalar vanishing order under a nonsingular local Jacobian;
- singular power-chart weighted-degree rule;
- distinction between objective order and Euclidean gradient trajectories.

Sources:

- `src/memory_frontier/parameterization.py`
- `tests/test_parameterization_order.py`

### F.2 Rare-edge softmax boundary

Include:

- exact binary-logit probability velocity
  \(\dot p=Cp^{d+1}(1-p)^2\);
- exact antiderivative and
  \(\tau_d\sim(Cd)^{-1}\delta^{-d}\);
- full `K`-way target-edge Jacobian norm formula and bounds;
- prior-art boundary on general softmax slowness.

Sources:

- `src/memory_frontier/softmax_boundary.py`
- `tests/test_softmax_boundary.py`
- `docs/softmax_boundary_bootstrap.md`

## Appendix G — Smooth linear state-space validation

Include:

- five-link recurrent delay line;
- exact gain `g=prod_i w_i`;
- exact loss improvement and missing-link gradient norm;
- autograd slopes for orders 1–5;
- exact-zero cutoff;
- fixed-step SGD threshold counts clearly labeled illustrative.

Sources:

- `tests/test_linear_ssm_validation.py`
- `experiments/linear_ssm_validation.py`

## Appendix H — Shared routes and conservation geometry

This appendix is supporting theory, not part of the main novelty spine.

Include:

- exponent-vector balance laws;
- exponent-support/nullspace characterization of diagonal quadratic invariants;
- shared-route rank-one mode solution;
- bilinear SVD mode decomposition;
- higher-order approximate invariant drift;
- finite-step GD breaking of continuous-flow invariants.

Sources include:

- `src/memory_frontier/support_invariants.py`
- `src/memory_frontier/shared_routes.py`
- `src/memory_frontier/bilinear_routes.py`
- `tests/test_exponent_support_invariants.py`
- `tests/test_shared_route_invariants.py`
- `tests/test_shared_route_mode_dynamics.py`
- `tests/test_bilinear_route_spectrum.py`
- `tests/test_approximate_invariant_drift.py`
- `tests/test_discrete_invariant_drift.py`

## Appendix I — Near-degeneracy failures of leading-order prediction

Include one scalar and one spectral adversarial example:

- higher-order terms reverse a near-tied scalar route race;
- higher-order terms rotate/reverse a nearly degenerate bilinear construction
  mode.

State the common margin principle in terms of the first correction degree
`p>d`, without claiming a universal trajectory theorem.

Sources:

- `src/memory_frontier/near_tie.py`
- `src/memory_frontier/spectral.py`
- `tests/test_near_tie_route_reversal.py`
- `tests/test_spectral_gap_reversal.py`
- `experiments/bilinear_route_spectrum.py`

## Appendix J — Earlier diagnostic results

Move the following here unless page budget allows a brief mention:

- exact capacity enumeration and predictive spectrum;
- source-synchronization correction;
- hard-forward / soft-backward gradient oracle;
- collapsed readout symmetry trap;
- unreachable-state counterfactual gradients;
- HardCellOracle and surrogate-vs-hard local minima;
- adiabatic and boundary geometry;
- edit alignment;
- decoder co-training details.

These results demonstrate breadth and helped discover the central theorem, but the
paper should not force the reader to process them before the construction-order
story is complete.

## Suggested main-text proof policy

Main text should contain:

1. definitions of the three construction orders;
2. one-paragraph proof of the hierarchy;
3. one-paragraph induction proof of dormant forward equivalence;
4. the exact single-class census protocol and counts;
5. the structural/metric separation with the two order-to-time tables;
6. one smooth-SSM validation panel;
7. one near-tie caution sentence or figure.

Everything else should be cross-referenced to the appendices above.
