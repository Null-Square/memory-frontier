# Paper spine: same predictor, different learnability

This document is a consolidation target, not a new theorem. Its purpose is to
separate the paper's central contribution from supporting mathematics and to make
remaining scientific gaps explicit.

## One-sentence thesis

> In learned finite-memory systems, predictive capacity and current forward
> behavior do not determine gradient accessibility: behaviorally dormant internal
> topology can change the derivative order at which a useful computation becomes
> visible, and therefore change its small-initialization construction-time class,
> without changing the current predictor.

A shorter version suitable for the introduction is:

\[
\boxed{\text{forward equivalence does not imply accessibility equivalence.}}
\]

The paper should avoid the broader and weaker claim "the same function can have
different optimization geometry." That phenomenon is already well established in
neural-network reparameterization and overparameterized embedding literature.

## Main theorem 1: construction-order hierarchy

For an affine finite-memory transition family, fixed source, finite horizon, and
fixed decoder-equality partition, define:

- \(d_{\rm support}\): the minimum number of perturbative transition edges on a
  source-valid product-state path to a different readout class;
- \(d_{\rm operator}\): the first nonzero degree of the exact quotient
  construction operator after equal decoder rows are collapsed;
- \(d_{\rm loss}\): the first nonzero degree of the scalar finite-horizon loss.

Then

\[
\boxed{
d_{\rm support}
\le d_{\rm operator}
\le d_{\rm loss}.
}
\]

Every exact scalar coefficient factors as

\[
\boxed{
c_\alpha
=-\langle G_\alpha,\log q\rangle.
}
\]

This theorem is the paper's mathematical spine because it separates three
conceptually different causes of apparent gradient inaccessibility:

1. **structural/source constraint:** no useful path exists below
   \(d_{\rm support}\);
2. **path/operator cancellation:** signed source-weighted contributions cancel,
   so \(d_{\rm operator}>d_{\rm support}\);
3. **decoder cancellation:** the decoder log-vector annihilates the first
   nonzero operator, so \(d_{\rm loss}>d_{\rm operator}\).

### Genericity corollary

Conditional on a fixed support cell and the stated analytic nondegeneracy
conditions, both strict inequalities are exceptional analytic cancellations. In
particular, with continuously sampled decoder values in a fixed equality
partition,

\[
\boxed{d_{\rm loss}=d_{\rm operator}\quad\text{almost surely}.}
\]

Under both nondegeneracy conditions,

\[
\boxed{
d_{\rm support}=d_{\rm operator}=d_{\rm loss}
\quad\text{almost surely}.}
\]

The explicit \((1,1,2)\) decoder-cancellation fixture stays in the main text or a
nearby proposition because it demonstrates why the operator level is necessary.

## Main theorem 2: dormant forward equivalence

Fix a source, reference controller, horizon, and initial memory. A transition row
\((m,x)\) is forward-active when the source-memory product process can occupy
memory \(m\) while symbol \(x\) has positive source probability before a
transition step.

If a second controller agrees with the reference on every forward-active row but
is arbitrary on all forward-dormant rows, then the two systems have identical
source-memory occupancies throughout the horizon:

\[
\boxed{
\widetilde\mu_t(s,m)=\mu_t(s,m)
\quad\forall t<T.
}
\]

Therefore their current prediction loss is identical for **every fixed decoder**,
not merely for one chosen scalar objective value.

This theorem makes the central comparison exact: the dormant structure can be
changed inside one current forward-equivalence class.

### Central corollary: same forward process, arbitrary construction order

A perturbative edge can make a previously dormant region reachable. Zero-cost
base wiring inside that region then changes the mixed zero-cost/unit-cost path
geometry used by \(d_{\rm support}\), even though it was irrelevant to the base
forward process.

The delay-chain family realizes, for any depth \(R\), forward-equivalent base
controllers with useful loss orders

\[
\boxed{1,2,\ldots,R.}
\]

The 1,000-instance single-class census strengthens this beyond a hand-designed
prefix family. It fixes source, architecture, current reachable controller,
decoder, trainable directions, and horizon, randomizes only dormant zero-cost
wiring, and obtains:

| exact order triple | count |
|---|---:|
| `(1,1,1)` | 235 |
| `(2,2,2)` | 282 |
| `(3,3,3)` | 244 |
| `(4,4,4)` | 155 |
| `(5,5,5)` | 84 |

with zero hierarchy violations.

This should be the paper's headline construction, because it isolates the only
changing object as behaviorally dormant topology.

## Main theorem 3: order-to-time map

For an isolated beneficial leading construction monomial of total degree \(d\),
symmetric/weighted-balanced gradient flow reduces exactly to

\[
\dot s=C s^{d-1}
\]

up to the appropriate multiplicity/metric prefactor. The completion-time classes
are

\[
\boxed{
\tau_d(\delta)
\sim
\begin{cases}
O(1), & d=1,\\
O(\log(1/\delta)), & d=2,\\
\Theta(\delta^{-(d-2)}), & d\ge3.
\end{cases}}
\]

The exponent itself is not a new general optimization result; related
small-initialization escape laws are known in deep homogeneous/deep-linear
systems. The finite-memory contribution is the exact mapping from latent
computation construction order to which member of this hierarchy applies.

The strongest consequence is qualitative rather than a constant-factor speedup:
behaviorally dormant rewiring can move a forward-equivalent system between

- finite bootstrap time,
- logarithmic bootstrap time,
- polynomially diverging bootstrap time.

## Main theorem 4: robustness boundary

The paper should close two obvious objections immediately after the dynamic
consequence.

### Regular coordinates cannot erase the order gap

For a scalar loss germ

\[
F(\varepsilon)-F(0)=P_d(\varepsilon)+O(\|\varepsilon\|^{d+1}),
\]

a smooth local reparameterization with nonsingular Jacobian preserves vanishing
order:

\[
\boxed{
\operatorname{ord}_0(F\circ\phi)
=
\operatorname{ord}_0 F.
}
\]

This is classical function-germ mathematics, not a novelty claim. Its role is to
show that the topology-induced scalar order separation is not removable by an
ordinary local coordinate change.

Singular charts can change order. For

\[
\varepsilon_i=\theta_i^{r_i},
\]

the exact pullback order is the weighted degree

\[
\boxed{
\min_{c_\alpha\ne0}\sum_i r_i\alpha_i.
}
\]

### Ordinary positive conditioning changes constants, not the isolated class

For one monomial with exponent vector \(\alpha\) and positive diagonal metric
\(M=\mathrm{diag}(m_i)\), the exact balance laws become

\[
\boxed{
\frac{x_i^2}{m_i\alpha_i}
-
\frac{x_j^2}{m_j\alpha_j}
=\text{constant},
}
\]

and on the metric-balanced manifold

\[
\dot s
=C\prod_i(m_i\alpha_i)^{\alpha_i/2}s^{d-1}.
\]

Thus ordinary positive diagonal conditioning changes the natural metric and time
prefactor but not the degree-controlled finite/log/polynomial class.

Again, the general homogeneous-scaling mechanism is prior art; this is a
robustness consequence for the finite-memory construction framework.

## Boundary theorem / main-text caution: competing routes

The paper should not leave readers with the impression that the smallest degree
or largest leading coefficient always predicts the complete later trajectory.

If the first correction that distinguishes routes has degree \(p>d\), then the
normalized leading-geometry error on the construction time scale is

\[
O(\delta^{p-d}).
\]

Therefore leading route labels require a margin. In scalar route competition, a
leading coefficient gap of order

\[
O(\delta^{p-d})
\]

can be reversed by higher-order terms. The exact non-delayed finite-memory
witness freezes such a reversal.

For coupled bilinear constructions, singular values are the exact quadratic
construction growth rates. A spectral gap satisfying

\[
\Delta\sigma
=O(\delta^{p-d})
\]

is likewise vulnerable to higher-order rotation; the exact spectral-gap fixture
freezes a mode reversal.

These results are best presented as **scope boundaries** for the leading-order
theory, not as additional central claims.

## What belongs in the main paper

A compact main text should contain roughly:

1. **Problem and distinction:** capacity vs current behavior vs accessibility.
2. **Exact finite-memory setup and loss polynomial.**
3. **Construction-order hierarchy theorem.**
4. **Dormant forward-equivalence theorem.**
5. **Same-forward arbitrary-order family + single-class random census.**
6. **Order-to-construction-time consequence.**
7. **Regular-coordinate / positive-conditioning robustness.**
8. **One route near-tie reversal as a caution.**
9. **Related work and limitations.**

Everything else should support these points rather than compete with them.

## What should move to appendices

Strong but secondary material includes:

- exact finite-memory capacity oracle and exhaustive small-K enumeration;
- source-validity synchronization correction;
- hard-forward/STE gradient oracle;
- collapsed-readout symmetry trap;
- accessibility cutoff and delayed higher-order blindness;
- hard counterfactual automata;
- gradient accessibility operator / intrinsic rank census;
- HardCellOracle and hybrid hard-cell dynamics;
- surrogate-vs-hard local minima;
- adiabatic flow and boundary-race geometry;
- joint decoder co-training details;
- exponent-vector balance laws;
- exponent-support/nullspace conservation laws;
- approximate invariant drift;
- finite-step GD breaking;
- shared-route exact solution;
- full bilinear SVD construction spectrum;
- spectral-gap reversal details.

These are valuable because they show the framework is not a one-fixture accident,
but most should not be headline claims.

## Closest prior art and claim boundary

### 1. Finite-state-controller gradient optimization

Finite-state controllers for partially observable environments have long been
optimized with gradient-based methods, and local-optimum / small-gradient issues
are known. The paper must not claim to discover that FSC memory gradients can be
small or zero.

Our narrower claim is an exact structural explanation of the **first useful
nonzero derivative degree**, including its source-aware factorization and dormant
forward-equivalence consequences.

### 2. Same-function neural-network embeddings

Fukumizu, Yamaguchi, Mototake, and Tanaka, *Semi-flat minima and saddle points by
embedding neural networks to overparameterization* (NeurIPS 2019), studies
unit-replication, inactive-unit, and inactive-propagation embeddings that realize
the same function while producing different flat/minimum/saddle landscape
geometry.

This is important adjacent prior art. Therefore the paper should **not** use
"same function, different optimization landscape" as its novelty sentence.

Our distinction is more specific:

- the forward-equivalence class is source-aware and finite-memory computational;
- the dormant wiring can be changed while source, decoder, architecture, current
  reachable dynamics, and even the trainable transition directions are fixed;
- the outcome is an exact integer-valued construction order with a
  support/operator/loss decomposition;
- that integer maps to a finite/log/polynomial bootstrap-time hierarchy.

### 3. Deep-linear / homogeneous small-initialization dynamics

Balancedness, singular-mode dynamics, and \(\delta^{-(d-2)}\)-type saddle escape
scalings have strong prior art. We use these mechanisms as consequences once the
finite-memory construction topology identifies \(d\); we do not claim the
optimization mathematics in isolation.

### 4. Reparameterization and path geometry

Parameterization dependence of Euclidean gradients, natural-gradient invariance,
and path-normalized methods are established. The paper's role is to show that the
scalar construction-order gap survives regular local charts, while singular
charts have a precise weighted-degree effect.

### 5. Spectral perturbation

Davis--Kahan/Wedin-type subspace sensitivity already explains why nearly
degenerate singular modes are unstable to perturbations. The finite-memory
spectral-reversal example is an exact realization and a caution on our leading
construction spectrum, not a new general perturbation theorem.

## Claims we should make

The strongest defensible claims are:

- finite-memory predictive capacity/current behavior and gradient accessibility
  are distinct notions;
- exact useful derivative order admits the support/operator/loss hierarchy;
- generic decoder values expose the first nonzero construction operator;
- source-aware dormant rewiring can preserve the complete current forward process
  while changing construction order;
- a single exact forward-equivalence class can realize a broad accessibility
  spectrum with fixed trainable directions;
- construction order determines the leading isolated small-initialization
  bootstrap class;
- the scalar order gap survives regular local coordinates and the isolated class
  survives positive diagonal conditioning;
- near degeneracy, higher-order terms can overturn route/mode labels, with a
  precise margin scale.

## Claims we should not make

Do not claim:

- vanishing gradients in memory systems are new;
- same-function parameterizations having different landscapes are new;
- balancedness or singular-mode dynamics are new;
- the \(d-2\) escape exponent is new general mathematics;
- all optimizers preserve the construction-time class;
- support/operator order is invariant under arbitrary nonlinear parameterization;
- the leading route always predicts the completed computation;
- the finite exact models directly prove the same mechanism dominates large
  neural sequence models.

## Proposed figures

### Figure 1: same forward process, different construction order

Show two or several controllers with identical reachable state-zero dynamics and
identical current predictor. Gray unreachable states contain different dormant
wiring. Overlay the shortest mixed construction path and label orders 1, 2, 4.
This should be the conceptual figure in the introduction.

### Figure 2: hierarchy and cancellation

Diagram

\[
d_{\rm support}\to d_{\rm operator}\to d_{\rm loss}
\]

with path cancellation and decoder cancellation as the two strict-gap mechanisms.
Include the generic equality statement.

### Figure 3: order-to-time hierarchy

Plot or schematic of completion time versus small initialization on log scales for
orders 1--5: finite, logarithmic, \(\delta^{-1}\), \(\delta^{-2}\),
\(\delta^{-3}\). Mark dormant rewiring as moving horizontally between classes
without changing current behavior.

### Figure 4: one-class random census

Bar chart of the 1,000 fixed-class random dormant topology counts:
235, 282, 244, 155, 84 for orders 1--5.

### Optional Figure 5: near-tie failure boundary

One exact reversal showing that higher-order corrections matter when the leading
margin is \(O(\delta^{p-d})\).

## Proposed paper title

Primary candidate:

**Same Predictor, Different Learnability: Construction Order in Finite-State Memory**

Alternates:

- **Capacity, Accessibility, and Construction Order in Learned Finite-State Memory**
- **Dormant Computation and Gradient Accessibility in Finite-State Memory**

## Remaining scientific gates

The theory is now sufficiently developed that only a small number of additions
would materially improve the paper.

### Gate A: independent model-family sanity check

Add one deliberately small differentiable recurrent/state-space experiment where
two initializations implement the same current predictor but have analogous
latent path depths. Test whether early gradient magnitude and construction time
show the predicted order dependence. This is external-validity evidence only,
not part of the proof.

### Gate B: systematic related-work audit

Search specifically for:

- higher-order/multiplicity analyses of inactive-unit embeddings;
- computational-path derivative order in automata/WFA/RNN learning;
- exact FSC results relating memory graph distance to derivative order;
- dormant/inactive subnetworks altering learning while preserving function;
- algebraic multiplicity / singular learning theory treatments of identical
  functions with different local vanishing orders.

The Fukumizu et al. line makes this audit essential before final novelty wording.

### Gate C: paper-level exposition proof pass

Rewrite the construction-order and dormant-equivalence proofs in notation suitable
for a paper, independent of implementation details. Verify every stated theorem's
assumptions against the regression fixtures and explicitly list failure cases.

After these gates, additional local lemmas should be added only if a reviewer-level
objection demands them.