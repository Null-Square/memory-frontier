# Related-work audit and final novelty boundary

This document is an adversarial novelty audit for the construction-order project.
Its purpose is not to maximize apparent novelty. It records the closest lines of
prior work, states what they already establish, and narrows the paper claim to the
part that survived the comparison.

The current thesis should be read against all of the work below:

> In finite-memory predictors, the **first useful nonzero local derivative degree**
> of a latent computation admits a source-aware support/operator/loss
> decomposition. Behaviorally dormant controller topology can change that local
> construction order inside one exact current forward-equivalence class, even
> when the source, architecture, decoder, reachable base dynamics, horizon, and
> trainable direction family are fixed.

The project should not use broader phrases such as “same function, different
optimization,” “memory gradients can vanish,” “sparse controller graphs improve
learning,” “finite-state-controller objectives are polynomial,” or “dormant
features activate in stages” as novelty claims.

## 1. Aberdeen and Baxter: zero-gradient regions in finite-state controllers

**Douglas Aberdeen and Jonathan Baxter, “Scaling Internal-State Policy-Gradient
Methods for POMDPs,” ICML 2002.**

This is the most important historical finite-state-controller predecessor found in
the audit.

The paper explicitly studies **zero-gradient regions** for policies with internal
state. It reports that small/random near-uniform controllers have difficulty
learning internal memory because observation histories induce nearly uniform
internal-state trajectories. The resulting gradient with respect to internal-state
parameters can be near zero.

More strongly, its Theorem 4 gives a symmetry condition under which the
internal-state gradient is exactly zero: if the internal transition distribution
is independent of current internal state and the action distribution is identical
across internal states, the gradient with respect to internal-state parameters
vanishes.

The same work proposes sparse internal-state transition graphs as a way to avoid
these small-gradient regions and empirically compares sparse and dense
controllers.

### Therefore we must not claim

- that zero memory gradients in FSCs are new;
- that uniform/symmetric memory controllers can be gradient-dead is new;
- that sparse FSC transition graphs can improve gradient learning is new;
- that internal-state topology affects trainability in a qualitative sense is new.

### What remains different

Aberdeen--Baxter studies first-order policy gradients around symmetric stochastic
controllers and methods for escaping those regions. The targeted audit did not
find an integer-valued **first useful Taylor degree** for a specified latent
computation, a source-valid shortest construction cost, a quotient occupancy
operator, or a forward-dormant rewire theorem.

Our finite-memory result is therefore not “why FSC gradients vanish” in general.
It asks a narrower question:

\[
\textit{if the first derivative vanishes, at what exact higher order does a
particular useful memory computation first become visible, and what structural
object determines that order?}
\]

## 2. Boularias and Chaib-draa: polynomial FSC/PSR value functions

**Abdeslam Boularias and Brahim Chaib-draa, “Predictive Representations for Policy
Gradient in POMDPs,” ICML 2009.**

This paper is the closest predecessor on **polynomial degree**.

For a finite horizon and direct probability coordinates, it shows that history
probabilities and the value functions of FSC and PSR policies are multivariate
polynomials in policy parameters. For a generic fully connected FSC it obtains a
history-probability degree of `2t+1`. It argues that an equivalent PSR can have
lower polynomial degree and consequently a simpler value-function landscape.

It also explicitly cites Aberdeen--Baxter's proposal to reduce FSC outdegree as a
way of reducing degree. Its toy example compares an FSC value function of degree
two with an equivalent PSR value function of degree one.

### Therefore we must not claim

- that finite-horizon FSC objectives are polynomial is new;
- that policy/controller representation can alter polynomial degree is new;
- that graph sparsity can reduce a global polynomial-degree measure is new;
- that lower polynomial degree may correlate with easier optimization is new.

### Crucial distinction: maximal/global degree vs local first useful degree

Boularias--Chaib-draa studies the **degree of the complete value-function
polynomial** (or of history probabilities): essentially the largest degree that
can occur under a representation over a horizon.

Our construction order is the opposite end of the Taylor support at a specified
base point:

\[
\boxed{
 d_{\rm loss}
 =\min\{|\alpha|>0:[\varepsilon^\alpha]L\ne0\}.
}
\]

It is the **lowest nonconstant local degree**, not the maximal polynomial degree.
The targeted text search of the 2009 paper found no discussion of Taylor order,
first nonzero coefficient, vanishing order, or a local expansion around
forward-equivalent controller bases.

The project further inserts a structural level not present in that global-degree
analysis:

\[
\boxed{
 d_{\rm support}
 \le d_{\rm operator}
 \le d_{\rm loss},
}
\]

where `d_support` is a source-valid mixed base/perturbative path cost and
`d_operator` is the first nonzero quotient occupancy operator. The factorization

\[
[\varepsilon^\alpha]L=-\langle G_\alpha,\log q\rangle
\]

then isolates path cancellation from decoder cancellation.

This distinction should be stated explicitly in the paper because “polynomial
degree of FSCs” by itself has clear prior art.

## 3. Braziunas and Boutilier: sequential failures of gradient search

**Darius Braziunas and Craig Boutilier, “Stochastic Local Search for POMDP
Controllers,” AAAI 2004.**

This paper identifies a basic sequential pathology of gradient-based optimization
for POMDP controllers and motivates non-gradient local moves that better mimic
dynamic-programming reasoning.

This is philosophically close to our observation that a useful multistep memory
construction can be invisible to a local gradient until several components are
jointly available.

### Therefore we must not claim

- that sequential controller construction can defeat ordinary gradient search is
  new;
- that local controller changes may need to be evaluated counterfactually is new
  at a qualitative level.

### What remains different

The 2004 work is an algorithmic/local-search treatment, not an exact Taylor-order
factorization. The targeted audit found no source-valid construction-cost theorem
or higher-order local derivative spectrum analogous to
`d_support <= d_operator <= d_loss`.

## 4. Same-function neural-network embeddings

**Kenji Fukumizu, Shoichiro Yamaguchi, Yoh-ichi Mototake, and Mirai Tanaka,
“Semi-flat minima and saddle points by embedding neural networks to
overparameterization,” NeurIPS 2019**, together with the later neural-network
**Embedding Principle** literature.

Fukumizu et al. construct wider-network embeddings by unit replication, inactive
units, and inactive propagation. These embeddings realize the same input-output
function as the narrower network while changing flat/minimum/saddle geometry.

### Therefore we must not claim

- “the same function can have different optimization geometry” as a new idea;
- that inactive parameters can change local landscape degeneracy while preserving
  the represented function is new;
- that function-equivalent embeddings can create saddle/flat directions is new.

### What remains different

Our intervention is not width embedding. Within a fixed finite-memory architecture
and a fixed current source-conditioned forward process, we rewire only
**source/horizon forward-dormant transition rows**. In the strongest random census,
the source, architecture, decoder, current reachable dynamics, horizon, and
trainable transition-direction family are all held fixed; only zero-cost dormant
wiring changes.

The resulting object is not merely Hessian signature or flatness. It is the exact
first useful derivative degree of a latent computation, with a combinatorial and
operator factorization.

## 5. Alternating Gradient Flows and dormant feature activation

**Daniel Kunin et al., “Alternating Gradient Flows: A Theory of Feature Learning
in Two-layer Neural Networks,” NeurIPS 2025 / arXiv:2506.06489.**

AGF analyzes small-initialization feature learning beginning with dormant neurons.
It describes alternating dormant-feature alignment and rapid activation, and
quantifies the order, timing, and magnitude of feature-acquisition events.

### Therefore we must not claim

- that dormant features activate sequentially under gradient flow is new;
- that small initialization induces staged feature learning is new;
- that activation order/timing of dormant components can be predicted is new in
  general neural-network dynamics.

### What remains different

AGF's “order” is an ordering of feature-acquisition events. Our construction
order is a **Taylor degree** of a specified computation at a fixed base
parameterization. It is computed from source-valid finite-memory transition
geometry, before solving the subsequent feature/route competition.

The route and spectral results in this repository are therefore best treated as
boundary/dynamics results, not as claims to supersede AGF-style feature-learning
theory.

## 6. Dead Directions and singular-learning order

**Tejas Pradeep Shirodkar, “Dead Directions: Geometric Singular Learning,”
arXiv:2606.05957, 2026.**

This work treats Fisher-degenerate parameter directions as singular geometric
objects with a definite **KL order**, determined by how rapidly KL divergence
vanishes along the dead direction. It connects that rate to Fisher curvature,
singular-learning invariants, deep-network geometry, and optimizer conditions.

### Therefore we must not claim

- that a degenerate parameter direction can have a higher-order vanishing rate is
  new;
- that “order” can characterize dead/singular directions is new generic
  mathematics;
- that higher-order flatness can be read as singular geometry is new.

### What remains different

Our theorem does not begin with an arbitrary singular direction or KL geometry.
It begins with a **latent finite-memory computation** and derives its lowest useful
multivariate degree from the source-memory construction graph, then inserts the
quotient operator to distinguish structural invisibility from cancellation.

The dormant-rewire theorem is also an intervention statement: it constructs
parameter points with the same complete current forward process but different
latent computation order.

## 7. Deep-linear/homogeneous dynamics

Deep-linear and homogeneous-network literature already supplies:

- balancedness and related conservation laws;
- singular-mode learning dynamics;
- saddle-to-saddle / small-initialization analyses;
- polynomial escape-time scalings such as the `d-2` exponent in homogeneous
  multiplicative bottlenecks.

### Therefore we must not claim

- the isolated monomial flow as new general mathematics;
- balancedness as new;
- the `delta^{-(d-2)}` escape exponent as new;
- SVD mode dynamics as new.

### What remains different

Those tools are consequences once our finite-memory theorem identifies which
local construction degree and which coupled leading operator a latent memory
computation has.

The linear-SSM experiment should be presented as **external validation**, not as a
new deep-linear theorem.

## 8. Reparameterization and optimization geometry

Order of vanishing of a scalar function germ under local diffeomorphism is
classical. Euclidean gradient trajectories depend on parameterization/metric;
natural/Riemannian-gradient covariance is also established.

### Therefore we must not claim

- regular-coordinate invariance of scalar vanishing order as new;
- parameterization dependence of ordinary Euclidean gradients as new;
- metric/preconditioning effects as new general optimization theory.

### What remains different

These facts close scope objections around the construction-order result:
regular local charts cannot erase the scalar order gap, while singular charts can
change it according to a weighted-degree law; positive diagonal conditioning
changes isolated-flow constants but not the degree-controlled escape class.

## 9. Exact novelty matrix

| Prior line | Already known | Do **not** claim | Distinct project object |
|---|---|---|---|
| Aberdeen--Baxter FSC gradients | uniform/symmetric FSCs can have zero or tiny memory gradients; sparse graphs help | vanishing memory gradients, sparse FSC benefit | exact higher local order of a specified latent computation |
| Boularias--Chaib-draa FSC/PSR | finite-horizon value is polynomial; representation/graph changes maximal degree | FSC polynomiality or graph-degree relation | **minimum nonconstant local degree** + support/operator/loss factorization |
| Braziunas--Boutilier | sequential structure can make gradient search poor | sequential gradient pathology | exact source-valid construction cost and higher-order visibility |
| Same-function NN embeddings | same function, different flat/saddle geometry | same function/different landscape | dormant finite-memory rewiring inside one source-conditioned forward-equivalence class |
| AGF | dormant features activate in predictable stages | dormant-feature activation order/timing | Taylor degree before a latent computation becomes locally visible |
| Dead Directions | singular/dead directions have KL vanishing order | generic higher-order dead directions | computation-specific graph/operator derivation and intervention |
| Deep-linear/homogeneous | multiplicative bottleneck dynamics and escape exponents | balancedness, SVD modes, `d-2` exponent | topology-to-degree map for finite-memory computations |

## 10. Strongest defensible novelty claim after the audit

The final paper should use a statement close to the following:

> **Construction-order theorem.** For an affine finite-memory transition family
> around a construction origin, the first useful nonzero derivative degree of a
> latent predictive computation is bounded and resolved by a source-aware
> hierarchy
> \(d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}\), where the first gap
> is path/operator cancellation and the second is decoder cancellation.
> Moreover, source/horizon forward-dormant rewiring can change these local orders
> while preserving the complete current source-memory occupancy process; a single
> exact forward-equivalence class with a fixed trainable direction family realizes
> a broad order spectrum.

The dynamic consequence can then be stated separately:

> Once a leading isolated construction has degree `d`, known homogeneous-flow
> mathematics maps that degree to the finite/logarithmic/polynomial
> small-initialization bootstrap hierarchy.

This division is important: the first paragraph is the candidate original
finite-memory contribution; the second deliberately imports established
optimization dynamics.

## 11. What the targeted audit did **not** find

As of this audit, targeted searches across FSC/POMDP policy-gradient work,
finite-state controller optimization, same-function neural embeddings,
singular-learning geometry, and dormant-feature dynamics did **not locate** a
prior theorem combining all of the following:

1. a local lowest nonconstant derivative order around a fixed finite-memory base;
2. a lower bound given by **source-valid** mixed base/perturbative path cost;
3. an intermediate quotient occupancy operator that exactly identifies signed
   construction cancellation;
4. a decoder factorization that isolates readout cancellation;
5. a source-aware dormant-rewire theorem preserving the entire current occupancy
   process;
6. an order spectrum produced inside one forward-equivalence class while the
   trainable direction family is fixed.

This is not proof of novelty. It is the defensible conclusion of the current
search and should be phrased as such until final bibliography/reviewer checks.

## 12. Manuscript implications

The related-work section should discuss Aberdeen--Baxter and
Boularias--Chaib-draa **before** the modern neural-network analogies, because they
are the closest domain-specific predecessors.

A clean framing is:

1. prior FSC work establishes zero-gradient symmetry regions and sparse-graph
   remedies;
2. prior FSC/PSR work establishes global polynomial-degree effects of
   representation;
3. this paper asks a different local structural question: **what is the first
   useful degree at one current controller, and which latent computation causes
   it?**;
4. the answer is the support/operator/loss construction hierarchy plus dormant
   forward equivalence;
5. modern same-function, singular-learning, and dormant-feature theories place
   that result in a broader landscape-geometry context.

That ordering makes the novelty claim both narrower and stronger.
