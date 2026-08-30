# Focused related-work boundary

This note records the closest conceptual neighbors found in the targeted literature
audit. It is deliberately conservative: the purpose is to remove broad novelty
claims that the literature already supports and isolate the narrower contribution
that remains project-specific.

## Executive boundary

The following ideas are **not** new in isolation:

- the same input-output function can have different parameter-space geometry;
- inactive or surplus neural units can preserve a function while creating flat,
  minimum, or saddle directions;
- wider/deeper neural networks can contain critical embeddings of narrower or
  shallower networks with the same output function;
- singular models have directions along which KL/loss vanishes at higher order;
- small-initialization homogeneous networks can remain near degenerate saddles for
  long times and activate features sequentially;
- dormant neurons can activate one-by-one and create staircase learning dynamics;
- balancedness, singular-mode growth, and polynomial small-initialization escape
  times occur in deep homogeneous models;
- Euclidean optimization depends on parameterization and metric.

The strongest project-specific claim should therefore be stated at the
**finite-memory computational level**, not as a generic statement about function
space versus parameter space.

## Comparison table

| Adjacent line | What is already established | What remains distinct here |
|---|---|---|
| Finite-state-controller gradient learning | Memory policies/controllers can be optimized by gradients; local optima and weak/zero memory gradients can occur. | Exact source-aware factorization of the first useful transition-construction degree into support, quotient operator, and decoder cancellation. |
| Fukumizu et al., NeurIPS 2019, *Semi-flat minima and saddle points by embedding neural networks to overparameterization* | Unit replication, inactive units, and inactive propagation embed the same network function into a wider model while changing local flat/minimum/saddle geometry. | Same source, same finite-memory architecture, same current reachable controller, same decoder, and even the same trainable transition directions can be held fixed while only unreachable/dormant zero-cost wiring changes the exact useful derivative degree. |
| Zhang et al., NeurIPS 2021 / JML 2022, *Embedding Principle* | Critical points of narrower networks can be embedded in wider networks with the same output function and increased degeneracy; Hessian inertia/critical manifolds are studied. | Our comparison is not width/depth lifting of a critical point. It is source-conditioned computational topology inside one finite-state forward-equivalence class, with an integer construction-order hierarchy and path/operator factorization. |
| Bai et al., depth embedding principle | Shallower-network critical points lift into deeper networks with the same outputs and greater degeneracy; training may encounter the lifted manifolds. | Our order is tied to a specific latent memory computation and shortest source-valid mixed base/perturbative paths, rather than architecture-depth embedding of an already critical function. |
| Singular learning theory / local learning coefficients | Non-identifiability produces analytic singular sets; KL divergence can vanish at high order, and RLCT/LLC measure local effective complexity. | Construction order is not proposed as a replacement RLCT. It identifies the first total derivative degree of a useful finite-memory computation and decomposes that degree combinatorially through source-valid transition topology and decoder quotient classes. |
| Shirodkar 2026, *Dead Directions: Geometric Singular Learning* | A dead direction is assigned a KL order from the rate at which KL vanishes; Fisher curvature decay recovers that order, and depth can produce higher KL order. | We should not claim the generic notion of a high-order dead direction. Our contribution is the exact finite-memory topology-to-order oracle, multivariate useful-computation degree, dormant forward-equivalence intervention, and source/readout factorization. |
| Jacot et al. and deep-linear saddle-to-saddle work | Small initialization around depth-induced degenerate saddles yields balanced dynamics, singular-mode selection, and diverging escape times. | The degree entering those dynamics is derived here from a finite-memory computation graph rather than architectural depth alone; dormant topology can change it without changing the current forward process. |
| Kunin et al., NeurIPS 2025, *Alternating Gradient Flows* | Dormant neurons activate sequentially; feature-acquisition order, timing, and magnitude can be modeled in small-initialization two-layer/homogeneous settings. | We should not claim staged dormant feature activation. Our result concerns how forward-irrelevant finite-state wiring changes the derivative degree of an identifiable latent memory computation inside a fixed current predictor. |
| Davis--Kahan / Wedin perturbation theory | Nearly degenerate eigenspaces/singular subspaces rotate strongly under perturbations. | The spectral-gap reversal is an exact finite-memory realization and scope boundary, not a general perturbation theorem. |
| Path-SGD / natural-gradient / information geometry | Same-function rescalings can yield different Euclidean gradients; metric-aware methods address reparameterization dependence. | Regular local charts preserve our scalar loss-germ order, while singular power charts have an exact weighted-degree pullback. Positive diagonal conditioning changes the isolated-flow prefactor but not the degree class. |

## The distinction from neural-network embedding work

The closest conceptual overlap is the same-function embedding literature.
Fukumizu et al. explicitly constructs surplus inactive units and inactive
propagation that preserve the narrower network function, and analyzes whether the
embedded parameter is a flat minimum or saddle. Zhang et al. and follow-up work
prove broad critical-embedding principles across width and depth.

The finite-memory result should therefore **not** be introduced as

> two parameter settings can compute the same function but have different local
> landscapes.

That sentence is prior art.

The sharper statement is:

> For a fixed source and horizon, the exact rows of a finite-state transition
> system that are never exercised by the current source-memory product process may
> be rewired without changing that process at all. Holding the source, architecture,
> decoder, current reachable dynamics, and local trainable transition directions
> fixed, such dormant rewiring can nevertheless change the minimum number of
> perturbative transition factors needed to expose an informative readout class,
> and hence change the first useful scalar derivative degree.

That statement has three ingredients absent from the broad embedding result:

1. **source-aware computational reachability**, not only function-preserving
   algebraic embedding;
2. **construction topology**, with zero-cost base edges and unit-cost learnable
   edges;
3. an exact **support → quotient operator → scalar order** decomposition.

The single-class 1,000-sample census makes the intervention especially clean:
only dormant base wiring changes.

## The distinction from singular learning theory

Singular learning theory is highly relevant because our collapsed/dormant points
are degenerate and because loss/KL can vanish to high order. Recent work even uses
"KL order" explicitly for dead directions. We should cite this and avoid
terminological novelty claims.

However, the mathematical objects answer different questions.

### SLT-style question

Near a parameter realizing a target distribution, how singular is the model
family? How rapidly does KL grow away from the singular set, and what Bayesian or
Fisher invariants follow?

### Construction-order question

Starting from a current finite-memory predictor that may be suboptimal, what is
the first multivariate degree at which a **particular useful memory computation**
can affect finite-horizon prediction loss? Which source-valid transition paths
create that coefficient, which path contributions cancel, and which decoder
contrasts expose it?

Construction order can therefore be computed and interpreted even when the base
point is not a population optimum or a point on a target-realization singular set.
It also keeps the exponent vector of the latent computation, not only a
one-dimensional approach order.

This distinction should be made explicit in related work.

## The distinction from dormant-feature dynamics

Alternating Gradient Flows is close in language because it starts with dormant
neurons and predicts which feature activates next. The conceptual overlap is real:
both projects treat learning as construction/activation of latent computation from
small initialization.

The finite-memory theory differs in what determines accessibility:

- AGF utility is defined by the current residual/feature geometry of homogeneous
  neurons;
- our first useful degree is determined by source-valid paths through a finite
  transition graph, quotienting behaviorally indistinguishable memory states;
- dormant topology can be modified while preserving the entire current forward
  process and all trainable directions, thereby changing the degree itself.

A productive final paper should present these as complementary theories rather
than competitors: AGF addresses staged feature learning in neural models; the
present work gives an exact computational-topology oracle in learned memory.

## Candidate novelty sentence

A defensible introduction sentence is:

> We give an exact finite-memory construction-order theory that maps source-aware
> latent transition topology to the first useful loss derivative degree, and show
> that this degree can vary arbitrarily inside a single current forward-equivalence
> class through rewiring that is behaviorally dormant at initialization.

A stronger but still defensible follow-up is:

> For generic decoder values the structural construction order is visible in the
> scalar loss, and in isolated leading-route regimes it selects the finite,
> logarithmic, or polynomial small-initialization bootstrap-time class.

## Claims to avoid after this audit

Avoid all of the following:

- "We are the first to show same-function models can optimize differently."
- "We discover higher-order dead directions."
- "We discover dormant feature activation."
- "We discover that depth/path products create high-order saddles."
- "We discover polynomial saddle-escape times."
- "We discover parameterization dependence of gradients."
- "We discover that inactive units create flat directions."

The paper is stronger when those mechanisms are treated as context and the
finite-memory computational factorization is stated precisely.