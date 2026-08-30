# Conceptual taxonomy: what exactly differs between forward-equivalent memories?

The paper should keep five quantities separate. They answer different questions
and have different invariance properties.

| notion | question | paper object | structural or optimizer-dependent? | can dormant rewiring change it while current forward behavior is fixed? |
|---|---|---|---|---|
| **predictive capacity** | Can some controller in the architecture represent the desired memory computation? | architecture/state budget and attainable predictors | structural/global | not necessarily; dormant rewiring is inside the same architecture |
| **current forward behavior** | What predictor does the current base controller implement on source-valid histories? | source-memory occupancy process plus decoder | forward behavioral | **no** for a certified dormant rewire; this is the equivalence theorem |
| **construction visibility order** | At what first local scalar degree does the objective distinguish a latent computation? | \(d_{loss}\), with lower layers \(d_{support}\) and \(d_{operator}\) | local objective geometry; scalar order invariant under regular local charts | **yes** |
| **beneficial descent order** | At what first ray-leading degree is there an admissible loss-decreasing construction direction? | \(d_\downarrow\) on an admissible perturbation cone | local objective + feasibility cone | **yes**, when the leading visible form/sign changes |
| **bootstrap time** | How long does an optimizer take to construct the computation from scale \(\delta\)? | flow/iteration hitting time | parameterization, metric, optimizer, initialization and route coupling dependent | **yes**, mediated by construction order and optimizer geometry |

The central paper comparison fixes the first two columns of behavior—architecture
capacity and current source-conditioned predictor—while changing the third and,
in beneficial fixtures, the fourth. The fifth then follows only after an optimizer
geometry is specified.

## Construction hierarchy

The finite-memory theorem decomposes visibility order into

\[
\boxed{
d_{\rm support}
\le d_{\rm operator}
\le d_{\rm loss}.
}
\]

- `d_support` is a source-valid mixed path cost.
- `d_operator` asks whether minimal-cost signed occupancy effects survive after
  states with equal decoder rows are quotiented.
- `d_loss` asks whether the surviving occupancy effect couples to the numerical
  decoder log-vector.

Thus the hierarchy separates structural absence of a path, cancellation among
paths, and cancellation at readout.

## Visibility is not automatically descent

Let

\[
L(x)-L(0)=P_d(x)+P_{d+1}(x)+\cdots
\]

on admissible cone `K`. Then `d_loss=d` means only that `P_d` is nonzero somewhere
in parameter space. A useful construction requires an admissible ray with

\[
P_d(v)<0.
\]

If such a ray exists, beneficial order equals visible order. Otherwise the first
useful move may occur at a higher ray-leading degree.

This distinction is important at simplex boundaries, where local perturbation
coordinates may be one-sided.

## Order is structural; time is geometric

For the same beneficial isolated square-free degree-`d` construction:

### Affine probability-coordinate Euclidean flow

\[
\dot p=Cp^{d-1}.
\]

This produces finite, logarithmic, then `delta^{-(d-2)}` escape classes.

### Rare-edge Euclidean softmax flow

\[
\dot p=\Theta(p^{d+1}).
\]

This produces

\[
\tau_d=\Theta(\delta^{-d}).
\]

The time exponent is therefore not intrinsic to the latent computation. What is
common is the degree ordering: reducing construction order removes an
asymptotically severe small-initialization factor in either geometry.

The preferred summary diagram is

\[
\boxed{
\text{capacity}
\;
\text{and current behavior}
\quad\not\Rightarrow\quad
\text{construction accessibility}
}
\]

followed by

\[
\boxed{
\text{dormant topology}
\to
\text{construction order/local homogeneity}
\to
\text{optimizer geometry}
\to
\text{bootstrap time}.
}
\]

## Recommended manuscript language

Use:

- “same current predictor” or “forward-equivalent” for the dormant comparison;
- “visible at order `d`” for a nonzero first scalar degree;
- “beneficial order-`d` construction” only when the admissible sign condition is
  verified;
- “under affine probability-coordinate flow” or “under rare-edge softmax flow”
  before giving an escape exponent.

Avoid:

- “same parameters” — the dormant base wiring is intentionally different;
- “optimization difficulty is intrinsic to the function” — optimizer geometry
  matters;
- “construction order determines training time” without specifying the metric;
- “nonzero coefficient implies learnability” without a valid descent ray.
