# Leading computation polynomial: what it predicts and what it does not

The multivariate loss polynomial is the right local object for independent
memory-construction parameters, but a first adversarial test shows that its
smallest weighted monomial does **not** by itself determine which computation
training builds first.

This is a useful falsification. It sharpens the next bridge from local
construction order to full algorithm formation.

## Question

Suppose

\[
L(\varepsilon)=L_0+\sum_\alpha c_\alpha\varepsilon^\alpha,
\]

and the parameters are initialized at unequal small scales

\[
\varepsilon_j=t^{w_j}.
\]

Along that initialization ray, a monomial has weighted loss order

\[
w^\top\alpha.
\]

A tempting conjecture is:

> the useful leading monomial with minimum weighted order identifies the
> computation that gradient training constructs first.

The exact branching fixture below falsifies that conjecture.

## Exact parallel-route fixture

Use the binary delay-2 repeat source with switch probability `0.1` and horizon
`T=12`. Start from collapsed memory state 0 and provide two disjoint dormant
routes to identically useful final decoders:

```text
route A: 0 --x--> A1 --y--> A2
route B: 0 --u--> B1 --v--> B2
```

Both entrance links are triggered by token 0. Both second links advance on
either token. `A2` and `B2` use the same decoder; every other memory state uses
the uniform decoder.

The exact multivariate oracle gives the leading loss terms

\[
L-L_0
=
-Cxy-Cuv+O(\|\varepsilon\|^3),
\]

with

\[
C\approx0.100852714422.
\]

Thus the two computational routes have exactly the same leading predictive
coefficient.

## Unequal initialization scales

Initialize

\[
x=y=t,
\qquad
u=t^{0.1},
\qquad
v=t^2.
\]

The weighted loss orders are

\[
\operatorname{ord}(xy)=2,
\qquad
\operatorname{ord}(uv)=2.1.
\]

So minimum weighted loss order predicts route A.

But differentiating removes one factor:

\[
\partial_xL\sim-Cy\sim-Ct,
\qquad
\partial_yL\sim-Cx\sim-Ct,
\]

while

\[
\partial_uL\sim-Cv\sim-Ct^2,
\qquad
\partial_vL\sim-Cu\sim-Ct^{0.1}.
\]

The small `v` parameter in route B therefore receives a much larger initial
update than either route-A parameter.

At `t=10^-12`, the exact full finite-horizon polynomial gives an initial
`v`-gradient magnitude of about `6.19e-3`, while the route-A gradient
magnitudes are about `9.81e-14`.

A deterministic projected-gradient race on the **full exact polynomial** with
learning rate `0.02` and route-completion threshold `0.02` builds route B first:

| initialization scale | weighted-loss prediction | first route built | steps |
|---:|---|---|---:|
| `1e-8` | A | B | 67 |
| `1e-12` | A | B | 160 |
| `1e-20` | A | B | 722 |

No sampling or random seed is involved.

This establishes an exact finite-memory counterexample to the naive pipeline

\[
\text{minimum weighted loss monomial}
\Longrightarrow
\text{first learned computation}.
\]

## Why the conjecture fails

The loss polynomial is a scalar object. Training follows its vector field:

\[
\frac{d\varepsilon_j}{ds}
=-\frac{\partial L}{\partial\varepsilon_j}.
\]

For a monomial `c_alpha epsilon^alpha`, coordinate `j` sees exponent

\[
\boxed{
w^\top\alpha-w_j
}
\]

whenever `alpha_j>0`.

Therefore two routes can be ranked one way by their loss contribution and a
different way by the coordinate-wise gradients that assemble them. Once the
parameters move, the initialization-ray exponents are no longer sufficient
either; the full continuous trajectory and probability-simplex competition
matter.

The refined hierarchy is:

1. **loss support** — which computational monomials exist and with what signs;
2. **gradient support** — which coordinates each monomial exposes and at what
   derivative order;
3. **continuous dynamics** — how those coordinate updates bootstrap or compete;
4. **algorithm/boundary event** — which computation becomes behaviorally active
   first.

This reconnects the multivariate construction-order theory with the earlier
boundary-geometry result: static local accessibility is not the same object as
the trajectory that reaches a new hard algorithm.

## Reused parameters and exponent multiplicity

The exact polynomial also confirms that exponent vectors must be multisets, not
simple supports.

If one trainable parameter controls both missing links of the delay-2 chain,
the first useful term appears at exponent

\[
(2),
\]

not `(1)`. Thus exponent entries greater than one are operationally relevant
whenever one parameter is reused by a contributing computation.

This does not imply parameter tying is innocuous: tying changes the
parameterization and can create cancellation. It only shows that the polynomial
representation correctly records multiplicity when a parameter is reused.

## Cancellation must be applied before support extraction

Structural path counting is insufficient. Distinct source-conditioned
contributions can cancel exactly at the same exponent.

The regression suite includes a tied one-parameter fixture whose two first-order
branches have equal and opposite contributions. After coefficient aggregation,
all nonconstant coefficients vanish at the tested horizon. The correct leading
support is therefore empty, even though structural perturbative routes exist.

So the object used for prediction must be the **coefficient-cancelled exact
polynomial**, not a graph-theoretic list of candidate paths.

## Scientific status

### Proven / exact

- the finite-horizon loss is an exact multivariate polynomial for affine
  transition perturbations;
- the parallel-route fixture has leading support `xy` and `uv` with equal
  negative coefficients;
- under the stated unequal initialization scales, route A has smaller weighted
  loss order but route B has a much stronger gradient coordinate;
- the reused-parameter fixture has a leading exponent entry greater than one;
- exact coefficient cancellation can erase structurally present support.

### Deterministic finite experiment

- projected gradient descent on the exact full polynomial builds route B first
  for the three frozen initialization scales reported above.

### Not established

- a general predictor of first learned computation for arbitrary branching
  memory graphs;
- whether initial gradient-order information is sufficient once routes share
  parameters or simplex constraints strongly;
- whether the same route-ranking failure persists with jointly learned decoders
  and non-delayed stochastic languages.

## Refined next target

The next bridge should therefore not be

\[
\text{leading loss polynomial}
\to
\text{learned computation}.
\]

A better target is

\[
\boxed{
\text{target memory graph}
\to
\text{exact loss polynomial}
\to
\text{exact gradient polynomial / vector field}
\to
\text{predicted route or boundary race}
\to
\text{observed first algorithm}.
}
\]

The immediate falsification questions are now sharper:

- can a route-race predictor built from the gradient polynomial forecast the
  first computation across arbitrary small branching graphs?
- when does the initial asymptotic gradient ranking fail because later
  continuous dynamics reverse the race?
- how should shared parameters and probability-simplex coupling enter that
  predictor?
- does joint decoder learning introduce new low-order routes or cancellations?
- can the same phenomenon be reproduced on a non-delayed source?
