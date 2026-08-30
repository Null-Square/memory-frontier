# Exponent-vector construction flow

Leading computation polynomials need exponent vectors, not simple supports,
because a predictive computation can reuse one trainable transition parameter
multiple times. The same multiplicity information also changes the local
gradient-flow geometry.

For one isolated beneficial monomial, the square-free balancedness theorem has a
simple exact extension to arbitrary positive exponent vectors.

## General monomial

Let

\[
L-L_0
=-C\prod_{i=1}^p x_i^{\alpha_i},
\qquad
C>0,
\qquad
\alpha_i\in\mathbb N_{>0}.
\]

The exponent vector

\[
\boldsymbol\alpha=(\alpha_1,\ldots,\alpha_p)
\]

records parameter multiplicity. Its total degree is

\[
d=\sum_i\alpha_i.
\]

Gradient flow gives

\[
\dot x_i
=C\alpha_i x_i^{\alpha_i-1}
\prod_{j\ne i}x_j^{\alpha_j}.
\]

Equivalently, if

\[
P(x)=\prod_jx_j^{\alpha_j},
\]

then for positive coordinates

\[
\dot x_i=C\alpha_i\frac{P(x)}{x_i}.
\]

`monomial_gradient_flow_velocity` evaluates this vector field directly and also
handles zero coordinates without dividing by them.

## Weighted squared invariants

Different multiplicities make the raw squared coordinates move at different
rates:

\[
\frac{d}{dt}x_i^2
=2C\alpha_iP(x).
\]

But after dividing by multiplicity,

\[
\boxed{
\frac{d}{dt}\frac{x_i^2}{\alpha_i}
=2CP(x)
}
\]

for every active coordinate. Therefore

\[
\boxed{
\frac{x_i^2}{\alpha_i}
-
\frac{x_j^2}{\alpha_j}
=\text{constant}
}
\]

along the isolated monomial gradient flow.

The natural balanced variables are thus

\[
y_i=\frac{x_i^2}{\alpha_i},
\]

not the unweighted squares. `normalized_squared_coordinates` exposes these
coordinates explicitly.

The square-free theorem is the special case `alpha_i=1`, where this reduces to
the ordinary conserved differences `x_i^2-x_j^2`.

## Weighted-balanced manifold

If all normalized squared coordinates start equal, write

\[
\boxed{
x_i=\sqrt{\alpha_i}\,s.}
\]

This weighted-balanced manifold is invariant. Define

\[
A_{\alpha}
=\prod_i\alpha_i^{\alpha_i/2}.
\]

Then

\[
P(x)=A_{\alpha}s^d
\]

and the entire exponent-vector flow reduces exactly to

\[
\boxed{
\dot s
=C A_{\alpha}s^{d-1}.
}
\]

Thus the multiplicity pattern enters through the prefactor
`A_alpha`, while the power of `s` depends only on total degree `d`.

`balanced_exponent_vector_completion_time` implements the resulting exact
completion time for a normalized scale `s`.

## Escape-time consequence

For fixed normalized threshold and small balanced initialization `delta`, every
exponent vector with the same total degree has the same qualitative escape-time
class:

\[
\tau=O(1),\qquad d=1,
\]

\[
\tau=O(\log(1/\delta)),\qquad d=2,
\]

and

\[
\boxed{
\tau=\Theta(\delta^{-(d-2)}),
\qquad d\ge3.
}
\]

Multiplicity changes the constant through `A_alpha` but not the small-scale
power. For example, the degree-four exponent vectors

```text
(1,1,1,1)
(2,1,1)
(2,2)
(3,1)
(4)
```

all have `delta^-2` balanced escape-time divergence even though they correspond
to very different parameter reuse patterns.

This sharpens the distinction between **construction order** and
**construction metric**:

- total exponent degree controls the local derivative/escape exponent;
- exponent allocation controls how Euclidean parameter motion is distributed
  across the reused factors.

## Relation to parameter tying

A single reused scalar parameter with multiplicity `d` is the exponent vector

\[
(d).
\]

Its weighted-balanced coordinate is

\[
s=x/\sqrt d.
\]

The generalized formula exactly matches
`tied_repeated_parameter_completion_time` after translating between raw and
normalized coordinates. This recovers the earlier result that tying a
square-free degree-`d` diagonal route into one scalar changes Euclidean gradient
speed even though the diagonal scalar loss curve remains `-C x^d`.

The important point is that repeated use and parameter tying are not invisible
once dynamics are considered: the exponent vector is part of the optimization
geometry.

## Prior-art boundary

Conserved squared-norm differences and balanced manifolds are established in
deep linear and homogeneous-network gradient-flow theory. The weighted
exponent-vector identity above is an elementary monomial extension of that same
multiplicative geometry and is not claimed as a new general optimization
phenomenon.

The finite-memory role is narrower: the exponents come from exact
source-conditioned computational walks in a finite-memory predictor. A repeated
transition parameter can therefore be interpreted simultaneously as

1. a multiplicity in the exact leading computation polynomial;
2. a weight in the conserved construction-flow metric; and
3. a contributor to the total construction order that controls the local
   bootstrap-time class.

This gives the repeated-edge falsification question a concrete dynamical answer
without reducing the computation to a simple graph path.

## Claim boundary

### Exact / algebraic and regression-tested

- for one beneficial monomial, `x_i^2/alpha_i-x_j^2/alpha_j` is conserved;
- the weighted-balanced manifold `x_i=sqrt(alpha_i)s` is invariant;
- on that manifold, the flow reduces to
  `ds/dt = C*A_alpha*s**(d-1)`;
- total degree controls the balanced small-initialization escape exponent;
- the one-coordinate exponent vector `(d)` agrees exactly with the existing
  tied-parameter completion-time oracle after coordinate normalization.

### Not established

- conservation of these quantities for sums of competing monomials;
- that total degree alone predicts route races with shared parameters;
- invariance under optimizer or parameterization changes;
- an exact mapping from every finite-memory computational walk to one isolated
  monomial once coefficient cancellation and branching interactions are present.

The next adversarial target is therefore **shared-parameter competing routes**,
where multiple leading monomials act on the same weighted-balanced coordinates
and these isolated-monomial invariants generally no longer hold.
