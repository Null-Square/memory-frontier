# Finite steps collapse shared-route balancedness to a matched ray

The finite-step defect has a particularly clean form for a family of computations
sharing one entrance parameter.

Consider

\[
L=-x\sum_{j=1}^m a_j y_j,
\qquad a_j>0,
\]

with balance law

\[
Q=x^2-\sum_{j=1}^m y_j^2.
\]

Every supported exponent vector contains the shared entrance `x` and exactly one
exit `y_j`, so the continuous Euclidean gradient flow conserves `Q` exactly.

## Exact finite-step defect on the balanced cone

Write `a=(a_1,...,a_m)` and `y=(y_1,...,y_m)`. The gradient is

\[
\partial_xL=-a\cdot y,
\qquad
\partial_{y_j}L=-a_jx.
\]

Therefore the finite-step defect polynomial is

\[
D=(a\cdot y)^2-\lVert a\rVert^2x^2.
\]

On the continuously balanced cone `Q=0`, where

\[
x^2=\lVert y\rVert^2,
\]

one gradient-descent step has exact balance change

\[
\boxed{
\Delta Q
=
\eta^2\left[(a\cdot y)^2-
\lVert a\rVert^2\lVert y\rVert^2\right].
}
\]

By Cauchy-Schwarz,

\[
\boxed{
\Delta Q
=-\eta^2\lVert a\rVert^2
\lVert y_{\perp a}\rVert^2
\le0.
}
\]

Thus vanilla finite-step gradient descent does **not** preserve the full balanced
cone. Equality holds exactly when the exit vector is parallel to the route
coefficient vector,

\[
\boxed{y\parallel a.}
\]

For two exits,

\[
L=-axy-bxz,
\]

the formula reduces to

\[
\boxed{
\Delta Q=-\eta^2(by-az)^2
}
\]

on `x^2-y^2-z^2=0`.

## The coefficient-matched ray is exactly invariant under finite steps

Take

\[
y=s a,
\qquad
x=s\lVert a\rVert.
\]

This lies on the balanced cone. One vanilla gradient-descent step gives

\[
\boxed{
(x^+,y^+)=(1+\eta\lVert a\rVert)(x,y).
}
\]

So the coefficient-matched balanced ray is preserved exactly and simply expands
by a common multiplicative factor. Continuous gradient flow has an entire
quadratic family of conserved level sets; finite-step gradient descent selects a
much smaller exact balanced geometry.

`tests/test_discrete_shared_route_geometry.py` freezes the two-exit case and
checks both the negative-square defect and exact preservation of the
coefficient-matched ray.

## Interpretation

This supplies a concrete route-coupling effect of discretization. In continuous
flow, balancedness constrains the shared entrance against the aggregate exit
norm. Under finite steps, generic points on that cone are pushed toward

\[
Q<0,
\]

meaning the exit norm becomes larger than the shared-entrance magnitude in this
quadratic metric. Only the route-mixture direction matched to the loss
coefficients avoids this finite-step imbalance.

The result is exact for vanilla gradient descent on the stated quadratic route
family. It is parameterization- and optimizer-specific and should not be read as
a general property of learned finite-state systems.

## Prior-art boundary

Finite-learning-rate breaking of symmetry-derived gradient-flow conservation laws
is established in prior work. The point here is the exact shared-computation
specialization: a finite-memory route polynomial makes the surviving discrete
balanced ray and its coefficient dependence explicit.
