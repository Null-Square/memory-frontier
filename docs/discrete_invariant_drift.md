# Finite learning rate breaks balance laws through a separate exact defect

The exponent-support results describe conservation laws of **continuous Euclidean
gradient flow**. Actual training uses finite steps. Even if a quadratic balance
law is exact for the complete loss polynomial, vanilla gradient descent need not
preserve it.

This gives a second, algebraically separate source of balance-law drift from the
higher-order loss terms studied in `approximate_invariant_drift.md`.

## Exact one-step decomposition

Let

\[
Q_w(x)=\sum_i w_i x_i^2,
\qquad
g(x)=\nabla L(x),
\]

and take one vanilla gradient-descent step

\[
x^+=x-\eta g(x).
\]

Because `Q_w` is quadratic, expansion is exact rather than asymptotic:

\[
\boxed{
Q_w(x^+)-Q_w(x)
=
-2\eta\sum_i w_i x_i g_i
+
\eta^2\sum_i w_i g_i^2.
}
\]

The first term is exactly `eta` times the continuous gradient-flow derivative,

\[
\dot Q_w=-2\sum_i w_i x_i g_i.
\]

Define the **discretization-defect polynomial**

\[
\boxed{
D_w(x)=\sum_i w_i\bigl(\partial_iL(x)\bigr)^2.
}
\]

Then

\[
\boxed{
\Delta_\eta Q_w
=
\eta\dot Q_w+
\eta^2D_w.
}
\]

`quadratic_invariant_discretization_defect_coefficients` constructs `D_w`
exactly as a sparse polynomial after exponent aggregation, and
`quadratic_invariant_discretization_degree` returns its first nonzero total
degree.

If `Q_w` is an exact continuous-time invariant of the complete polynomial, the
one-step violation is simply

\[
\boxed{
\Delta_\eta Q_w=\eta^2D_w(x).
}
\]

So continuous conservation does not imply discrete conservation.

## Exact two-link example

For

\[
L=-Cxy,
\qquad
Q=x^2-y^2,
\]

continuous gradient flow conserves `Q` exactly. One gradient-descent step gives

\[
x^+=x+\eta Cy,
\qquad
y^+=y+\eta Cx,
\]

and therefore

\[
\boxed{
Q^+=(1-\eta^2C^2)Q.
}
\]

The balance error is geometrically contracted rather than conserved. The finite
step happens to preserve the exactly balanced manifold `Q=0`, but not the full
family of continuous-flow level sets.

This is an optimizer statement, not a property of the represented predictor.

## Local accumulated scaling

Suppose the first nonconstant loss degree is `d`, and let `r` be the first total
degree of `D_w` after exact cancellation. At `x=delta*y`, one finite-step defect
has size

\[
\eta^2 O(\delta^r).
\]

A fixed window in the natural local construction time

\[
t=O(\delta^{2-d})
\]

contains `O(delta^(2-d)/eta)` vanilla-GD steps. Provided the rescaled trajectory
remains bounded and the step is small enough for the local accumulation
argument, the normalized discrete contribution scales as

\[
\boxed{
\Delta Q_w/\delta^2
=
O\!\left(\eta\,\delta^{r-d}\right).
}
\]

Generically a degree-`d` loss has gradient degree `d-1`, so `D_w` begins at
`2d-2`. Exact cancellations can raise `r`; therefore `r` should be computed from
the aggregated defect polynomial rather than assumed.

## Competing with higher-order loss drift

The previous result gives continuous full-loss drift

\[
\frac{\Delta Q_w}{\delta^2}
=O(\delta^{p-d}),
\]

where `p` is the first loss degree that violates the leading balance law.
The finite-step contribution is

\[
O(\!\left(\eta\delta^{r-d}\right)).
\]

Thus the relative asymptotic sizes are controlled by **two independent breaking
orders**:

- `p`: first support degree that breaks the balance law in the loss itself;
- `r`: first degree of the optimizer discretization-defect polynomial.

Their formal crossover scale is

\[
\boxed{
\eta\asymp\delta^{p-r}
}
\]

when both leading coefficients are nonzero.

This is a scaling statement; coefficient ratios matter substantially in any
finite fixture.

## Shared-route finite-memory witness

For the exact non-delayed shared-route witness,

\[
L_{\rm lead}=-a\,xy-b\,xz,
\qquad
Q=x^2-y^2-z^2.
\]

The leading flow conserves `Q`. Its exact finite-step defect is already degree
two:

\[
\boxed{r=2.}
\]

The full exact horizon-12 loss first breaks the same balance law at degree

\[
\boxed{p=3,}
\]

while the leading construction degree is `d=2`. Therefore on the local
construction scale,

\[
\boxed{
\text{continuous higher-order drift}=O(\delta),
}
\]

but

\[
\boxed{
\text{finite-step drift}=O(\eta).
}
\]

At fixed learning rate, decreasing initialization eventually makes the optimizer
discretization error larger than the higher-order loss error. To retain the
continuous leading-balance geometry asymptotically, the learning rate must shrink
with initialization in this witness.

`experiments/discrete_invariant_drift.py` compares vanilla GD with high-accuracy
RK4 integration of the same **full exact finite-horizon polynomial** over one
unit of physical time, along the frozen ray

\[
(x,y,z)=\delta(1,0.7,0.5).
\]

The audit gives approximately

```text
continuous normalized-drift slope in delta : 0.9962
GD-minus-flow normalized slope in eta       : 0.9995
```

and a fitted coefficient-level crossover near

\[
\boxed{\eta\approx 19.8\,\delta.}
\]

The numerical factor is fixture- and ray-specific. The structural result is the
`O(delta)` versus `O(eta)` separation.

## Interpretation

There are now two different ways a locally conserved computation geometry can
fail:

\[
\boxed{
\text{higher-order loss support}
\quad\text{vs}\quad
\text{optimizer discretization}.
}
\]

They should not be conflated. Taking `delta -> 0` suppresses the first mechanism
on this witness but does nothing to suppress the second at fixed `eta`.
Conversely, taking `eta -> 0` recovers continuous flow but does not remove the
full loss polynomial's higher-order symmetry breaking.

This suggests a two-scale local optimization state:

\[
\boxed{
(\delta,\eta)
}
\]

in addition to the hard algorithm, continuous boundary geometry, and polynomial
support already identified elsewhere in the project.

## Prior-art boundary

Finite learning rates breaking conservation laws of continuous gradient flow are
not a novelty claim. In particular, Kunin et al. (2020), *Neural Mechanics:
Symmetry and Broken Conservation Laws in Deep Learning Dynamics*, explicitly
studies symmetry-derived conservation laws and finite-learning-rate corrections.
The broader symmetry/conservation connection is also established in Zhao et al.
(2022), *Symmetries, flat minima, and the conserved quantities of gradient
flow*.

The finite-memory contribution here is the exact sparse defect oracle and the
comparison of its degree `r` with the independently defined loss-breaking degree
`p` inside a concrete computation-construction polynomial.

## Claim boundary

### Exact / algebraic

- the one-step identity
  `Delta Q = eta*dQ/dt + eta**2*D_w` for vanilla gradient descent;
- `D_w=sum_i w_i*(partial_i L)**2`;
- the exact sparse defect polynomial and its first nonzero degree `r`;
- `Q_plus=(1-eta**2*C**2)Q` for the isolated `-Cxy` example;
- `r=2` for the frozen shared-route leading polynomial.

### Local asymptotic / deterministic finite audit

- normalized accumulated discretization drift is
  `O(eta*delta**(r-d))` on bounded local rescaled-time windows;
- on the frozen full shared-route polynomial, GD-minus-flow drift is linear in
  `eta` to numerical precision over the tested range;
- the fitted crossover is about `19.8*delta` on the frozen ray.

### Not established

- the same scaling for momentum, Adam, stochastic gradients, projection, or
  adaptive step schedules;
- a global guarantee up to fixed nonlocal thresholds;
- that preserving the balance law improves hard algorithm selection;
- optimizer- or parameterization-invariant meaning for the defect degree `r`.
