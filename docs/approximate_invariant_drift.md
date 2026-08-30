# Higher-order support gives a local lifetime for leading balance laws

The exponent-support theorem gives exact diagonal quadratic invariants of a
polynomial gradient flow. A finite-memory loss polynomial often has a simpler
**leading** support than its full higher-order expansion, so the useful next
question is quantitative:

> if a balance law is exact for the leading computation polynomial but broken by
> higher-order terms, how accurately is it conserved near the collapsed point?

The answer is controlled by a degree gap.

## Setup

Write the exact finite polynomial as

\[
L(x)=L_d(x)+L_{d+1}(x)+\cdots,
\]

where `d` is the first nonconstant total degree and each `L_r` is homogeneous of
total degree `r`.

Let

\[
Q_w(x)=\sum_i w_i x_i^2
\]

be an exact invariant of the leading flow, so every exponent vector in the
support of `L_d` is orthogonal to `w`.

Define the **first breaking degree**

\[
\boxed{
 p=\min\{\lvert\alpha\rvert:
 c_\alpha\ne0,\ \alpha\cdot w\ne0\}.
}
\]

If no such exponent exists, `Q_w` is an exact invariant of the complete
polynomial. Otherwise `p>d` whenever the leading balance law is genuine.

`quadratic_invariant_derivative_coefficients` constructs the exact sparse
polynomial

\[
\boxed{
\frac{dQ_w}{dt}
=
\sum_\alpha
[-2c_\alpha(\alpha\cdot w)]x^\alpha,
}
\]

and `quadratic_invariant_breaking_degree` returns `p` after coefficient and
orthogonality tolerances are applied.

## Instantaneous breaking order

Put `x=delta*y` with `y=O(1)`. Because all terms below degree `p` cancel from the
invariant derivative,

\[
\boxed{
\frac{dQ_w}{dt}
=
\delta^p B_p(y)+O(\delta^{p+1}),
}
\]

where `B_p` is the degree-`p` part of the exact derivative polynomial.

Thus `p` is an exact local order for symmetry breaking of the balance law.

## Construction-scale accumulated drift

The leading degree-`d` gradient field has size `delta^(d-1)`. With

\[
x(t)=\delta y(s),
\qquad
s=\delta^{d-2}t,
\]

its local dynamics become

\[
\frac{dy}{ds}=F_d(y)+O(\delta).
\]

Since

\[
Q_w(x)=\delta^2 Q_w(y),
\]

we obtain on every fixed rescaled-time interval for which `y(s)` remains bounded,

\[
\boxed{
\frac{d}{ds}Q_w(y)
=O(\delta^{p-d}).
}
\]

Therefore

\[
\boxed{
\Delta Q_w/\delta^2
=O(\delta^{p-d}).
}
\]

The quantity

\[
\boxed{g=p-d}
\]

is the **symmetry-breaking degree gap**. Each additional degree between the
leading computation and the first violating monomial buys one additional power
of small initialization in normalized balance-law drift.

This statement is local. It controls fixed `O(1)` windows in the natural
rescaled construction time. It does **not** imply that the balance law remains
accurate all the way to a fixed absolute threshold independent of `delta`, where
the rescaled coordinates can grow like `1/delta` and the local expansion is no
longer uniform.

## Exact finite-memory shared-route witness

Use the non-delayed order-2 Markov source and shared-route controller from
`test_shared_route_invariants.py`. Its exact horizon-12 loss polynomial begins
with two beneficial degree-two computations,

\[
xy,\qquad xz,
\]

so

\[
d=2
\]

and the leading balance law is

\[
Q=x^2-y^2-z^2.
\]

The complete exact polynomial has cubic terms

\[
x^2y,\qquad x^2z,
\]

whose exponent vectors have nonzero inner product with `(1,-1,-1)`. Hence

\[
\boxed{p=3,\qquad g=p-d=1.}
\]

This is regression-tested directly from
`multivariate_controller_loss_coefficients`; the breaking degree is not inserted
by hand.

`experiments/approximate_invariant_drift.py` integrates the **full exact
finite-horizon polynomial** for one unit of local gradient-flow time, starting
from

\[
(x,y,z)=\delta(1,0.7,0.5).
\]

For `delta` from `1e-2` to `1e-4`, the normalized drift

\[
\frac{|Q(t)-Q(0)|}{\delta^2}
\]

has a frozen log-log slope of approximately

\[
\boxed{0.9962,}
\]

matching the predicted degree gap `g=1`.

This numerical slope is an audit of the exact full polynomial, not an additional
algebraic theorem.

## Relation to the support-rank result

The two support statistics answer different questions:

- `rank(A)` determines how many exact diagonal quadratic balance laws survive a
  chosen polynomial support;
- `p-d` determines how softly a balance law of the leading support is broken by
  higher-order support.

So two models can have the same leading support rank and the same leading
construction order, yet differ in how long their local balance geometry remains
predictive because their first symmetry-breaking degrees differ.

This adds another layer to the optimization-state hierarchy:

\[
\boxed{
\text{leading support}
\to
\text{balance laws}
\to
\text{breaking degree gap}
\to
\text{local drift}
\to
\text{full route dynamics}.
}
\]

## Prior-art boundary

The general link between parameter symmetries and gradient-flow conservation laws
is established in prior work, including *Neural Mechanics: Symmetry and Broken
Conservation Laws in Deep Learning Dynamics* (Kunin et al., 2020) and
*Symmetries, flat minima, and the conserved quantities of gradient flow* (Zhao
et al., 2022). Finite learning rates are also known to break continuous-time
conservation laws.

The result here should therefore be framed specifically as an exact
finite-memory polynomial diagnostic: the exponent support identifies the first
higher-order term that breaks a leading computation's balance law, and the degree
gap predicts its local small-initialization drift order. No claim is made that
approximate symmetry or broken-conservation analysis in general is new.

## Claim boundary

### Exact / algebraic

- the derivative polynomial of `Q_w` has coefficient
  `-2*c_alpha*(alpha dot w)` for every exponent `alpha`;
- `p` is the first total degree with a nonzero coefficient in that derivative
  polynomial;
- if `w` annihilates the degree-`d` leading support and `p>d`, the normalized
  local drift over fixed rescaled time is `O(delta**(p-d))`;
- on the frozen finite-memory shared-route witness, `d=2` and `p=3` exactly.

### Deterministic finite experiment

- the full exact horizon-12 polynomial gives a normalized-drift slope about
  `0.9962` on the frozen scale range.

### Not established

- a uniform invariant-lifetime law up to fixed nonlocal parameter thresholds;
- that the degree gap alone predicts route winners or hard boundary crossings;
- robustness under finite-step SGD, momentum, adaptive optimizers, noise, or
  nonlinear reparameterization;
- completeness beyond diagonal quadratic candidate invariants.
