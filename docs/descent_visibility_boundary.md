# Visibility order versus beneficial descent order

The construction-order hierarchy identifies the first degree at which a latent
computation becomes **visible to the scalar objective**. A paper about
learnability should not silently identify this with the first degree that permits
a loss-decreasing move inside the valid transition cone.

This note makes the distinction exact.

## Homogeneous expansion

Let a smooth scalar objective near a construction origin have Taylor expansion

\[
L(x)-L(0)=P_d(x)+P_{d+1}(x)+\cdots,
\]

where each \(P_k\) is homogeneous of degree \(k\) and
\(P_d\not\equiv0\). The scalar visibility order is

\[
\boxed{d_{\rm vis}=d.}
\]

In the finite-memory paper this is \(d_{\rm loss}\).

Let \(K\) be the admissible cone of local perturbation rays. For example, when a
construction coordinate directly adds an absent transition probability, the
corresponding local coordinate is constrained to be nonnegative.

For a nonzero ray \(v\in K\), define its first visible degree

\[
d(v)=\min\{k:P_k(v)\ne0\}.
\]

If the leading coefficient on that ray is negative,

\[
P_{d(v)}(v)<0,
\]

then for all sufficiently small \(t>0\),

\[
L(tv)<L(0),
\]

with improvement of order \(t^{d(v)}\).

Define the beneficial ray order

\[
\boxed{
d_{\downarrow}
=\inf\{d(v):v\in K\setminus\{0\},\;P_{d(v)}(v)<0\}.
}
\]

If no such ray exists, set \(d_{\downarrow}=\infty\).

## Proposition: when visibility equals beneficial accessibility

Because no homogeneous term exists below \(d_{\rm vis}\),

\[
\boxed{d_{\downarrow}\ge d_{\rm vis}.}
\]

Moreover,

\[
\boxed{
d_{\downarrow}=d_{\rm vis}
\iff
\exists v\in K:\;P_{d_{\rm vis}}(v)<0.
}
\]

The proof is immediate from the one-dimensional expansion

\[
L(tv)-L(0)
=t^{d(v)}P_{d(v)}(v)+o(t^{d(v)}).
\]

Thus the construction-order theorem should be interpreted as an exact theorem
about **visibility**. A dynamical statement about building a useful computation
requires a sign/descent condition on the leading homogeneous form.

## Why a nonzero leading term is not enough

Consider the positive orthant \(K=\mathbb R_+^2\) and

\[
L(x,y)-L(0)=x-y^2.
\]

The scalar visibility order is one because of the linear term \(x\). But that
first-order term is harmful on the admissible cone: increasing \(x\) increases
loss. Along the admissible ray \((0,1)\), the first nonzero term is instead the
beneficial quadratic \(-y^2\). Hence

\[
\boxed{d_{\rm vis}=1,\qquad d_{\downarrow}=2.}
\]

This simple example is why the paper should avoid defining usefulness by
nonvanishing alone.

If \(P_d\) is strictly positive on every nonzero admissible normalized ray, then
no sufficiently small ray descent exists at degree \(d\). If \(P_d\ge0\) but has
a nontrivial zero set, any beneficial move must lie in that zero set and is
controlled by higher homogeneous terms.

## The paper's main constructions satisfy the descent condition

The isolated construction-time results use a beneficial square-free leading
monomial

\[
L-L_0=-C\prod_{i=1}^d x_i,
\qquad C>0,
\]

on the positive construction cone. For every interior positive ray,

\[
P_d(v)=-C\prod_i v_i<0.
\]

Therefore

\[
\boxed{d_{\downarrow}=d_{\rm loss}=d}
\]

for those fixtures. The finite-memory regressions used for the bootstrap claims
explicitly verify the relevant leading coefficient is beneficial.

The shared-route examples similarly label route coefficients as beneficial only
after checking their sign; near-tie reversals concern competition among already
beneficial routes.

## Recommended terminology

Use the following language in the manuscript:

- **construction order / loss order:** first nonzero local degree, a visibility
  statement;
- **beneficial construction of order \(d\):** a degree-\(d\) leading form that is
  negative along at least one admissible construction ray;
- **gradient accessibility of a useful computation:** construction visibility
  plus the relevant admissible descent condition;
- **bootstrap-time law:** only after the beneficial sign condition and the stated
  isolated-route/metric assumptions are imposed.

This distinction strengthens rather than weakens the main claim. Dormant topology
can change the degree at which a computation becomes visible while forward
behavior is held fixed; in the paper's main scaffold witnesses, the first visible
term is also beneficial, so the visibility separation translates directly into a
learnability separation.
