# Generic visibility of the first construction operator

The exact hierarchy

\[
d_{\mathrm{support}}
\le d_{\mathrm{operator}}
\le d_{\mathrm{loss}}
\]

allows strict inequalities. The deterministic neutral-decoder fixture realizes

\[
(1,1,2).
\]

That strict decoder inequality is nevertheless nongeneric in the precise
measure-theoretic sense below.

## Fixed construction geometry

Fix:

- the unifilar source;
- the affine controller transition family;
- the finite horizon;
- the partition of memory states into decoder-equality classes.

Let

\[
d=d_{\mathrm{operator}}<\infty.
\]

Stack the degree-`d` construction-operator coefficients into

\[
\mathcal G_d.
\]

By definition,

\[
\mathcal G_d\ne0.
\]

For decoder class distributions `q_C` in the interior of their probability
simplices, define

\[
z(q)=\operatorname{vec}(\log q_C).
\]

The complete vector of degree-`d` scalar loss coefficients is

\[
c_d(q)=-\mathcal G_d z(q).
\]

## Almost-sure equality of operator and loss order

Choose any nonzero row `g` of `G_d`. The associated scalar coefficient is

\[
f(q)=-g^Tz(q).
\]

This is a real-analytic function on the product of open probability simplices.
It is not identically zero.

To see this, choose a decoder class on which `g` has a nonzero block. Holding all
other decoder classes fixed, vary that class inside its simplex. A nonzero fixed
vector cannot have constant inner product with `log q` over the full simplex
interior. Equivalently, approaching a boundary coordinate for which the
corresponding component of `g` is nonzero makes the relevant `log q_i` diverge.

A nonzero real-analytic function has a zero set of Lebesgue measure zero in the
connected open domain. Therefore

\[
\Pr[f(q)=0]=0
\]

for any decoder distribution with a density on the class-simplex interior.
Since simultaneous cancellation of every degree-`d` coefficient is a subset of
this one coefficient's zero set,

\[
\boxed{
\Pr[d_{\mathrm{loss}}>d_{\mathrm{operator}}]=0.
}
\]

Hence

\[
\boxed{
d_{\mathrm{loss}}=d_{\mathrm{operator}}\quad\text{almost surely}}
\]

under continuously sampled decoder values, conditional on the fixed equality
partition and finite operator order.

This result explains why the stratified random census saturates the second
inequality in all tested cases while exact hand-tuned cancellation remains
possible.

## Generic support-to-operator equality

A similar but slightly more conditional statement applies to the first
inequality.

Fix a source/transition support pattern and a readout-equality partition. Within
an irreducible support cell, the finite-horizon occupancy coefficients and hence
the entries of the first candidate operator are analytic functions of the
positive source probabilities, supported base-transition probabilities, and
nonzero perturbation weights (the stationary distribution is analytic wherever
the induced source chain remains irreducible).

Let

\[
d=d_{\mathrm{support}}.
\]

If there exists at least one admissible numerical assignment on that fixed
support pattern for which the degree-`d` quotient operator is nonzero, then the
set of assignments for which **all** degree-`d` operator entries cancel is an
analytic zero set of measure zero. Under that nondegeneracy condition,

\[
\boxed{
d_{\mathrm{operator}}=d_{\mathrm{support}}\quad\text{almost surely}.}
\]

The qualification is necessary: a support pattern may encode an exact signed
symmetry that forces its first candidate operator to vanish identically. The
operator calculation, rather than graph distance alone, detects that case.

## Combined generic statement

Under both nondegeneracy conditions,

\[
\boxed{
d_{\mathrm{support}}
=d_{\mathrm{operator}}
=d_{\mathrm{loss}}
\quad\text{almost surely}.}
\]

The hierarchy is therefore useful in two complementary ways:

1. it gives hard lower bounds without genericity assumptions;
2. it identifies the exact nullspaces/analytic exceptional sets responsible for
   higher-than-structural derivative order.

This is why explicit cancellation witnesses remain scientifically important even
though random continuous families usually saturate all three orders.
