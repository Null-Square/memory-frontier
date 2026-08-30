# Shared quadratic routes have one growing coefficient-aligned mode

Independent-route completion times stop being the right leading object once
several computations share a trainable entrance. For a broad quadratic shared
family, the coupled leading gradient flow is still exactly solvable.

## Shared-entrance polynomial

Consider

\[
L(x,y)=-x\sum_{j=1}^m a_jy_j,
\qquad a_j\ge0,
\]

where `x` is one shared entrance parameter and the `y_j` are route-specific exit
parameters. The positive `a_j` are beneficial route strengths extracted from the
leading loss polynomial.

Let

\[
c=\lVert a\rVert,
\qquad
u=\frac{a}{c},
\qquad
q=\nu\cdot y,
\qquad
y_\perp=y-q\nu.
\]

The gradient-flow equations are

\[
\dot x=a\cdot y=cq,
\]

\[
\dot y=ax.
\]

Therefore

\[
\boxed{
\dot x=cq,
\qquad
\dot q=cx,
\qquad
\dot y_\perp=0.
}
\]

So all route coupling reduces to a two-dimensional growing/decaying mode plus a
frozen orthogonal exit mixture.

## Exact solution

For initial state `(x0,q0,y_perp)`,

\[
\boxed{
x(t)=x_0\cosh(ct)+q_0\sinh(ct),}
\]

\[
\boxed{
q(t)=q_0\cosh(ct)+x_0\sinh(ct),}
\]

and

\[
\boxed{y_\perp(t)=y_\perp(0).}
\]

`shared_entrance_quadratic_flow` implements this solution exactly.

The exponent-support balance law

\[
Q=x^2-\lVert y\rVert^2
\]

is immediate because

\[
x^2-q^2
\]

is invariant in the two-dimensional hyperbolic subsystem and
`||y_perp||` is constant.

## Route mixtures align with predictive coefficients

The only growing exit direction is `a`. Any component of `y` orthogonal to `a`
is frozen while the aligned component grows like `exp(ct)`. Thus, whenever the
growing mode is present,

\[
\boxed{
\frac{y(t)}{\lVert y(t)\rVert}
\longrightarrow
\frac{a}{\lVert a\rVert}
}
\]

as time grows within the validity regime of the quadratic leading flow.

This is a different prediction from treating the exits as independent routes:
**shared-route competition selects a coefficient-aligned mixture rather than
letting every exit evolve independently.**

## Zero-exit initialization gives exact threshold races

If all route-specific exits start at zero but the shared entrance is positive,

\[
y(0)=0,
\qquad
x(0)=x_0>0,
\]

then

\[
\boxed{
y_j(t)=
\frac{a_j}{c}x_0\sinh(ct).}
\]

Hence the exit ratios equal the route-strength ratios at every positive time,

\[
\boxed{
y_i(t):y_j(t)=a_i:a_j.}
\]

For a common exit threshold `theta`, the exact leading crossing time is

\[
\boxed{
\tau_j=
\frac1c
\operatorname{asinh}
\left(\frac{\theta c}{a_jx_0}\right).
}
\]

Stronger routes therefore cross first. `shared_entrance_zero_exit_crossing_times`
implements this formula.

## Exact finite-memory witness

Use the non-delayed order-2 Markov shared-route witness already frozen in the
support-invariant tests. Its leading horizon-12 polynomial is

\[
L_{\rm lead}
=-a\,xy-b\,xz,
\]

where `y` recognizes suffix `00` and `z` recognizes suffix `01`. The exact
coefficients are approximately

\[
\boxed{a=0.1752686701,}
\]

\[
\boxed{b=0.02294580441,}
\]

so

\[
\boxed{a/b\approx7.63837549.}
\]

From shared entrance `x0=1e-3`, zero exits, and common threshold `0.01`, the
exact quadratic mode oracle predicts

```text
suffix 00 exit: 17.0095
suffix 01 exit: 28.4982
```

so suffix `00` is predicted to form first.

`experiments/shared_route_mode_dynamics.py` integrates the **full exact
finite-horizon polynomial** with RK4. The frozen audit gives approximately

```text
suffix 00 exit: 17.070
suffix 01 exit: 28.875
```

corresponding to relative timing errors of about `0.36%` and `1.32%`.

Thus the exact leading shared-mode dynamics correctly predict both the winner and
the construction times in this local finite-memory fixture.

## Relation to the discrete-step result

The previous finite-step analysis showed that vanilla gradient descent does not
preserve the whole continuous balanced cone. The mode decomposition explains the
special surviving ray: when

\[
y\parallel a
\]

and

\[
x=\lVert y\rVert,
\]

the state lies entirely in the coefficient-aligned growing eigenmode. Finite GD
scales that ray by one common factor, while orthogonal exit components are the
source of its negative balance defect.

So the continuous and discrete results fit together:

- continuous flow: orthogonal exit mixtures are frozen;
- growing behavior is coefficient-aligned;
- finite steps preserve the pure aligned ray exactly but alter balanced states
  containing frozen orthogonal components.

## Interpretation for learned finite memory

A shared entrance is not merely a parameter-counting detail. It changes the
leading construction dynamics from a collection of pairwise races into a
low-rank coupled mode system. The relevant leading object is therefore

\[
\boxed{
\text{route coefficient vector }a
}
\]

along with the initial projection onto its growing mode.

For the finite-memory witness, this coefficient vector already identifies which
suffix computation is locally favored and by how much.

## Prior-art boundary

The hyperbolic solution of this rank-one bilinear gradient system is elementary
and closely related to standard deep-linear/factorization dynamics. It is not a
novel mathematical mechanism. The project-specific use is to derive the route
coefficient vector from an exact finite-memory predictive-loss polynomial and
use it to predict coupled computation formation.

## Claim boundary

### Exact / algebraic

- the shared quadratic flow reduces to `(x,q)` plus frozen `y_perp`;
- the cosh/sinh solution above;
- exit components orthogonal to the coefficient vector are constant;
- zero-exit route ratios exactly equal coefficient ratios;
- the common-threshold crossing-time formula;
- the frozen finite-memory leading coefficients and their predicted ordering.

### Deterministic finite experiment

- the full exact horizon-12 polynomial crosses the two frozen thresholds within
  about `1.4%` of the leading-mode predictions.

### Not established

- coefficient alignment after the trajectory leaves the local polynomial regime;
- that the strongest leading route is globally optimal under hard controller
  search;
- analogous closed forms for higher-degree shared constructions;
- invariance under nonlinear parameterization or optimizers other than the stated
  Euclidean dynamics.
