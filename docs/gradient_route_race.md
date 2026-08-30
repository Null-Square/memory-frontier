# Gradient-vector-field route races

The leading multivariate loss polynomial identifies candidate memory
constructions, but the previous branching experiment showed that weighted loss
order alone does not determine which construction gradient training builds
first. The next object is the gradient vector field induced by that polynomial.

This note separates an exact result for isolated construction monomials from
finite branching experiments in which the full polynomial perturbs that leading
dynamics.

## Exact derivative polynomial

For

\[
L(\boldsymbol\varepsilon)
=\sum_\alpha c_\alpha\boldsymbol\varepsilon^\alpha,
\]

every gradient coordinate is again an exact finite polynomial,

\[
\frac{\partial L}{\partial\varepsilon_j}
=\sum_{\alpha:\alpha_j>0}
 c_\alpha\alpha_j
 \boldsymbol\varepsilon^{\alpha-e_j}.
\]

`multivariate_polynomial_gradient_coefficients` computes these sparse
coefficient dictionaries exactly from the finite-horizon loss polynomial.
`multivariate_polynomial_hessian_coefficients` does the same for the Hessian.
No autograd or finite differencing is required.

This makes the local hierarchy explicit:

\[
\text{loss support}
\rightarrow
\text{gradient support}
\rightarrow
\text{gradient-flow dynamics}
\rightarrow
\text{route/boundary event}.
\]

## Exact square-free route dynamics

Consider one isolated useful construction with `R` independent missing links,

\[
L-L_0=-C\prod_{i=1}^R \varepsilon_i,
\qquad C>0.
\]

Under gradient flow,

\[
\dot\varepsilon_i
=C\prod_{j\ne i}\varepsilon_j.
\]

Therefore

\[
\frac{d}{ds}\varepsilon_i^2
=2C\prod_{j=1}^R\varepsilon_j
\]

for every coordinate. Hence

\[
\boxed{
\varepsilon_i^2-\varepsilon_j^2
=\text{constant}
}
\]

for every pair `i,j`.

Write

\[
\varepsilon_i(s)^2=a_i^2+r(s).
\]

Then the `R`-dimensional leading-route flow reduces exactly to one scalar
ordinary differential equation,

\[
\boxed{
\dot r
=2C\prod_i\sqrt{a_i^2+r}.
}
\]

If route completion means every link reaches a threshold `theta`, then

\[
r_* = \max(0,\theta^2-\min_i a_i^2)
\]

and the exact leading-route completion time is

\[
\boxed{
\tau
=\int_0^{r_*}
\frac{dr}{2C\prod_i\sqrt{a_i^2+r}}.
}
\]

This formula also exposes a bootstrap singularity: if two or more required
coordinates start exactly at zero, every first derivative vanishes and the
route remains stuck. With exactly one zero coordinate and all other links
positive, the integral is finite.

## Closed form for two-link routes

For

\[
L-L_0=-Cxy,
\]

let

\[
m=\min(x_0,y_0),
\qquad
M=\max(x_0,y_0),
\qquad
D=M^2-m^2.
\]

If `D>0`, the exact time until both links reach `theta` is

\[
\boxed{
\tau
=\frac{1}{C}
\left[
\operatorname{asinh}\frac{\theta}{\sqrt D}
-\operatorname{asinh}\frac{m}{\sqrt D}
\right].
}
\]

For the symmetric case `x_0=y_0=m>0`,

\[
\boxed{
\tau=\frac1C\log\frac{\theta}{m}.
}
\]

`two_link_leading_completion_time` implements this closed form. Thus two
disjoint degree-two construction monomials have a genuine leading-gradient
route race: compute each exact `tau` and choose the smaller one.

## It fixes the weighted-loss-order counterexample

In the previous parallel-route fixture,

```text
route A: x y
route B: u v
```

with equal leading coefficient `-C`, initialize

\[
x=y=t,
\qquad
u=t^{0.1},
\qquad
v=t^2.
\]

Weighted loss order ranks route A because

\[
2<2.1.
\]

The exact gradient completion times rank route B instead. At `t=10^-12` and
completion threshold `0.02`, route B is predicted to complete far earlier than
route A, matching full exact-polynomial projected training.

So the gradient vector field repairs the first falsified conjecture.

## But leading-gradient dynamics are still not sufficient

A second adversarial case makes the two leading-route times nearly tie. Set

\[
t=10^{-8}
\]

and initialize the four route links with exponent weights

\[
(0.2,1.0,0.6,0.2).
\]

The exact leading two-link predictor gives approximately

\[
\tau_A=7.2353930,
\qquad
\tau_B=7.2291420,
\]

so it predicts route B.

However, deterministic projected gradient descent on the **full exact
finite-horizon polynomial** with learning rate `0.02` builds route A first after
369 steps.

This is not a failure of the two-link theorem: that theorem is exact for the
isolated leading monomial. The branching controller contains higher-order terms
and route interactions that perturb the vector field during the race.

Therefore

\[
\boxed{
\text{exact leading-route dynamics}
\not\Rightarrow
\text{exact full-polynomial route winner}.
}
\]

## Curvature exposes the reversal locally

For full gradient flow

\[
\dot\varepsilon=-\nabla L,
\]

the local acceleration is

\[
\boxed{
\ddot\varepsilon
=H_L\nabla L.
}
\]

Using the exact gradient and Hessian of the full polynomial at the same
counterexample point, approximate each coordinate locally by

\[
\varepsilon_j(s)
\approx
\varepsilon_j(0)+v_js+\frac12a_js^2.
\]

Solving the quadratic threshold races gives approximately

\[
\hat\tau_A=8.0244,
\qquad
\hat\tau_B=8.0884.
\]

The curvature correction therefore flips the prediction from B to A **before
integrating the trajectory**, matching the observed first completed route.

`quadratic_two_link_route_times` implements this local second-order predictor.
It should be treated as a predictor, not as an exact theorem for the full
polynomial.

## Frozen 64-case audit

A deterministic grid uses initialization scale `t=10^-8`, threshold `0.02`, and
per-link exponent weights from

```text
{0.2, 0.6, 1.0}.
```

Cases in which a route is already complete at initialization are removed,
leaving 64 route races.

Using a relative tie tolerance of `1e-8`:

- the exact leading two-link predictor is decisive in 50 cases and matches the
  full-polynomial winner in 48: **48/50 = 96%**;
- its two errors are the symmetric curvature-reversal pair;
- the local Hessian-corrected predictor is decisive in 24 cases and matches all
  24: **24/24**;
- it abstains or ties on the remaining cases rather than being promoted to a
  general theorem.

This is a deterministic finite experiment, not a population claim.

## Numerical precision boundary

The finite-horizon coefficient oracle is exact up to floating-point arithmetic.
At initialization scales so tiny that genuine route gradients fall below
roughly the numerical coefficient noise floor, nominally zero lower-order
coefficients can dominate the evaluated vector field. Such cases must not be
interpreted as scientific counterexamples to the perturbative theory.

For route-race audits, either coefficients below a justified tolerance must be
pruned or the tested scale must stay above the oracle's floating-point noise
floor.

## Refined research statement

The strongest supported hierarchy is now:

1. the exact loss polynomial identifies candidate computational constructions;
2. differentiating it identifies the exact local gradient polynomial;
3. isolated square-free construction monomials admit exact gradient-flow
   completion dynamics;
4. branching higher-order terms can reverse a near-tied leading-route race;
5. exact local curvature can already reveal some such reversals;
6. general algorithm formation still depends on the evolving continuous vector
   field and eventual behavioral/boundary events.

The next falsification target is to test this hierarchy on shared-parameter and
higher-degree routes, then with jointly learned decoders and a non-delayed
stochastic language.
