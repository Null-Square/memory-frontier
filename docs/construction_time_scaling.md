# Construction order predicts leading escape-time scaling

The multivariate construction-order results identify the first useful derivative
order, and the route-race work shows how a leading computational monomial induces
a gradient vector field. For symmetric routes, these two views combine into an
exact trajectory-level statement: **construction order determines the
small-initialization scaling of leading gradient-flow completion time**.

This is a stronger bridge from local derivative geometry to algorithm formation
than gradient-norm attenuation alone.

## Symmetric square-free route

Consider one beneficial degree-`d` construction monomial

\[
L-L_0=-C\prod_{j=1}^d\varepsilon_j,
\qquad C>0,
\]

with independently trainable missing links initialized symmetrically,

\[
\varepsilon_1(0)=\cdots=\varepsilon_d(0)=\delta>0.
\]

Gradient flow preserves this symmetry. Writing the common coordinate as `x`,

\[
\boxed{
\dot x=Cx^{d-1}.
}
\]

If a computation is considered formed when every missing link reaches a fixed
threshold `theta > delta`, the exact leading-flow completion time is

\[
\boxed{
\tau_1=\frac{\theta-\delta}{C},
}
\]

for `d=1`,

\[
\boxed{
\tau_2=\frac1C\log\frac{\theta}{\delta},
}
\]

for `d=2`, and

\[
\boxed{
\tau_d=
\frac{\delta^{2-d}-\theta^{2-d}}
{C(d-2)},
\qquad d\ge3.
}
\]

`symmetric_square_free_completion_time` implements these formulas exactly.

## Three qualitative accessibility regimes

As initialization scale tends to zero with threshold fixed:

\[
\boxed{
\tau_1=O(1),
}
\]

\[
\boxed{
\tau_2=O(\log(1/\delta)),
}
\]

and

\[
\boxed{
\tau_d=\Theta(\delta^{-(d-2)}),
\qquad d\ge3.
}
\]

So derivative order creates a qualitative hierarchy in construction time:

- first-order accessibility has a finite escape-time limit;
- second-order accessibility has only logarithmic slowdown;
- third and higher orders have polynomially diverging construction times;
- each additional missing factor beyond order two adds one inverse power of the
  initialization scale.

This gives a direct dynamical interpretation to the order spectrum inside one
functional equivalence class.

For the delay-5 family, behaviorally identical collapsed predictors with orders
`1,2,3,4,5` therefore have leading construction-time laws

```text
order 1: constant
order 2: log(1/delta)
order 3: delta^-1
order 4: delta^-2
order 5: delta^-3
```

before higher-order polynomial corrections are included.

## Scaffold speedup is not the same as instantaneous gradient speedup

The earlier gradient-attenuation result says that prewiring `d-1` factors can
improve the leading entrance-gradient scale by approximately

\[
\delta^{1-d}.
\]

Integrated construction time has a different asymptotic law. Comparing an
order-`d` unscaffolded route against a first-order dormant-prewired route gives

\[
\frac{\tau_d}{\tau_1}
=\Theta(\delta^{-(d-2)})
\]

for `d >= 3`, while `d=2` gives only a logarithmic ratio. Thus dormant topology
can change the **asymptotic class** of bootstrap time:

- polynomial to logarithmic when order drops to two;
- logarithmic to finite when order drops from two to one.

This is a trajectory-level sense in which dormant scaffolding acts as an
optimization preconditioner.

## Exact zero-initialization barrier

The same formulas sharpen the symmetry-trap picture. If `delta=0` exactly,
then

- a degree-1 route has nonzero velocity and can start moving;
- every degree-`d >= 2` symmetric route has zero gradient and remains stuck under
  the isolated leading flow.

Thus the distinction between first and higher construction order is not only a
small-gradient issue: exact symmetry creates a true bootstrap barrier whenever
multiple missing factors are simultaneously required.

## Parameter tying changes time even on the same diagonal loss curve

Now tie all `d` coordinates to one scalar parameter `epsilon`. Along the diagonal
the scalar loss curve is still

\[
L-L_0=-C\varepsilon^d.
\]

But Euclidean gradient flow in the tied parameter is

\[
\dot\varepsilon=Cd\varepsilon^{d-1},
\]

because the scalar derivative collects all `d` factor contributions. Therefore

\[
\boxed{
\tau_d^{\rm tied}
=\frac1d\tau_d^{\rm independent,symmetric}.
}
\]

`tied_repeated_parameter_completion_time` checks this exact factor.

This is deliberately a parameterization-sensitive statement. The scalar loss
curve on the diagonal is the same, but the parameter-space metric and chain rule
change gradient dynamics. It reinforces the earlier warning that scalar
perturbative order or tied paths cannot replace the independent multivariate
representation.

## Full exact-polynomial audit

The closed-form times above are exact for the isolated leading monomial. A
separate deterministic experiment compares them against gradient-flow
integration of the **full exact finite-horizon loss polynomial** for the delay-5
functional-equivalence spectrum.

Use:

```text
source: delayed repeat, delay 5, switch probability 0.1
horizon: 13
initial missing-link probability: 0.02
completion threshold: 0.1
final decoder half-gap: 0.4
```

`experiments/construction_time_scaling.py` integrates the coefficient-cancelled
full polynomial with fixed-step RK4. The frozen audit is approximately:

| order | leading time | full-polynomial time | full / leading |
|---:|---:|---:|---:|
| 1 | 1.0742 | 1.3196 | 1.2285 |
| 2 | 21.6102 | 22.5935 | 1.0455 |
| 3 | 537.0868 | 552.1253 | 1.0280 |
| 4 | 16112.6055 | 16434.8576 | 1.0200 |
| 5 | 554989.7446 | 564147.0754 | 1.0165 |

The leading-flow law is therefore already close for orders 2 through 5 at this
finite initialization scale, while the first-order case receives a visibly
larger higher-order correction.

This table is a deterministic numerical experiment, not an exact theorem about
the full polynomial. The exact result is the isolated-monomial completion-time
law.

## Relation to gradient attenuation

The earlier attenuation result measured

\[
|\nabla L|\propto\delta^{d-1}
\]

for a degree-`d` construction coordinate at symmetric small scale. The
completion-time theorem integrates that local vector field and shows why the
trajectory consequence changes character at degree two:

- `d=1`: constant velocity;
- `d=2`: velocity proportional to position, giving exponential growth and a
  logarithmic escape time;
- `d>=3`: velocity vanishes superlinearly near zero, giving polynomially
  divergent escape times.

Thus the useful hierarchy is not merely "higher order means smaller gradient."
It predicts the asymptotic time required to bootstrap a computation under the
leading gradient dynamics.

## Prior-art boundary

The monomial ODE and its small-initialization escape exponents are not claimed as
new mathematical phenomena. Small-initialization deep-linear work already
studies saddle-to-saddle dynamics and multiplicative layer interactions (for
example Jacot et al., *Saddle-to-Saddle Dynamics in Deep Linear Networks*,
arXiv:2106.15933). Recent deep nonlinear theory derives an analogous critical
bottleneck escape-time law

\[
\tau=\Theta(\epsilon^{-(r-2)})
\]

for `r` small bottleneck layers (Rawal and DeWeese, *A Theory of Saddle Escape in
Deep Nonlinear Networks*, arXiv:2605.01288).

The finite-memory contribution is narrower: the exact finite-horizon prediction
polynomial identifies a concrete **computational memory construction** with each
multiplicative factor, and behaviorally dormant topology can move one fixed
predictor through the entire order/escape-time hierarchy without changing its
current forward function. The delay family also provides exact capacity and loss
oracles, so the optimization-time law can be compared against the globally
useful finite-memory computation rather than only against network depth.

## Claim boundary

### Exact / algebraic

- symmetric square-free degree-`d` leading flow reduces to
  `dx/ds = C*x**(d-1)`;
- the completion-time formulas above follow exactly;
- tied reuse of one scalar parameter changes the same diagonal gradient-flow
  time by the exact factor `1/d`;
- zero symmetric initialization is absorbing for every isolated route of degree
  at least two.

### Deterministic finite experiment

- the delay-5 full finite-horizon polynomial closely follows the leading-time
  oracle in the frozen RK4 audit above.

### Not established

- that leading completion time accurately predicts arbitrary branching or
  competing-route training;
- that the same time laws hold after strong decoder/transition coupling away
  from the local regime;
- invariance under reparameterization or optimizer changes;
- an exact full-polynomial escape-time theorem beyond isolated monomials.

The result should therefore be used as a local dynamical law and as a candidate
predictor of algorithm-formation times, not as a replacement for the full hybrid
training dynamics.
