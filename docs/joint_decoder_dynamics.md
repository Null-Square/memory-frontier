# Joint decoder learning preserves the construction-time barrier

The fixed-decoder construction-order results survive when the useful decoder is
learned from exact symmetry. The local joint expansion already shows an order
shift

\[
3\rightarrow2
\]

between the unscaffolded and dormant-prewired suffix-`01` controllers. On this
frozen non-delayed witness, the **full smooth finite-horizon loss factorizes
exactly**, allowing the decoder and transition probabilities to be co-trained by
an exact joint gradient vector field.

This closes a gap between local joint derivative order and actual joint smooth
training dynamics.

## Exact factorization

Use the order-2 Markov source and suffix-`01` memory from
`joint_decoder_order.md`. Let the final decoder logits be

\[
(-a,+a)
\]

and let `epsilon` denote the trainable transition probabilities.

Only memory state 2 has a nonuniform decoder. Therefore the finite-horizon loss
can be written exactly as

\[
\boxed{
L(a,\boldsymbol\varepsilon)
=\log2+f(a)R(\boldsymbol\varepsilon),
}
\]

where

\[
\boxed{
f(a)=\log\cosh a-\frac35a
}
\]

is the decoder excess cross-entropy after suffix `01`, and
`R(epsilon)` is the exact finite-horizon occupancy polynomial of memory state 2.

The strengthened regression in `tests/test_joint_decoder_order.py` verifies that
**every nonconstant transition-loss coefficient**, not only the leading one,
scales by the same factor `f(a)` across multiple decoder contrasts. The
factorization also follows directly because all other decoder rows are uniform.

The exact gradient flow is consequently

\[
\boxed{
\dot a=-f'(a)R(\boldsymbol\varepsilon),
}
\]

and

\[
\boxed{
\dot{\boldsymbol\varepsilon}
=-f(a)\nabla R(\boldsymbol\varepsilon),
}
\]

with

\[
f'(a)=\tanh a-\frac35.
\]

No STE, sampling, numerical differentiation, or autograd is involved in this
vector field.

## Leading joint dynamics

At decoder symmetry,

\[
f(0)=0,
\qquad
f'(0)=-\frac35.
\]

The leading occupancy coefficient for the useful suffix route is

\[
\frac5{42}.
\]

Hence the unscaffolded joint loss begins

\[
\boxed{
L-L_0
=-\frac1{14}a\varepsilon_1\varepsilon_2+\cdots,
}
\]

whereas dormant prewiring gives

\[
\boxed{
L-L_0
=-\frac1{14}a\varepsilon_1+\cdots.
}
\]

If decoder contrast and all still-missing transition links start at the same
small positive scale `delta`, the leading route is therefore square-free degree
3 without the scaffold and degree 2 with it.

The construction-time theorem then predicts

\[
\boxed{
\tau_{\rm unscaffolded}
=14\left(\frac1\delta-\frac1\theta\right)
}
\]

and

\[
\boxed{
\tau_{\rm prewired}
=14\log\frac\theta\delta.
}
\]

Thus jointly learning the decoder changes the fixed-decoder `2 -> 1` order
spectrum into `3 -> 2`, but the trajectory consequence becomes particularly
clear:

\[
\boxed{
\text{power-law bootstrap time}
\rightarrow
\text{logarithmic bootstrap time}.
}
\]

## Full exact-vector-field audit

`experiments/joint_decoder_dynamics.py` constructs the exact occupancy
polynomial `R`, differentiates it analytically, and integrates the full joint
vector field above with fixed-step RK4.

Use a common completion threshold

\[
\theta=0.1
\]

for decoder contrast and all missing transition probabilities. The deterministic
frozen audit is:

### Dormant-prewired controller — joint degree 2

| initial scale | leading time | full joint time | full / leading |
|---:|---:|---:|---:|
| 0.005 | 41.9403 | 44.8397 | 1.0691 |
| 0.010 | 32.2362 | 34.9913 | 1.0855 |
| 0.020 | 22.5321 | 25.0047 | 1.1097 |
| 0.030 | 16.8556 | 19.0513 | 1.1303 |

### Unscaffolded controller — joint degree 3

| initial scale | leading time | full joint time | full / leading |
|---:|---:|---:|---:|
| 0.005 | 2660.0000 | 2732.1747 | 1.0271 |
| 0.010 | 1260.0000 | 1316.5320 | 1.0449 |
| 0.020 | 560.0000 | 600.5440 | 1.0724 |
| 0.030 | 326.6667 | 357.9396 | 1.0957 |

The leading monomial is increasingly accurate as the initialization shrinks,
and the full smooth joint dynamics retain the large scaffold advantage.

This audit is deterministic numerical integration of an exact vector field. The
reported crossing times are not themselves closed-form theorems about the full
loss.

## What changed relative to the fixed-decoder story

Decoder symmetry does not create an alternate low-order route that bypasses
transition construction. Instead it contributes another required factor.

The hierarchy on this witness is therefore:

```text
fixed useful decoder:
    unscaffolded order 2
    prewired      order 1

learned symmetric decoder:
    unscaffolded order 3
    prewired      order 2
```

Dormant topology still removes exactly one missing computational factor in both
cases.

This is important for the broader framing: construction order should count every
independently trainable ingredient required before a dormant computation can
both be **reached** and **change prediction**.

## Claim boundary

### Exact / algebraic and regression-tested

- the full finite-horizon loss factorizes as `log(2) + f(a) R(epsilon)` on this
  witness;
- `R` is an exact finite transition polynomial;
- `f(a)=log(cosh(a))-(3/5)a`;
- the resulting joint gradient vector field is exact;
- the leading joint degrees are 3 without dormant prewiring and 2 with it.

### Deterministic finite experiment

- RK4 integration of the full exact joint vector field follows the predicted
  power-vs-log construction-time separation over the frozen initialization
  scales above.

### Not established

- that arbitrary learned readouts always add exactly one construction factor;
- that decoder parameterizations with multiple contrasts have the same joint
  order;
- that the leading joint monomial predicts competing computation races;
- invariance under optimizer, metric, or nonlinear parameterization changes.

The defensible conclusion is narrower: **joint decoder learning does not remove
the dormant construction-order phenomenon in this exact non-delayed witness;
it adds one common symmetry-breaking factor, and the corresponding full smooth
training dynamics preserve the scaffolded bootstrap-time advantage.**
