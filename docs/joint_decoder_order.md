# Joint decoder and transition construction order

The earlier perturbative construction-order results held the decoder fixed while
transition links were assembled. That leaves an important question: does the
same high-order barrier survive when the useful decoder must itself be learned
from a symmetric initialization?

On the generic second-order Markov witness from `non_delayed_order_witness.md`,
the answer is yes. Decoder learning adds one additional local symmetry-breaking
factor, while dormant transition prewiring still removes one transition factor.

## Setup

Use the observable order-2 source

```text
suffix   P(next=1)
00       0.1
01       0.8
10       0.6
11       0.2
```

and the three-state suffix-`01` memory construction

```text
0 --0--> 1 --1--> 2.
```

Let the two missing transition probabilities be

\[
\varepsilon_1,\varepsilon_2.
\]

Memory states 0 and 1 keep the uniform decoder. Parameterize the final memory
state by one antisymmetric decoder-logit contrast `a`:

\[
\operatorname{logits}(Q_2)=(-a,+a).
\]

Thus

\[
Q_2(1)=\sigma(2a).
\]

At `a=0`, every decoder row is uniform and the current predictor is exactly the
collapsed predictor.

## Exact route coefficient as a function of decoder contrast

For horizon `T=12`, the coefficient multiplying the completed suffix-`01`
transition route is exactly

\[
\boxed{
C(a)
=\frac{5}{42}
\left[
\log\cosh a-\frac35a
\right].
}
\]

The factor `5/42` is the exact finite-horizon occupancy coefficient of the route.
Conditional on reaching the suffix-`01` decoder, the source has

\[
P(X_{t+1}=1\mid 01)=\frac45,
\]

and the excess cross-entropy of logits `(-a,+a)` relative to the uniform decoder
is

\[
\log\cosh a-\frac35a.
\]

`tests/test_joint_decoder_order.py` checks this closed form against the exact
multivariate transition-loss oracle for several values of `a`, both with and
without dormant downstream prewiring.

## Local joint expansion

Near decoder symmetry,

\[
\log\cosh a
=\frac{a^2}{2}+O(a^4),
\]

so

\[
\boxed{
C(a)
=-\frac1{14}a
+\frac5{84}a^2
+O(a^4).
}
\]

Therefore the unscaffolded controller has leading joint term

\[
\boxed{
L-L_0
=-\frac1{14}
 a\,\varepsilon_1\varepsilon_2
+\text{higher-order terms}.
}
\]

At the exactly symmetric point, every transition-only derivative vanishes. The
first useful derivative involving decoder learning is the three-way mixed
derivative in

\[
(a,\varepsilon_1,\varepsilon_2).
\]

The construction is therefore third order in the joint local parameterization.

## Dormant prewiring

Now prewire the behaviorally unreachable downstream transition

```text
1 --1--> 2
```

while keeping the entrance link absent. The current forward predictor remains
identical because memory state 1 is unreachable from reset.

Only the entrance transition remains to be constructed, so the leading joint
term becomes

\[
\boxed{
L-L_0
=-\frac1{14}
 a\,\varepsilon_1
+\text{higher-order terms}.
}
\]

Thus decoder symmetry raises both parameterizations by one common factor, but
does not remove the dormant-topology advantage:

\[
\boxed{
3\text{rd-order joint accessibility}
\rightarrow
2\text{nd-order joint accessibility}.
}
\]

This is the joint-decoder analogue of the fixed-decoder transition result

\[
2\rightarrow1.
\]

## Interpretation

The result separates two independent construction requirements:

1. a useful memory state must become prediction-distinct through decoder
   symmetry breaking;
2. the source history must be routed into that state through the required
   transition construction.

Locally these requirements multiply. A fully symmetric readout does not make
transition construction irrelevant; it contributes an additional missing
factor to the same computation polynomial.

Dormant scaffolding still acts as an optimization preconditioner because it
supplies transition factors in advance. It cannot supply the missing decoder
contrast unless decoder structure is also prewired.

## Claim boundary

### Exact / regression-tested

- at `a=0`, all transition-only nonconstant coefficients vanish;
- the route coefficient is exactly
  `(5/42)[log(cosh(a))-(3/5)a]` on the frozen witness;
- its derivative at decoder symmetry is exactly `-1/14`;
- without prewiring, the leading joint local term is proportional to
  `a*epsilon_1*epsilon_2`;
- with the downstream link dormant-prewired, it is proportional to
  `a*epsilon_1`;
- both parameterizations still execute the same collapsed predictor at the
  symmetric base point.

### Not established

- a general theorem for arbitrary jointly learned decoder parameterizations;
- invariance of joint construction order under nonlinear reparameterization of
  decoder logits;
- which factor is learned first under full joint gradient dynamics;
- whether decoder and transition dynamics can create lower-order effective
  routes away from the symmetric point.

The correct general object is therefore not only a transition perturbation
polynomial, but a joint local expansion over every independently trainable
component required to make the dormant computation both reachable and
prediction-distinct.
