# Gradient Accessibility Operator

The exact-gradient oracle makes it possible to define a local accessibility
object that is less arbitrary than seed success rates or one chosen decoder
perturbation.

## Definition

Fix a hard transition table `F` through transition logits `z`, and suppose every
memory state has the same decoder logits `a_0`. At this symmetric decoder point,
the hard-forward prediction loss is independent of the current memory label, so
the transition-logit STE gradient is zero.

Now perturb decoder logits by an infinitesimal matrix `D`. The first-order change
in the transition gradient is linear in `D`. We define

\[
\boxed{
\mathcal A_F
=
\frac{\partial (\nabla_z L)}{\partial a}
\bigg|_{a_m=a_0\ \forall m}
}
\]

and call it the **Gradient Accessibility Operator (GAO)**.

The operator is local to a fixed hard argmax cell and a specified softmax
backward temperature. It is a mixed derivative of the hard-forward / soft-
backward surrogate vector field, not a Hessian of a globally smooth hard loss.

## Exact dynamic-programming construction

For decoder-logit direction `D`, let `q=softmax(a_0)`. The derivative of the
immediate expected loss at source state `s`, memory state `m`, is

\[
\delta c(s,m)
=-\sum_x P(x\mid s)
\left(D_{mx}-\sum_yq_yD_{my}\right).
\]

Holding the hard table fixed, propagate the value perturbation backward:

\[
\delta V_T=0,
\]

\[
\delta V_t(s,m)
=
\delta c(s,m)
+
\sum_xP(x\mid s)
\delta V_{t+1}(\delta(s,x),F(m,x)).
\]

The exact first-order response of the pre-softmax transition derivative is

\[
\delta G_{mxn}
=
\frac1T
\sum_{t=0}^{T-2}\sum_s
 d_t(s,m)P(x\mid s)
 \delta V_{t+1}(\delta(s,x),n),
\]

where `d_t` is the unperturbed hard forward occupancy. Applying the full
`K`-way softmax Jacobian gives the corresponding perturbation of
`grad_z L`. No sampling or autograd is needed.

The implementation assembles the complete matrix by applying this exact linear
recursion to a basis of decoder-logit perturbations simultaneously.

## Gauge nullspaces

Decoder logits have an intrinsic row-wise gauge: adding a constant to every
token logit in one memory state's decoder changes no probability distribution.
Those directions are therefore exact null vectors of `A_F`.

Transition-softmax gradients likewise sum to zero inside every `(memory, token)`
row. Consequently raw matrix dimension is not meaningful by itself. The API
reports singular values and a tolerance-aware numerical rank.

We interpret:

- `rank(A_F)`: number of independent transition-gradient directions that can be
  born at first order from decoder symmetry breaking;
- singular values: local amplification strengths of those accessible directions;
- right singular vectors: decoder symmetry-breaking patterns that most strongly
  create transition gradients;
- left singular vectors: resulting transition-gradient patterns.

## Forward equivalence does not determine accessibility

For the binary delay-3 source, compare two `K=4` controllers whose reachable row
is fully collapsed to memory 0. From reset, both stay in memory 0 for every
possible token sequence and have identical uniform-decoder loss `log 2`.

The first controller also collapses every unreachable row. The second contains
the unreachable chain

```text
1 -> 2 -> 3 -> 0
```

for either input token.

At horizon 20, transition margin 0.7, backward temperature 0.8, and uniform
symmetric decoder base point, the exact operators are

\[
\boxed{\operatorname{rank}\mathcal A_{blind}=0}
\]

and

\[
\boxed{\operatorname{rank}\mathcal A_{scaffold}=1}.
\]

The scaffold's leading singular value is approximately

\[
0.107354361.
\]

Thus two models can have the same current function and loss while one has no
first-order route from decoder symmetry breaking into transition learning and
the other has a nonzero route.

## Exhaustive forward-equivalent topology scan

With the reachable row fixed collapsed, `K=4`, binary tokens, there are

\[
4^{3\times2}=4096
\]

possible choices for the three unreachable transition rows. They all implement
the same forward function from reset at the symmetric decoder point.

On the delay-3 source above, exhaustive exact evaluation gives:

| GAO rank | Number of unreachable topologies |
|---:|---:|
| 0 | 52 |
| 1 | 636 |
| 2 | 1416 |
| 3 | 1992 |

The median leading singular value is approximately `0.247485`, with maximum
approximately `0.423743`. The scan is reproducible with
`experiments/accessibility_rank_scan.py`; it is intentionally not part of CI.

This result is stronger than saying that initialization changes optimization.
The complete current input/output function is fixed while the linearized
learnability operator varies over a wide range solely because of behaviorally
unreachable hard-state topology.

## Relation to earlier results in this repository

The GAO unifies several earlier fixtures:

- the symmetry trap is a zero-accessibility point before decoder symmetry is
  broken;
- the predictive-spectrum theorem is a special factorization of `A_F` for an
  observable Markov source;
- higher-order blindness corresponds to a collapsed topology whose operator
  cannot transmit the relevant delayed mode;
- counterfactual self-loops change the temporal filter inside `A_F`;
- delay-matched dormant chains create a nonzero singular mode at exactly the
  required temporal depth.

The broader research distinction is therefore:

1. **capacity frontier:** which finite-memory predictors exist;
2. **behavioral state:** which hard predictor is currently executed;
3. **accessibility operator:** which infinitesimal representation changes can
   create useful transition-gradient directions from that parameterization.

Mixed derivatives, finite-state policy gradients, and spectral operators are of
course established mathematical tools. The narrower claim here is the exact
application to a hard-forward/soft-backward recurrent memory system where
behaviorally unreachable automaton structure changes the rank and spectrum of
local learnability while leaving the current model function unchanged.
