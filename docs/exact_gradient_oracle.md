# Exact straight-through gradient oracle

The finite-memory project now has an exact gradient oracle in addition to the
exact hard-controller loss oracle. The computation is NumPy-only; PyTorch is
used only in regression tests as an independent implementation.

## Setup

Let `F(m,x)` be the hard deterministic controller selected by the transition
logits, let `r_m(x)` be the current decoder distribution, and let

\[
d_t(s,m)=P(S_t=s,M_t=m)
\]

be the exact forward occupancy from the reset memory state. Define the immediate
expected log loss

\[
c(s,m)=-\sum_x P(x\mid s)\log r_m(x).
\]

For a horizon of `T` prediction positions, define the hard-controller future
value recursively by

\[
V_T(s,m)=0,
\]

and

\[
V_t(s,m)=c(s,m)+\sum_x P(x\mid s)
V_{t+1}(\delta(s,x),F(m,x)).
\]

## Counterfactual transition derivative

Temporarily treat the transition tensor `H[m,x,n]` as an independent continuous
weight, while evaluating the forward pass at the hard one-hot controller. Then

\[
\boxed{
G_{mxn}
=
\frac{\partial L}{\partial H_{mxn}}
=
\frac1T\sum_{t=0}^{T-2}\sum_s
 d_t(s,m)P(x\mid s)
 V_{t+1}(\delta(s,x),n).
}
\]

This has a direct interpretation: the derivative asks for the future loss if
mass currently at `(s,m)` emits `x` and is counterfactually routed to target
memory `n`. Behaviorally unreachable memory states therefore matter whenever
their counterfactual future values differ.

## Exact STE logit gradient

Let

\[
\sigma_{mx}=\operatorname{softmax}(z_{mx}/\tau)
\]

be the backward softmax row. The hard-forward straight-through gradient is the
full softmax-Jacobian vector product

\[
\boxed{
\frac{\partial L}{\partial z_{mxj}}
=
\frac{\sigma_{mxj}}{\tau}
\left(
G_{mxj}-\sum_n\sigma_{mxn}G_{mxn}
\right).
}
\]

For `K>2` this is genuinely a `K`-way object. Reducing it to a pairwise
current-vs-alternative scale generally gives the wrong answer because every
counterfactual target contributes to the softmax baseline. This was the source
of the earlier scratch `K=3` discrepancy.

## Exact decoder-logit gradient

Let

\[
A_{mx}=\frac1T\sum_{t,s}d_t(s,m)P(x\mid s)
\]

be the exact average joint mass of memory state `m` and next token `x`. If the
decoder is parameterized by logits with probabilities `r_m`, then

\[
\boxed{
\frac{\partial L}{\partial a_{mx}}
=
\left(\sum_y A_{my}\right)r_m(x)-A_{mx}.
}
\]

Thus both transition and decoder gradients are available without automatic
differentiation or sequence sampling.

## Verification

The regression suite compares the NumPy oracle against the existing PyTorch
exact-distribution straight-through implementation for `K=3` and `K=4`, binary
and ternary alphabets, first-order and delayed sources, arbitrary readout logits,
and non-unit backward temperatures. A separate finite-difference test validates
the pre-softmax transition-tensor derivative independently.

A 100-case scratch stress test over random source/controller/readout/horizon
configurations produced maximum absolute discrepancies of approximately

- `8.9e-16` in loss,
- `3.6e-16` in transition-logit gradients,
- `3.3e-16` in decoder-logit gradients.

## Consequence

The project can now separate three exact objects:

1. **capacity:** the best hard finite-state algorithm that exists;
2. **behavior:** the hard algorithm currently executed from reset;
3. **accessibility:** the exact surrogate gradient induced by all counterfactual
   memory targets, including targets that are behaviorally unreachable.

This makes the gradient field itself a ground-truth object rather than something
we estimate from training trajectories.
