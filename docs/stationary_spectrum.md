# General stationary-law pressure theorem

The doubly-stochastic spectral result has a broader stationary form. This note states the exact extension used by the regression tests.

Let `P` be any finite row-stochastic observable Markov source with strictly positive stationary token law `pi`, so `pi^T P = pi^T`. The source state is the previous token. A `q`-state hard memory controller is fully collapsed to memory 0.

Initialize the collapsed decoder row to the exact stationary marginal `pi`, i.e. logits `log(pi)`. Let an unused memory state's decoder logits be

\[
\log \pi + d.
\]

All other decoder rows equal the collapsed row. Canonical transition logits give the selected target a common margin `a`, and the backward softmax temperature is `tau`.

Define

\[
s=\frac{e^{a/\tau}}{e^{a/\tau}+q-1},\qquad
 t=\frac{1}{e^{a/\tau}+q-1},
\]

and

\[
K=\frac{T-1}{T}\frac{t(s+1-t)}{\tau}.
\]

For token `x`, let

\[
p_x=g_{0,x,0}-g_{0,x,1}
\]

be the straight-through descent pressure toward the unused state. Then the exact raw pressure is

\[
\boxed{
p=K D_\pi\left(Pd-\psi_\pi(d)\mathbf 1\right),
}
\]

where

\[
D_\pi=\operatorname{diag}(\pi),\qquad
\psi_\pi(d)=\log\mathbb E_{Y\sim\pi}[e^{d_Y}].
\]

This identity is finite-horizon and exact for finite decoder-logit contrast.

Now rescale by stationary token frequency,

\[
r=D_\pi^{-1}p,
\]

and define the stationary weighted centering operator

\[
\Pi_\pi=I-\mathbf 1\pi^\top.
\]

Since `pi^T P = pi^T`, centering commutes with the Markov operator on deviations from the stationary mode. Therefore

\[
\boxed{
\Pi_\pi r = K P\Pi_\pi d.
}
\]

So the predictive operator appears directly in the surrogate-gradient field without requiring a uniform stationary distribution.

## Reversible spectral corollary

If the source is reversible with respect to `pi`, its Markov operator is self-adjoint in the `pi`-weighted inner product. For any non-stationary eigenmode

\[
Pd=\lambda d,\qquad \pi^\top d=0,
\]

we get

\[
\boxed{
\Pi_\pi D_\pi^{-1}p=K\lambda d.
}
\]

Thus predictive eigenvalue magnitude and sign still control the local memory-specialization direction after the correct stationary weighting.

## Verified fixtures

The regression suite checks two cases:

1. A nonuniform, nonreversible 3-state source with stationary law
   \[
   \pi=(7/24,13/24,1/6),
   \]
   where both the raw formula and weighted-centered operator identity match PyTorch autograd to numerical precision.
2. A nonuniform reversible 3-state chain with stationary law `(0.2,0.4,0.4)` and predictive eigenvalues `0.8` and `0.5`; both eigenmodes match the exact gradient scaling.

This removes the main symmetry objection to the earlier Walsh fixture. The uniform/doubly-stochastic theorem is now best viewed as the simplest corollary, not the full result.
