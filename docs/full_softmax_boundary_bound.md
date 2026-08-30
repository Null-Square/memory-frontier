# Full-softmax extension of the rare-edge boundary exponent

The exact completion-time formula in `softmax_boundary_bootstrap.md` uses one
scalar binary logit per missing edge. A controller row is more commonly
parameterized by a full `K`-way softmax with every logit trainable. The exact
binary prefactor changes under that Euclidean metric, but the rare-edge exponent
does not.

## Target-probability Jacobian

Let a full softmax row have probabilities

\[
(p,q_1,\ldots,q_{K-1}),
\qquad
\sum_{a=1}^{K-1}q_a=1-p,
\]

where `p` is the rare target edge. For logits `z`,

\[
\frac{\partial p}{\partial z_*}=p(1-p),
\qquad
\frac{\partial p}{\partial z_a}=-p q_a.
\]

Therefore

\[
\boxed{
\|\nabla_z p\|^2
=
p^2\left((1-p)^2+\sum_{a=1}^{K-1}q_a^2\right).
}
\]

The non-target probabilities have fixed sum `1-p`. Cauchy--Schwarz gives

\[
\sum_a q_a^2
\ge
\frac{(1-p)^2}{K-1},
\]

while concentration of their mass gives

\[
\sum_aq_a^2\le(1-p)^2.
\]

Hence

\[
\boxed{
\frac{K}{K-1}p^2(1-p)^2
\le
\|\nabla_zp\|^2
\le
2p^2(1-p)^2.
}
\]

The lower bound is attained when the non-target probabilities are uniform. The
upper bound is approached as their mass concentrates on one alternative.

## Consequence for a degree-d construction

Consider `d` independent controller rows with equal rare target probability `p`
and isolated leading objective

\[
L-L_0=c\prod_{i=1}^d p_i,
\qquad c<0.
\]

For any one row,

\[
-\frac{\partial L}{\partial p_i}
=(-c)p^{d-1}.
\]

Under Euclidean gradient flow in **all** logits of that row,

\[
\dot p_i
=-\frac{\partial L}{\partial p_i}
\|\nabla_{z_i}p_i\|^2.
\]

Combining with the Jacobian bounds gives

\[
\boxed{
(-c)\frac{K}{K-1}
 p^{d+1}(1-p)^2
\le
\dot p_i
\le
2(-c)p^{d+1}(1-p)^2.
}
\]

Thus for fixed finite `K`,

\[
\boxed{
\dot p=\Theta(p^{d+1})
\qquad p\to0.
}
\]

The non-target distribution may evolve with the logits, but these dimension-only
bounds hold pointwise throughout the trajectory. Integrating the differential
inequality between a rare initialization `delta` and any fixed positive threshold
therefore yields

\[
\boxed{
\tau_d^{K\text{-softmax}}(\delta)
=\Theta(\delta^{-d}).
}
\]

So the exponent obtained in the scalar binary-logit calculation is not a
one-logit artifact. A full trainable softmax row changes constants, including a
factor-of-two metric difference already visible at `K=2`, but preserves the
`delta^{-d}` rare-edge exponent.

## What remains binary-specific

The closed-form antiderivative

\[
F_d(p)
\]

and exact coefficient

\[
(Cd)^{-1}\delta^{-d}
\]

in the companion note refer to the scalar binary-logit metric
`p=sigmoid(z)`. A full two-logit softmax already has a different Euclidean metric
factor because both logits move. For `K>2`, the prefactor also depends on how
non-target probability mass is distributed.

Therefore the robust statement is:

\[
\boxed{
\text{full softmax: construction degree }d
\Longrightarrow
\text{rare-edge bootstrap exponent }d
}
\]

for the isolated symmetric route considered here, **up to positive constants**.
The structural effect of dormant scaffolding remains one-for-one: reducing
construction order by one removes one power of the rare-edge initialization from
the asymptotic time.

This extension does not claim a corresponding exact theorem for coupled route
sums, shared softmax rows, adaptive optimizers, or arbitrary non-symmetric
trajectories. Its role is to establish that the boundary exponent is stable under
the ordinary full-row softmax geometry used by stochastic controllers.
