# Construction order at a softmax boundary

The finite-memory construction-order theorem is most transparent in affine
transition-probability coordinates, where an absent edge is represented by
probability zero. A practical stochastic controller is often parameterized by
finite logits instead. Exact probability zero then lies at logit `-infinity`, so
the regular-coordinate invariance theorem does **not** apply at the construction
origin.

This note quantifies that boundary effect rather than treating it as a generic
parameterization caveat.

## Binary-logit model

Consider an isolated beneficial square-free construction

\[
L-L_0=c\prod_{i=1}^d p_i,
\qquad c<0,
\]

where each missing transition probability is parameterized by a binary logit

\[
p_i=\sigma(z_i).
\]

Start on the symmetric manifold

\[
p_1=\cdots=p_d=p.
\]

Symmetry is preserved by Euclidean gradient flow in the logits.

The probability-coordinate derivative is

\[
\frac{\partial L}{\partial p_i}
=c p^{d-1},
\]

while

\[
\frac{dp_i}{dz_i}=p(1-p).
\]

Therefore ordinary logit gradient flow gives

\[
\boxed{
\dot z=(-c)p^d(1-p)
}
\]

and, after mapping the trajectory back into edge probability,

\[
\boxed{
\dot p=(-c)p^{d+1}(1-p)^2.
}
\]

Near the absent-edge boundary,

\[
\dot p=(-c)p^{d+1}(1+O(p)).
\]

Direct Euclidean flow in probability coordinates would instead scale as

\[
\dot p=(-c)p^{d-1}.
\]

Thus the logit metric contributes exactly two additional powers of the rare-edge
probability to the **probability velocity**:

\[
\boxed{
\frac{\dot p_{\rm logit}}{\dot p_{\rm prob}}
=p^2(1-p)^2.
}
\]

This is not a contradiction with regular-coordinate invariance. At every fixed
interior probability the logit/probability maps are smooth local diffeomorphisms,
but the limiting construction origin `p=0` corresponds to `z=-infinity` and is
not covered by a finite regular chart.

## Exact completion time

Let `C=-c>0`. The symmetric probability flow is

\[
\dot p=Cp^{d+1}(1-p)^2.
\]

The exact time from edge probability \(\delta\) to \(\theta\), with
\(0<\delta<\theta<1\), is

\[
\tau_d^{\rm logit}(\delta,\theta)
=
\frac1C
\int_\delta^\theta
\frac{dp}{p^{d+1}(1-p)^2}.
\]

For integer \(d\ge1\), use

\[
\frac1{p^{d+1}(1-p)^2}
=
\frac{d+1}{1-p}
+
\frac1{(1-p)^2}
+
\sum_{k=1}^{d+1}
\frac{d+2-k}{p^k}.
\]

One antiderivative is

\[
F_d(p)
=
(d+1)\log\frac{p}{1-p}
+
\frac1{1-p}
-
\sum_{j=1}^{d}
\frac{d+1-j}{j}p^{-j}.
\]

Hence

\[
\boxed{
\tau_d^{\rm logit}(\delta,\theta)
=
\frac{F_d(\theta)-F_d(\delta)}{C}.
}
\]

The leading divergent term as \(\delta\to0\), with fixed positive \(\theta\), is

\[
\boxed{
\tau_d^{\rm logit}(\delta,\theta)
=
\frac{1}{Cd}\delta^{-d}(1+o(1)).
}
\]

Equivalently, for a large negative initial logit \(z_0\),

\[
\delta=\sigma(z_0)\sim e^{z_0}
\]

and

\[
\boxed{
\tau_d^{\rm logit}
\sim
\frac1{Cd}e^{-d z_0}
\qquad z_0\to-\infty.
}
\]

So finite negative logits never create an exactly stuck construction, but the
bootstrap time becomes exponentially large in the magnitude of the negative
initial logit.

## Comparison with affine probability-coordinate flow

For the same isolated degree-\(d\) monomial, direct Euclidean probability flow has

\[
\tau_d^{\rm prob}(\delta)
\sim
\begin{cases}
O(1),&d=1,\\
O(\log(1/\delta)),&d=2,\\
\Theta(\delta^{-(d-2)}),&d\ge3.
\end{cases}
\]

By contrast, binary-logit Euclidean flow gives

\[
\boxed{
\tau_d^{\rm logit}(\delta)=\Theta(\delta^{-d})
\quad\text{for every }d\ge1.
}
\]

The **absolute** bootstrap class is therefore optimizer/parameterization
sensitive at the simplex boundary.

The structural scaffold comparison survives in a useful form. Consecutive orders
satisfy

\[
\frac{\tau_d^{\rm logit}}
{\tau_{d-1}^{\rm logit}}
\sim
\frac{d-1}{d}\delta^{-1}.
\]

Thus each unit by which dormant prewiring reduces construction order removes one
power of rare-edge initialization from the logit bootstrap time.

This gives a more practical interpretation of the topology result:

\[
\boxed{
\text{construction order is structural; the map from order to wall-clock
bootstrap class depends on the parameterization/metric.}
}
\]

In affine probability coordinates the order-`d` isolated class has the familiar
`d-2` exponent. Under standard rare-edge logits it has exponent `d`. In either
case, reducing construction order makes the asymptotic bottleneck strictly less
severe.

## Relation to exact zero initialization

An exact absent probability `p=0` has no finite binary logit. Therefore statements
about a finite logit initialized at exact probability zero are ill-posed.

There are two distinct limits:

1. **Affine probability construction origin:** `p=0` is a finite boundary point.
   Degree `d>=2` isolated probability-coordinate gradient flow is exactly stuck at
   zero.
2. **Logit approach to the boundary:** initialize `z_0` finite but increasingly
   negative. Then `p>0` and the gradient is nonzero, but the construction time
   diverges as `exp(d |z_0|)`.

The finite-memory theorem concerns the first local polynomial geometry. This note
shows how that geometry manifests when approached through the common softmax/logit
parameterization.

## Scope

The formulas above are exact for a symmetric **binary-logit** parameterization of
independent missing edges and an isolated square-free leading monomial. A general
multiway softmax with all logits trained introduces rowwise coupling and gauge
freedom; the rare-edge Jacobian still vanishes linearly with rare probability,
but the exact coefficient geometry can differ. We do not claim optimizer
invariance or an exact `delta^{-d}` theorem for arbitrary coupled softmax rows,
adaptive optimizers, or non-isolated route sums.

The purpose of this result is narrower: it closes the most obvious practical
boundary objection to the construction-order paper. The exact finite-memory order
is not itself the logit escape exponent, but its scaffold advantage remains
asymptotically visible near a softmax boundary.
