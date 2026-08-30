# Construction order under regular preconditioning

The parameterization result establishes that the scalar loss-germ order is stable
under every smooth local reparameterization with nonsingular Jacobian. Ordinary
Euclidean gradient trajectories are not invariant, because a regular coordinate
change induces a positive-definite local metric.

This note asks the next question:

> does ordinary, nonsingular conditioning change only construction speed, or can
> it change the degree-controlled bootstrap class itself?

For isolated monomial constructions, positive diagonal conditioning admits an
exact answer.

## Diagonal-preconditioned monomial flow

Let

\[
L(x)=c\prod_{i=1}^p x_i^{\alpha_i},
\qquad c<0,
\qquad \alpha_i\in\mathbb N_{>0},
\]

and optimize with the constant positive diagonal metric

\[
M=\operatorname{diag}(m_1,\ldots,m_p),
\qquad m_i>0.
\]

The preconditioned gradient flow is

\[
\dot x=-M\nabla L.
\]

Writing

\[
C=-c>0,
\qquad
P(x)=\prod_i x_i^{\alpha_i},
\]

gives

\[
\dot x_i
=Cm_i\alpha_i\frac{P(x)}{x_i}.
\]

Therefore

\[
\frac{d}{dt}\frac{x_i^2}{m_i\alpha_i}
=
2CP(x)
\]

for every active coordinate. Hence

\[
\boxed{
\frac{x_i^2}{m_i\alpha_i}
-
\frac{x_j^2}{m_j\alpha_j}
=\text{constant}.
}
\]

This is the metric-weighted extension of the earlier exponent-vector balance
law. The metric does not destroy the integrable isolated-monomial geometry; it
changes the natural balanced coordinates.

## Metric-balanced reduction

On the balanced manifold

\[
x_i=\sqrt{m_i\alpha_i}\,s,
\]

let

\[
d=\sum_i\alpha_i
\]

and

\[
A_{M,\alpha}
=
\prod_i(m_i\alpha_i)^{\alpha_i/2}.
\]

Substitution into the exact flow gives

\[
\boxed{
\dot s=C A_{M,\alpha}s^{d-1}.
}
\]

Thus the exact completion time from normalized scale \(\delta\) to \(\theta\)
is the same scalar formula as in Euclidean coordinates, with only the effective
coefficient changed from \(c\) to

\[
cA_{M,\alpha}.
\]

Consequently the small-initialization classes remain

\[
\boxed{
\begin{array}{c|c}
 d & \tau_d(\delta) \\
\hline
1 & O(1) \\
2 & O(\log(1/\delta)) \\
d\ge3 & \Theta(\delta^{-(d-2)}).
\end{array}}
\]

A strictly positive diagonal preconditioner changes the balance metric and the
multiplicative time constant, but not the degree-controlled asymptotic class.

## Exact finite-memory check

The CI regression uses the exact leading coefficient of the delay-4 finite-memory
chain construction. Its leading useful monomial has total degree four, so the
Euclidean theory predicts

\[
\tau\asymp\delta^{-2}.
\]

The same exact monomial is evaluated under two positive diagonal metrics,
including

\[
M=\operatorname{diag}(0.5,2,1.5,3).
\]

The completion times differ by the exact prefactor

\[
A_M=\prod_i\sqrt{m_i},
\]

while the divergence power remains

\[
\boxed{2=d-2.}
\]

This is a direct finite-memory instance of metric-dependent speed but
metric-stable construction class.

## General homogeneous scaling

The degree robustness is not special to diagonal matrices at the level of local
time rescaling. If \(P_d\) is homogeneous of degree \(d\) and \(M\) is any
constant matrix, then

\[
F(x)=-M\nabla P_d(x)
\]

is homogeneous of degree \(d-1\):

\[
F(\lambda x)=\lambda^{d-1}F(x).
\]

Therefore solutions of the homogeneous leading system obey the exact scaling

\[
\boxed{
x_{\lambda}(t)
=\lambda x_1(\lambda^{d-2}t)}
\]

whenever the compared initial conditions differ by the common factor \(\lambda\).
This identifies \(\delta^{2-d}\) as the natural local time rescaling for every
constant nonsingular metric.

For arbitrary coupled homogeneous systems, converting this local rescaling into
a fixed-threshold escape theorem additionally requires trajectory/nondegeneracy
conditions. The diagonal isolated-monomial result above avoids that issue and is
exact globally on its balanced manifold.

## Connection to regular parameterization

For a regular coordinate map

\[
\varepsilon=\phi(\theta),
\]

Euclidean gradient flow in \(\theta\), pushed into \(\varepsilon\)-space, is

\[
\dot\varepsilon
=-D\phi(\theta)D\phi(\theta)^T\nabla_\varepsilon L.
\]

Near the construction origin,

\[
D\phi(\theta)D\phi(\theta)^T
=M_0+O(\|\theta\|),
\]

with \(M_0\) positive definite whenever the chart is regular. Thus the
parameterization theorem and the present preconditioning result fit together:

1. regular charts preserve the scalar vanishing degree;
2. their induced nonsingular metric changes local direction and speed;
3. the homogeneous leading vector field still carries the same degree-based
   natural time scale;
4. singular charts or degenerate metrics are the mechanisms capable of changing
   the effective order/class.

The exact diagonal theorem is deliberately narrower than a claim that arbitrary
optimizers or arbitrary state-dependent metrics preserve threshold-crossing
behavior.

## Prior-art boundary

Homogeneous gradient-flow time rescaling and small-initialization dynamics are
established topics in deep homogeneous and deep-linear optimization. Likewise,
preconditioning and reparameterization are standard geometric optimization
ideas. These general mechanisms are not novelty claims.

The finite-memory contribution is the robustness consequence for the
construction-order framework:

\[
\boxed{
\text{for an isolated finite-memory construction monomial, ordinary positive
conditioning changes constants and balance geometry, but not the
construction-order escape class.}
}
\]

Together with regular-coordinate invariance of the loss order, this substantially
narrows the set of parameterization/optimizer effects that can explain away the
forward-equivalent accessibility gaps.