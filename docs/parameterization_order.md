# Construction order under reparameterization

The general construction-order theorem identifies the first nonconstant degree of
an exact finite-horizon loss polynomial in affine transition-strength coordinates.
A natural objection is whether this degree is merely an artifact of choosing those
coordinates.

The answer has a sharp boundary:

\[
\boxed{
\text{regular local coordinate changes preserve construction/loss order.}
}
\]

Singular parameterizations can change it.

This note separates three statements that should not be conflated:

1. the **vanishing order of the scalar objective germ** is invariant under a
   local diffeomorphism;
2. the **ordinary Euclidean gradient-flow trajectory** is not generally
   invariant under that same coordinate change;
3. a **singular** parameterization can change the objective order itself and
   therefore can change the small-initialization bootstrap class.

The first fact is classical function-germ mathematics rather than a new general
theorem. Its role here is to set the correct invariance boundary for the
finite-memory construction-order claim.

## Objective-germ order

Let

\[
F(\varepsilon)-F(0)
=P_d(\varepsilon)+O(\|\varepsilon\|^{d+1}),
\qquad P_d\not\equiv0,
\]

where `P_d` is homogeneous of degree `d`. The vanishing order is

\[
\operatorname{ord}_0F=d.
\]

For the finite-memory systems in this repository, `F` can be the exact
finite-horizon expected log loss and `d` can be the scalar construction/loss
order from the support/operator/loss hierarchy.

## Regular-coordinate invariance

Let

\[
\varepsilon=\phi(\theta),
\qquad
\phi(0)=0,
\]

and assume `phi` is smooth enough for the required Taylor expansion. Write

\[
\phi(\theta)=J\theta+O(\|\theta\|^2).
\]

If

\[
\det J\ne0,
\]

then

\[
F(\phi(\theta))-F(0)
=P_d(J\theta)+O(\|\theta\|^{d+1}).
\]

Because `J` is invertible, a nonzero homogeneous polynomial cannot become the
zero polynomial after composition with `J`. Hence

\[
\boxed{
\operatorname{ord}_0(F\circ\phi)
=
\operatorname{ord}_0F.
}
\]

So construction order is not arbitrary under ordinary smooth changes of local
coordinates. It is an invariant of the scalar loss germ under local
diffeomorphism.

`compose_sparse_polynomial` freezes polynomial examples of the nonlinear case,
and `linear_pullback_coefficients` freezes exact changes of linear basis.

### Exact finite-memory regression

The delay-3 independent-link finite-memory polynomial has

\[
\operatorname{ord}_0L=3.
\]

The CI regression pulls the complete exact polynomial back through the invertible
matrix

\[
J=
\begin{pmatrix}
1&0.2&0\\
0&1&-0.3\\
0.1&0&1
\end{pmatrix},
\]

and verifies that the transformed polynomial still begins at degree three.

Thus the exact memory construction order survives a nontrivial mixing of the
transition-strength coordinates rather than only separate rescalings.

## The same forward-equivalence separation survives regular charts

Suppose two forward-equivalent controller parameter points have scalar loss
orders

\[
d_A\ne d_B
\]

in the affine transition-strength charts used by the construction theorem.
Apply arbitrary regular local coordinate changes independently to the two
systems. The theorem above gives

\[
\operatorname{ord}_0L_A'=d_A,
\qquad
\operatorname{ord}_0L_B'=d_B.
\]

Therefore

\[
\boxed{
d_A\ne d_B}
\]

remains true in every regular local chart.

This is the relevant response to the claim that the dormant-scaffold order gap
is only a coordinate convention. Ordinary invertible smooth reparameterization
cannot erase the gap.

## Singular maps can raise order

If `J` is singular, the leading form can vanish after pullback:

\[
P_d(J\theta)\equiv0.
\]

Then a higher-order term determines the transformed objective order.

A CI regression uses

\[
F(\varepsilon_1,\varepsilon_2)
=(\varepsilon_1-\varepsilon_2)^2+
\varepsilon_2^3
\]

and the rank-one map

\[
\varepsilon_1=\theta_1,
\qquad
\varepsilon_2=\theta_1.
\]

The quadratic initial form is annihilated exactly and

\[
F\circ\phi=\theta_1^3.
\]

Thus

\[
\boxed{2\to3}
\]

under a singular parameterization.

This does not contradict regular-coordinate invariance: the map is not a local
diffeomorphism and does not provide an equivalent local coordinate chart.

## Exact weighted-degree law for power charts

Consider the diagonal singular parameterization

\[
\varepsilon_i=\theta_i^{r_i},
\qquad r_i\in\mathbb N_{>0}.
\]

For

\[
F(\varepsilon)
=
\sum_\alpha c_\alpha\varepsilon^\alpha,
\]

the pullback is exactly

\[
F(\theta_1^{r_1},\ldots,\theta_n^{r_n})
=
\sum_\alpha c_\alpha
\theta_1^{r_1\alpha_1}\cdots
\theta_n^{r_n\alpha_n}.
\]

Because multiplication by strictly positive integer powers is injective on
exponent vectors, distinct monomials do not merge. Therefore

\[
\boxed{
\operatorname{ord}_0(F\circ\phi)
=
\min_{c_\alpha\ne0}
\sum_i r_i\alpha_i.
}
\]

The ordinary total degree is replaced by an exact weighted degree.

`diagonal_power_pullback_coefficients` and
`diagonal_weighted_vanishing_order` implement this law.

For the exact delay-3 finite-memory polynomial, the uniform square chart

\[
\varepsilon_i=\theta_i^2
\]

changes

\[
\boxed{3\to6}.
\]

This is a coordinate-induced order increase, but it is produced by a singular
chart whose Jacobian vanishes at the construction origin.

## Connection to the joint decoder result

The earlier joint transition/decoder witness has leading term

\[
-\frac1{14}a\varepsilon_1\varepsilon_2
\]

without a scaffold and

\[
-\frac1{14}a\varepsilon_1
\]

with the dormant downstream link prewired.

A regular decoder coordinate

\[
a=c\,u+O(u^2),\qquad c\ne0,
\]

preserves the joint orders three and two.

A singular decoder chart such as

\[
a=u^2
\]

raises them to four and three. This formalizes the earlier warning that “decoder
learning adds one factor” is not invariant under singular decoder
reparameterization, while showing that it **is** stable under regular local
decoder coordinates.

## Objective order versus Euclidean gradient dynamics

Regular-coordinate invariance of the scalar order does **not** make ordinary
Euclidean gradient flow coordinate invariant.

Let

\[
\widetilde F(\theta)=F(\phi(\theta)).
\]

Euclidean gradient flow in theta is

\[
\dot\theta
=-D\phi(\theta)^T\nabla_\varepsilon F.
\]

Mapped back to epsilon coordinates,

\[
\boxed{
\dot\varepsilon
=-D\phi(\theta)D\phi(\theta)^T
\nabla_\varepsilon F.
}
\]

Direct epsilon-Euclidean flow would instead be

\[
\dot\varepsilon=-\nabla_\varepsilon F.
\]

The coordinate change therefore acts as a state-dependent positive-definite
preconditioner when `Dphi` is invertible. Exact paths and time constants can
change even though objective order cannot.

For the scalar regular scaling

\[
\varepsilon=a\theta,
\]

theta-Euclidean flow mapped back to epsilon is faster than direct
epsilon-Euclidean flow by the exact factor

\[
a^2.
\]

The CI regression freezes this identity for a degree-four monomial.

This is standard Riemannian-optimization geometry. Natural/Riemannian gradient
methods can be formulated to transform covariantly when the metric is pulled
back with the parameterization. We do not claim reparameterization invariance of
ordinary Euclidean optimization.

## Small-initialization class

For an isolated beneficial monomial with scalar degree `d`, the previously
established symmetric Euclidean gradient-flow scaling is

\[
\tau_d\sim
\begin{cases}
O(1), & d=1,\\
O(\log(1/\delta)), & d=2,\\
\Theta(\delta^{-(d-2)}), & d\ge3.
\end{cases}
\]

A regular local coordinate change preserves `d`, so it preserves the degree that
sets the natural local time rescaling, although ordinary Euclidean trajectories
and constants remain metric-dependent.

A singular power parameterization can replace `d` by a larger weighted degree.
For the scalar map

\[
\varepsilon=\theta^r,
\qquad
F=-C\varepsilon^d,
\]

we obtain

\[
F\circ\phi=-C\theta^{rd}.
\]

Thus singular parameterization can alter the Euclidean bootstrap exponent from
`d-2` to `rd-2` when the polynomial-order regime applies.

That effect is a property of the chosen singular parameterization, not of the
underlying regular objective germ.

## Practical chart boundary

For stochastic transition probabilities in the **interior** of a simplex,
minimal probability coordinates and gauge-fixed logits are related by a smooth
local diffeomorphism. The order theorem therefore survives ordinary interior
logit/probability changes of coordinates.

At a probability-simplex **boundary**, a zero transition probability corresponds
to a divergent softmax logit rather than a finite regular chart. Parameterizations
used to approach or encode exact zeros can therefore be singular from the local
point of view relevant here. The fixed-affine transition-strength chart remains
the clean structural coordinate system for the exact construction theorem.

Hard-forward/STE argmax-cell dynamics are a separate object again: their forward
map is piecewise constant rather than a smooth reparameterization of the
stochastic transition family.

## Prior-art boundary

Order of vanishing/multiplicity of a function germ under diffeomorphism is a
classical local invariant; the mathematical mechanism is not claimed as novel.
Likewise, parameterization dependence of ordinary Euclidean gradients and
reparameterization invariance of appropriately defined natural/Riemannian
gradients are established information-geometry results.

The finite-memory contribution is the consequence for the construction-order
framework:

\[
\boxed{
\text{the topology-induced accessibility-order gap is stable under every
regular local chart, while singular parameterizations have an exact weighted
order law.}
}
\]

This substantially narrows the coordinate-dependence caveat attached to the
main construction theorem.
