# Dynamics and validation appendices

This file supplies Appendices E–G from `paper/appendix_map.md`. The optimization
identities for isolated homogeneous monomials are used as known dynamical
machinery; the paper's finite-memory contribution is the exact map from latent
construction topology to the degree of the homogeneous term to which that
machinery applies.

## Appendix E — Isolated construction dynamics in affine coordinates

### E.1 Square-free monomial flow

Consider an isolated beneficial leading term

\[
L(x)-L_0=-C\prod_{i=1}^d x_i,
\qquad C>0,
\]

with independent nonnegative construction coordinates. Euclidean gradient flow is

\[
\dot x_i
=C\prod_{j\ne i}x_j.
\]

If all coordinates start equally, \(x_i(0)=\delta\), symmetry is preserved and
\(x_i(t)=x(t)\) obeys

\[
\boxed{\dot x=Cx^{d-1}.}
\]

For a fixed threshold \(\theta>\delta\), the exact completion time is

\[
\boxed{
\tau_d(\delta,\theta)=
\begin{cases}
(\theta-\delta)/C, & d=1,\\[4pt]
C^{-1}\log(\theta/\delta), & d=2,\\[4pt]
\dfrac{\delta^{2-d}-\theta^{2-d}}{C(d-2)}, & d\ge3.
\end{cases}
}
\]

Thus as \(\delta\to0\) with fixed positive \(\theta\),

\[
\boxed{
\tau_d=
\begin{cases}
O(1), & d=1,\\
O(\log(1/\delta)), & d=2,\\
\Theta(\delta^{-(d-2)}), & d\ge3.
\end{cases}
}
\]

At exact zero initialization, degree one moves immediately while an isolated
square-free route of degree \(d\ge2\) is stationary.

These homogeneous-flow exponents are not claimed as new general optimization
mathematics. Their role here is to translate an exact finite-memory construction
order into a local bootstrap class under the stated affine-coordinate Euclidean
metric.

### E.2 Repeated parameters and exponent vectors

Let

\[
L(x)-L_0=-C\prod_{i=1}^p x_i^{\alpha_i},
\qquad \alpha_i\in\mathbb N_{>0}.
\]

The gradient flow is

\[
\dot x_i
=C\alpha_i x_i^{\alpha_i-1}
\prod_{j\ne i}x_j^{\alpha_j}.
\]

Writing \(P(x)=\prod_jx_j^{\alpha_j}\),

\[
\frac{d}{dt}x_i^2
=2C\alpha_iP(x),
\]

so

\[
\boxed{
\frac{d}{dt}\frac{x_i^2}{\alpha_i}=2CP(x)
}
\]

and therefore

\[
\boxed{
\frac{x_i^2}{\alpha_i}-\frac{x_j^2}{\alpha_j}
\quad\text{is conserved.}
}
\]

On the weighted-balanced manifold

\[
x_i=\sqrt{\alpha_i}\,s,
\]

let

\[
d=\sum_i\alpha_i,
\qquad
A_\alpha=\prod_i\alpha_i^{\alpha_i/2}.
\]

Then

\[
\boxed{\dot s=CA_\alpha s^{d-1}.}
\]

Hence the total degree \(d\), not the number of distinct scalar coordinates,
controls the small-initialization divergence exponent on this balanced manifold;
parameter multiplicity changes the metric/prefactor.

If one scalar is reused \(d\) times, so that \(L-L_0=-Cx^d\), then

\[
\dot x=Cd x^{d-1},
\]

and the completion time at the same raw scalar initialization and threshold is
exactly \(1/d\) of the independent symmetric-coordinate completion time.

### E.3 Positive diagonal preconditioning

Let the optimizer use a fixed positive diagonal preconditioner

\[
M=\operatorname{diag}(m_1,\ldots,m_p),
\qquad m_i>0,
\]

so \(\dot x=-M\nabla L\). For the exponent-vector monomial above,

\[
\frac{d}{dt}x_i^2=2Cm_i\alpha_iP(x),
\]

which gives conserved differences

\[
\boxed{
\frac{x_i^2}{m_i\alpha_i}
-
\frac{x_j^2}{m_j\alpha_j}.
}
\]

On the corresponding metric-balanced manifold

\[
x_i=\sqrt{m_i\alpha_i}\,s,
\]

we obtain

\[
\dot s
=C A_{\alpha,m}s^{d-1},
\qquad
A_{\alpha,m}
=\prod_i(m_i\alpha_i)^{\alpha_i/2}.
\]

A fixed strictly positive diagonal metric therefore changes the balanced geometry
and prefactor but not the degree-controlled finite/logarithmic/polynomial class.
This statement is deliberately narrower than optimizer invariance: adaptive,
state-dependent, singular, or non-diagonal metrics can change the time map.

### E.4 Finite-memory full-polynomial audits

The exact completion-time statements above apply to the isolated leading
homogeneous term. In the finite-memory witnesses the full finite-horizon loss is a
higher-degree polynomial. Numerical integration of the exact full polynomial
vector field shows that, over the frozen local thresholds used in the repository,
full-polynomial completion times remain close to the leading-order prediction and
preserve the scaffold ordering. These integrations are validation evidence, not
premises of the theorem; higher-order terms can dominate later or reverse a
near-tied route, as Appendix I documents.

## Appendix F — Parameterization and simplex-boundary geometry

### F.1 Regular local charts preserve scalar vanishing order

Let a scalar objective around the construction origin have

\[
L(\varepsilon)-L(0)=P_d(\varepsilon)+O(\|\varepsilon\|^{d+1}),
\]

where \(P_d\) is a nonzero homogeneous polynomial of degree \(d\). Consider a
smooth local reparameterization

\[
\varepsilon=\phi(\theta),
\qquad
\phi(0)=0,
\qquad
D\phi(0)=J.
\]

If \(J\) is nonsingular, then

\[
\phi(\theta)=J\theta+O(\|\theta\|^2)
\]

and therefore

\[
L(\phi(\theta))-L(0)
=P_d(J\theta)+O(\|\theta\|^{d+1}).
\]

Because composition of a nonzero polynomial with an invertible linear map cannot
make it identically zero,

\[
P_d\circ J\not\equiv0.
\]

Hence

\[
\boxed{
\operatorname{ord}_0(L\circ\phi)
=
\operatorname{ord}_0 L=d.
}
\]

The scalar construction order is therefore invariant under smooth finite local
coordinate changes with nonsingular Jacobian. This is an order statement, not a
claim that Euclidean gradient trajectories or wall-clock times are coordinate
invariant.

### F.2 Singular power charts change the effective order

For the singular diagonal chart

\[
\varepsilon_i=\theta_i^{r_i},
\qquad r_i\in\mathbb N_{>0},
\]

an active monomial \(c_\alpha\varepsilon^\alpha\) pulls back to

\[
c_\alpha\theta_1^{r_1\alpha_1}\cdots
\theta_n^{r_n\alpha_n}.
\]

The exponent map is injective, so distinct monomials do not merge under this
chart. The pulled-back scalar vanishing order is exactly

\[
\boxed{
\min_{\alpha:\,c_\alpha\ne0,\,|\alpha|>0}
\sum_i r_i\alpha_i.
}
\]

Thus singular coordinates can alter scalar order. This is why the regular-chart
result must not be extrapolated to a simplex boundary.

### F.3 Binary-logit rare-edge boundary

Consider again an isolated beneficial square-free construction

\[
L-L_0=c\prod_{i=1}^dp_i,
\qquad c<0,
\]

but parameterize each missing probability by a binary logit

\[
p_i=\sigma(z_i).
\]

On the symmetric manifold \(p_i=p\), ordinary Euclidean gradient flow in logits
gives

\[
\dot z=(-c)p^d(1-p)
\]

and therefore

\[
\boxed{
\dot p=(-c)p^{d+1}(1-p)^2.
}
\]

Direct Euclidean flow in affine probability coordinates instead has speed
\((-c)p^{d-1}\). Thus the logit-induced probability velocity is slower by the
exact metric factor

\[
\boxed{p^2(1-p)^2.}
\]

This does not contradict regular-chart order invariance: \(p=0\) corresponds to
\(z=-\infty\), not to a finite point with nonsingular coordinate Jacobian.

Let \(C=-c>0\). The exact completion time from
\(0<\delta<\theta<1\) is

\[
\tau_d^{\rm logit}(\delta,\theta)
=\frac1C\int_\delta^\theta
\frac{dp}{p^{d+1}(1-p)^2}.
\]

Using

\[
\frac1{p^{d+1}(1-p)^2}
=\frac{d+1}{1-p}+\frac1{(1-p)^2}
+\sum_{k=1}^{d+1}\frac{d+2-k}{p^k},
\]

one antiderivative is

\[
F_d(p)
=(d+1)\log\frac{p}{1-p}
+\frac1{1-p}
-\sum_{j=1}^d\frac{d+1-j}{j}p^{-j}.
\]

Hence

\[
\boxed{
\tau_d^{\rm logit}(\delta,\theta)
=\frac{F_d(\theta)-F_d(\delta)}{C}
}
\]

and, for fixed positive \(\theta\),

\[
\boxed{
\tau_d^{\rm logit}(\delta,\theta)
=\frac1{Cd}\delta^{-d}(1+o(1)).
}
\]

If \(z_0\to-\infty\), then \(\delta=\sigma(z_0)\sim e^{z_0}\), so

\[
\tau_d^{\rm logit}\sim\frac1{Cd}e^{-dz_0}.
\]

Finite logits therefore never represent an exactly absent edge, but approaching
the boundary produces an exponentially severe bootstrap in the magnitude of the
negative initial logit.

Consecutive construction orders satisfy

\[
\boxed{
\frac{\tau_d^{\rm logit}}
{\tau_{d-1}^{\rm logit}}
\sim\frac{d-1}{d}\delta^{-1},
}
\]

so reducing construction order by one removes one power of rare-edge
initialization even though the absolute time class differs from affine
probability-coordinate flow.

### F.4 Full \(K\)-way softmax target-edge bound

For a full \(K\)-way softmax row, let target probability be \(p\) and let the
remaining probabilities be \(q_a\), with \(\sum_aq_a=1-p\). The target
probability has logit gradient

\[
\frac{\partial p}{\partial z_{\rm target}}=p(1-p),
\qquad
\frac{\partial p}{\partial z_a}=-pq_a.
\]

Therefore

\[
\boxed{
\|\nabla_zp\|^2
=p^2\left((1-p)^2+\sum_aq_a^2\right).
}
\]

Cauchy--Schwarz and concentration of the non-target mass imply the sharp
dimension-only bounds

\[
\boxed{
\frac{K}{K-1}p^2(1-p)^2
\le
\|\nabla_zp\|^2
\le
2p^2(1-p)^2.
}
\]

The lower bound is attained by equal non-target probabilities; the upper bound is
approached as their mass concentrates on one category.

For \(d\) identical rows with target probability \(p\) and isolated objective
\(L=c\prod_ip_i\),

\[
\dot p=(-c)p^{d-1}\|\nabla_zp\|^2.
\]

For fixed \(K\), the bounds above give

\[
\boxed{\dot p=\Theta(p^{d+1})\quad(p\to0),}
\]

and therefore the same rare-edge bootstrap exponent \(\delta^{-d}\) as in the
exact binary-logit calculation, up to dimension/non-target-distribution-dependent
constants. The paper does not extend this statement to arbitrary coupled route
sums, adaptive optimizers, or non-isolated softmax dynamics.

## Appendix G — Smooth linear state-space validation

The finite-state theorem uses exact stochastic-controller polynomials. To verify
that construction order is not an artifact of that discrete formalism, we use an
independent differentiable recurrent system trained by ordinary PyTorch autograd.

### G.1 Five-link linear delay line

For scalar inputs \(u_t\), define a depth-\(R\) recurrent delay chain

\[
h_{1,t+1}=w_1u_t,
\qquad
h_{i,t+1}=w_i h_{i-1,t},\quad i=2,\ldots,R,
\]

with prediction \(\hat u_t=h_{R,t}\). For a valid delayed target, the end-to-end
gain is

\[
\boxed{g=\prod_{i=1}^Rw_i.}
\]

The frozen binary input sequence has squared target value one at every scored
time. Consequently the mean-square objective reduces exactly to

\[
L=\frac12(g-1)^2.
\]

The collapsed current predictor is obtained whenever at least one link is zero.
For each construction order \(r\in\{1,\ldots,5\}\), initialize the first \(r\)
links to zero and all downstream links to one. Every such initialization predicts
zero at every time and has exactly the same current loss \(1/2\).

### G.2 Exact ray formula and autograd scaling

Move the \(r\) missing links together to a small positive scale \(s\), leaving the
prewired downstream links equal to one. Then

\[
g=s^r
\]

and the exact improvement from the collapsed loss is

\[
\boxed{
\frac12-L
=s^r-\frac12s^{2r}.
}
\]

Hence the leading loss improvement has order \(r\). Differentiating the exact MSE
with respect to each missing link gives the missing-coordinate gradient norm

\[
\boxed{
\|\nabla_{1:r}L\|_2
=\sqrt r\,(1-s^r)s^{r-1}.
}
\]

Thus the gradient norm has leading order \(r-1\). The regression suite compares
ordinary PyTorch autograd against both formulas to machine precision.

Log-log fits over the frozen scales recover loss slopes \(1,2,3,4,5\) and gradient
slopes \(0,1,2,3,4\), respectively, within the stated regression tolerances.

### G.3 Exact-zero cutoff

At the collapsed base initialization,

\[
s=0.
\]

For \(r=1\), the missing-link gradient is nonzero. For every \(r\ge2\), every
missing-link gradient is exactly zero. Thus the smooth recurrent model reproduces
the same local derivative-order cutoff without hard controller transitions,
surrogate gradients, or a finite-state loss oracle.

### G.4 Illustrative SGD times

A separate fixed-step SGD experiment uses the same five-link system to show that
the derivative-order hierarchy produces a large practical bootstrap separation.
For the frozen optimizer configuration, the approximate threshold steps are

| order | steps |
|---:|---:|
| 1 | 4 |
| 2 | 43 |
| 3 | 361 |
| 4 | 3965 |
| 5 | 53327 |

These counts are deliberately labeled **illustrative evidence**. They depend on
the chosen learning rate, threshold, parameterization, and optimizer and are not
a universal time theorem. The exact validation claim is the analytically verified
loss/gradient order spectrum and the shared current zero predictor.
