# Appendices

This document is the paper-facing proof supplement. It follows the allocation in
`paper/appendix_map.md`: the main text carries only the definitions, proof ideas,
and headline witnesses needed for the causal story; detailed algebra and protocols
live here.

## Appendix A — Exact finite-memory polynomial machinery

### A.1 Source-memory dynamics

Let the source have finite state set \(\mathcal S\), alphabet \(\mathcal X\),
emission probabilities \(e_s(x)\), and deterministic source update
\(f(s,x)\). A controller has memory states \(\mathcal M\), reset memory
\(m_0\), and transition law \(P_\varepsilon(m'\mid m,x)\).

We use an affine local family

\[
P_\varepsilon=P_0+\sum_{j=1}^n\varepsilon_jD_j,
\]

where each direction has zero row sum. Let

\[
\mu_t^\varepsilon(s,m)=\Pr(S_t=s,M_t=m)
\]

be the occupancy before prediction at time \(t\). If the source is initialized in
its stationary law \(\pi\), then

\[
\mu_0(s,m)=\pi(s)\mathbf 1\{m=m_0\}.
\]

Define product-state propagation operators \(\mathcal B_0,\mathcal B_1,\ldots,
\mathcal B_n\) by replacing the controller transition in one update with
\(P_0,D_1,\ldots,D_n\), respectively. Then

\[
\mathcal B(\varepsilon)=\mathcal B_0+\sum_j\varepsilon_j\mathcal B_j
\]

and

\[
\mu_{t+1}^\varepsilon=\mathcal B(\varepsilon)\mu_t^\varepsilon.
\]

### A.2 Exact multivariate coefficient recursion

Write

\[
\mu_t^\varepsilon
=\sum_{\alpha\in\mathbb N^n}\varepsilon^\alpha\mu_{t,\alpha},
\qquad
\varepsilon^\alpha=\prod_j\varepsilon_j^{\alpha_j}.
\]

Only coefficients with \(|\alpha|\le t\) can occur. Equating powers gives

\[
\boxed{
\mu_{t+1,\alpha}
=\mathcal B_0\mu_{t,\alpha}
+\sum_{j:\alpha_j>0}\mathcal B_j\mu_{t,\alpha-e_j}.
}
\]

This is the exact recursion implemented by the repository polynomial oracle. It
also gives a walk interpretation: \(\mu_{t,\alpha}\) is the signed sum over
source-valid product-state walks of length \(t\) that use perturbation direction
\(D_j\) exactly \(\alpha_j\) times and use base-controller factors on every other
step.

### A.3 Source-marginal conservation for nonconstant coefficients

The source evolves independently of the controller. Therefore for every
\(\varepsilon\),

\[
\sum_m\mu_t^\varepsilon(s,m)=\Pr(S_t=s),
\]

and the right-hand side is independent of \(\varepsilon\). Comparing polynomial
coefficients yields

\[
\boxed{
\sum_m\mu_{t,\alpha}(s,m)=0,
\qquad |\alpha|>0.
}
\]

This identity is central: perturbation coefficients can redistribute signed
occupancy among memory states, but cannot create a nonconstant source marginal.

### A.4 Finite-horizon loss polynomial

For strictly positive decoder rows \(q_m(x)\), the horizon-\(T\) expected log loss
is

\[
L_T(\varepsilon)
=-\frac1T\sum_{t=0}^{T-1}\sum_{s,m,x}
\mu_t^\varepsilon(s,m)e_s(x)\log q_m(x).
\]

Since \(\mu_t^\varepsilon\) has degree at most \(t\),

\[
\boxed{\deg L_T\le T-1.}
\]

No asymptotic Taylor approximation is involved: throughout the finite-state
analysis, the coefficient objects are exact finite-horizon polynomial
coefficients.

## Appendix B — Construction-order hierarchy and genericity

### B.1 Readout quotient and coefficient factorization

Define predictive readout equivalence by

\[
m\sim m'\iff q_m=q_{m'}.
\]

Let \(\mathcal C\) be the resulting memory-state classes. For each multi-index
\(\alpha\), define

\[
G_\alpha(C,x)
=\frac1T\sum_{t=0}^{T-1}\sum_s\sum_{m\in C}
\mu_{t,\alpha}(s,m)e_s(x).
\]

Then direct substitution into the loss gives

\[
\boxed{
[\varepsilon^\alpha]L_T
=c_\alpha
=-\sum_{C,x}G_\alpha(C,x)\log q_C(x)
=-\langle G_\alpha,\log q\rangle.
}
\]

Thus source/transition construction geometry determines \(G_\alpha\); the decoder
values enter only through the final linear pairing with \(\log q\).

### B.2 Three orders

Let \(C_0\) be the readout class containing the reset state.

The **support order** \(d_{\rm support}\) is the minimum number of perturbative
factors on any source-valid walk of length at most \(T-1\) that reaches a memory
state outside \(C_0\), assigning zero cost to supported base edges and unit cost
to edges supplied by any perturbation direction.

The **operator order** is

\[
d_{\rm operator}
=\min\{|\alpha|>0:G_\alpha\ne0\},
\]

and the **loss order** is

\[
d_{\rm loss}
=\min\{|\alpha|>0:c_\alpha\ne0\}.
\]

We restrict the hierarchy theorem to the construction-origin regime
\(d_{\rm support}\ge1\).

### B.3 Full proof of the hierarchy

**Theorem B.1 (construction-order hierarchy).** In the construction-origin
regime,

\[
\boxed{
d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}.
}
\]

**Proof.** Fix a nonzero multi-index \(\alpha\) with
\(|\alpha|<d_{\rm support}\). By the walk interpretation in Appendix A, every
contribution to \(\mu_{t,\alpha}\) uses exactly \(|\alpha|\) perturbative factors.
By definition of \(d_{\rm support}\), no such source-valid walk can terminate in a
memory state whose readout class differs from \(C_0\). Therefore

\[
\mu_{t,\alpha}(s,m)=0
\]

for every \(m\notin C_0\), every source state, and every relevant time.

For the remaining class, source-marginal conservation gives

\[
\sum_{m\in C_0}\mu_{t,\alpha}(s,m)
=\sum_m\mu_{t,\alpha}(s,m)=0,
\]

because the coefficients outside \(C_0\) vanish. Hence

\[
G_\alpha(C,x)=0
\]

for every readout class \(C\) and symbol \(x\). Thus every nonconstant operator
coefficient below degree \(d_{\rm support}\) vanishes, proving

\[
d_{\rm support}\le d_{\rm operator}.
\]

The factorization

\[
c_\alpha=-\langle G_\alpha,\log q\rangle
\]

implies that a vanishing operator coefficient necessarily gives a vanishing scalar
coefficient. Therefore no scalar loss term can appear below the first nonzero
operator degree, so

\[
d_{\rm operator}\le d_{\rm loss}.
\]

Combining the two inequalities proves the theorem. \(\square\)

### B.4 Meaning of strict inequalities

The two possible gaps have different mechanisms.

If

\[
d_{\rm operator}>d_{\rm support},
\]

minimal-cost source-valid construction walks exist but their signed,
source-weighted quotient contributions cancel. This is **path/operator
cancellation**.

If

\[
d_{\rm loss}>d_{\rm operator},
\]

the first nonzero quotient operator survives, but its image is orthogonal to the
decoder log-vector. This is **decoder cancellation**.

The exact regression suite contains a three-memory-state witness with

\[
\boxed{(d_{\rm support},d_{\rm operator},d_{\rm loss})=(1,1,2).}
\]

Its source has four states with emission rows

\[
(0.9,0.1),\;(0.5,0.5),\;(0.5,0.5),\;(0.1,0.9),
\]

and one scalar transition direction that first opens state 1 and, when reused,
can reach state 2. The state-1 decoder is chosen at the unique interior root that
annihilates the degree-one scalar coefficient, while the quotient operator itself
is nonzero at degree one and the degree-two scalar residual remains nonzero. The
fixture therefore isolates decoder cancellation from structural/path
cancellation.

### B.5 Generic operator-to-loss equality

Fix the source, transition family, horizon, and decoder-equality partition, and let
\(d=d_{\rm operator}<\infty\). Stack the flattened degree-\(d\) operator
coefficients as rows of a nonzero matrix \(\mathcal G_d\). If

\[
z(q)=\operatorname{vec}(\log q_C),
\]

then the degree-\(d\) scalar coefficient vector is

\[
c_d(q)=-\mathcal G_dz(q).
\]

Choose any nonzero row \(g\) of \(\mathcal G_d\). The scalar function

\[
f(q)=-g^Tz(q)
\]

is real analytic on the product of open decoder simplices. It is not identically
zero: some decoder-class block of \(g\) is nonzero, and varying that class over an
alphabet of size at least two changes \(\log q\) over a positive-dimensional open
set; equivalently, approaching a simplex boundary in a coordinate where \(g\) is
nonzero makes the corresponding log component diverge.

A nonzero real-analytic function on a connected open set has a Lebesgue-measure
zero zero set. Therefore any decoder law with a density on the fixed product of
simplex interiors satisfies

\[
\boxed{
\Pr[d_{\rm loss}>d_{\rm operator}]=0,
\qquad
 d_{\rm loss}=d_{\rm operator}\ \text{a.s.}
}
\]

conditional on finite operator order and the fixed equality partition.

### B.6 Conditional generic support-to-operator equality

Fix a source/transition support pattern and readout-equality partition inside an
irreducible support cell, and let \(d=d_{\rm support}\). The degree-\(d\) quotient
operator entries are analytic in the positive source probabilities, supported
base-transition probabilities, and nonzero perturbation weights (with the source
stationary law analytic while irreducibility is preserved).

If at least one admissible numerical assignment on this fixed support pattern has
a nonzero degree-\(d\) quotient operator, then simultaneous cancellation of all
of those entries is an analytic exceptional set of measure zero. Under this
nondegeneracy condition,

\[
\boxed{d_{\rm operator}=d_{\rm support}\quad\text{a.s.}}
\]

The qualification is necessary because exact signed symmetries can force the
first candidate quotient operator to vanish identically throughout a support
cell.

Combining the two genericity statements gives

\[
\boxed{
d_{\rm support}=d_{\rm operator}=d_{\rm loss}
\quad\text{almost surely}
}
\]

under both stated nondegeneracy conditions.

## Appendix C — Dormant forward equivalence and the same-forward spectrum

### C.1 Source-aware active rows

For a reference controller \(P_0\), row \((m,x)\) is **forward-active through
horizon \(T\)** if at some transition time \(t<T-1\) there exists a source state
\(s\) such that

\[
\Pr_{P_0}(S_t=s,M_t=m)>0
\quad\text{and}\quad
e_s(x)>0.
\]

Otherwise the row is **forward-dormant**.

This definition is source-aware. A memory state may itself be reachable while one
of its symbol-conditioned rows is dormant because that symbol has zero source
probability whenever the memory state co-occurs with the source.

### C.2 Full induction proof

**Theorem C.1 (source-aware dormant forward equivalence).** Let \(P_0\) and
\(\widetilde P_0\) have the same source, reset memory, and horizon. Suppose they
agree on every transition row that is forward-active under \(P_0\). Then

\[
\boxed{
\widetilde\mu_t(s,m)=\mu_t(s,m)
\quad\text{for all }s,m,t<T.
}
\]

Consequently they induce exactly the same finite-horizon prediction process and,
for every fixed decoder, the same finite-horizon loss.

**Proof.** At \(t=0\), both systems use the same source initialization and reset
memory, so their occupancies agree. Assume the occupancies agree at some
\(t<T-1\). Consider any term contributing positive probability to the next
occupancy under the reference process. Such a term starts from a pair \((s,m)\)
with positive current occupancy and uses a symbol \(x\) with \(e_s(x)>0\). Hence
row \((m,x)\) is forward-active under the reference controller. By assumption,
\(P_0(\cdot\mid m,x)=\widetilde P_0(\cdot\mid m,x)\). Since the current
occupancies, source emissions, and deterministic source successors also agree,
every contribution to the next source-memory occupancy is identical. Therefore
\(\widetilde\mu_{t+1}=\mu_{t+1}\). Induction proves equality for the whole
horizon. The prediction distribution at each time is a fixed function of the
source-memory occupancy and decoder, so equality of prediction loss follows. \(\square\)

The theorem proves equality of the entire source-memory process, not merely one
scalar objective value.

### C.3 Why dormant rewiring can change accessibility

The theorem constrains only the current base process. A perturbation direction can
open an entrance into a region that is unreachable at the base. Once that happens,
base transitions stored in the formerly dormant region become usable at zero
additional perturbative cost. Thus two controllers can be exactly
forward-equivalent at \(\varepsilon=0\) but have different mixed
base/perturbative path costs after a counterfactual entrance is opened.

This is the mechanism behind the same-forward accessibility spectrum: dormant
wiring acts as a latent scaffold for a computation that the current predictor does
not yet exercise.

### C.4 Deterministic arbitrary-order family

Take a chain of \(R+1\) memory states. At the base, state 0 is absorbing and all
other states are unreachable. The final state has a predictive readout different
from the reset class. Give each missing prefix edge its own local perturbation
factor, and prewire an arbitrary suffix of the dormant chain in the base
controller. If exactly \(r\) prefix edges remain missing, every current base
controller in the family has the same forward process, while the first
source-valid construction of the final readout uses exactly \(r\) perturbative
factors. The exact fixtures saturate all three orders:

\[
(d_{\rm support},d_{\rm operator},d_{\rm loss})=(r,r,r),
\qquad r=1,\ldots,R.
\]

### C.5 One-class randomized dormant-topology census

The frozen census fixes all forward-facing ingredients and randomizes only dormant
base topology:

- seed: \(20260830\);
- samples: \(1000\);
- binary persistent Markov source with switch probability \(0.1\);
- six controller memory states (depth five);
- horizon \(10\);
- state 0 absorbing and therefore the only currently reachable memory state;
- one fixed five-parameter trainable direction family;
- one fixed decoder with a distinct final-state readout \((0.9,0.1)\);
- in dormant intermediate states 1 through 4, independently install with
  probability \(0.45\) a zero-cost skip edge of strength \(0.5\) to a uniformly
  selected later dormant state; otherwise return to reset.

Every sampled controller is first certified as a dormant rewire of the collapsed
reference controller. The exact construction-order oracle then computes the three
orders. The frozen result is

| exact triple | count |
|---|---:|
| \((1,1,1)\) | 235 |
| \((2,2,2)\) | 282 |
| \((3,3,3)\) | 244 |
| \((4,4,4)\) | 155 |
| \((5,5,5)\) | 84 |

There are \(1000\) certified same-forward controllers and zero hierarchy
violations. The census is breadth evidence inside one exact forward-equivalence
class; it is not presented as a population model for naturally trained networks.

## Appendix D — Visibility order, beneficial descent, and admissible cones

The scalar loss order is a visibility statement. A nonzero leading term need not
point downhill on the valid local parameter cone.

Let

\[
L(x)-L(0)=P_d(x)+P_{d+1}(x)+\cdots,
\]

where each \(P_k\) is homogeneous of degree \(k\), \(P_d\not\equiv0\), and let
\(K\) be the admissible cone of perturbation rays. Define

\[
d_{\rm vis}=d
\]

and, for nonzero \(v\in K\),

\[
d(v)=\min\{k:P_k(v)\ne0\}.
\]

The beneficial ray order is

\[
\boxed{
d_\downarrow
=\inf\{d(v):v\in K\setminus\{0\},\;P_{d(v)}(v)<0\},
}
\]

with value \(\infty\) when no such ray exists.

**Proposition D.1.**

\[
\boxed{d_\downarrow\ge d_{\rm vis}.}
\]

Moreover,

\[
\boxed{
d_\downarrow=d_{\rm vis}
\iff
\exists v\in K:\;P_{d_{\rm vis}}(v)<0.
}
\]

**Proof.** Along any admissible ray,

\[
L(tv)-L(0)
=t^{d(v)}P_{d(v)}(v)+o(t^{d(v)}).
\]

No homogeneous term exists below degree \(d_{\rm vis}\), so
\(d(v)\ge d_{\rm vis}\) for every ray. A sufficiently small descent occurs at
order \(d_{\rm vis}\) exactly when the degree-leading form is negative on at least
one admissible ray. \(\square\)

The distinction is necessary even in two variables. On
\(K=\mathbb R_+^2\),

\[
L(x,y)-L(0)=x-y^2
\]

has \(d_{\rm vis}=1\), but its linear term is harmful throughout the positive
cone. The ray \((0,1)\) first descends at degree two, so

\[
\boxed{d_{\rm vis}=1,\qquad d_\downarrow=2.}
\]

For the manuscript's isolated construction-time fixtures,

\[
L-L_0=-C\prod_{i=1}^d x_i,
\qquad C>0,
\]

on the positive construction cone. Every interior positive ray has negative
leading coefficient, hence

\[
\boxed{d_\downarrow=d_{\rm loss}=d.}
\]

Accordingly, the paper uses **construction/loss order** for visibility and invokes
**gradient accessibility of a useful computation** only when the relevant
admissible descent condition is also established.

## Remaining appendix blocks

Appendices E–J in `paper/appendix_map.md` remain allocated as follows and will be
ported into the venue-specific supplement after the central proof spine above is
frozen:

- E: isolated affine-coordinate construction dynamics and preconditioning;
- F: regular reparameterizations and rare-edge softmax boundary geometry;
- G: smooth linear state-space validation;
- H: shared-route conservation geometry;
- I: near-degeneracy failures of leading-order prediction;
- J: earlier diagnostic/oracle results.

The central correctness claims of the paper do not depend on H–J; those sections
provide robustness, boundary cases, and research provenance.
