# Paper-level proof exposition for the central construction theorems

This note rewrites the core results without implementation vocabulary. It is meant
as a proof skeleton for the eventual paper, with assumptions stated explicitly and
with the three distinct notions of order kept separate.

The central chain is

\[
\boxed{
 d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}
}
\]

followed by a source-aware forward-equivalence theorem showing that dormant
controller topology can change these orders while leaving the current process
unchanged.

## 1. Finite-memory prediction model

Let

- \(\mathcal S\) be a finite source-state set;
- \(\mathcal X\) be a finite alphabet;
- \(e_s(x)=\Pr(X_t=x\mid S_t=s)\) be the emission probabilities;
- \(f(s,x)\in\mathcal S\) be a deterministic/unifilar source update;
- \(\pi\) be the stationary source distribution;
- \(\mathcal M=\{1,\ldots,K\}\) be the controller-memory states;
- \(m_0\) be the reset memory state;
- \(q_m(x)>0\) be a fixed decoder distribution for memory state \(m\).

The controller belongs to an affine transition family

\[
P_\varepsilon(m'\mid m,x)
=
P_0(m'\mid m,x)
+
\sum_{j=1}^n \varepsilon_j D_j(m'\mid m,x),
\]

where every direction has zero row sum,

\[
\sum_{m'}D_j(m'\mid m,x)=0.
\]

We work either in a sufficiently small parameter neighborhood where the rows are
valid probability distributions or, when discussing coefficient identities, as a
formal polynomial family around \(\varepsilon=0\).

The source is autonomous: controller memory never changes the source dynamics.

For a finite horizon \(T\), define the product occupancy immediately before the
prediction at time \(t\) by

\[
\mu_t^\varepsilon(s,m)
=
\Pr(S_t=s,M_t=m).
\]

Initially,

\[
\mu_0^\varepsilon(s,m)
=
\pi(s)\mathbf 1\{m=m_0\}.
\]

The recurrence is

\[
\mu_{t+1}^\varepsilon(s',m')
=
\sum_{s,m,x}
\mu_t^\varepsilon(s,m)
 e_s(x)
 \mathbf 1\{f(s,x)=s'\}
 P_\varepsilon(m'\mid m,x).
\]

The expected finite-horizon log loss is

\[
L_T(\varepsilon)
=
-\frac1T
\sum_{t=0}^{T-1}
\sum_{s,m,x}
\mu_t^\varepsilon(s,m)e_s(x)\log q_m(x).
\]

Because the transition family is affine and the horizon is finite, this objective
is an exact polynomial in \(\varepsilon\) of total degree at most \(T-1\).

## 2. Occupancy polynomial lemma

Write the linear source-memory propagation operator as

\[
\mathcal B(\varepsilon)
=
\mathcal B_0+
\sum_{j=1}^n\varepsilon_j\mathcal B_j,
\]

where \(\mathcal B_0\) uses \(P_0\) and \(\mathcal B_j\) uses \(D_j\).
Then

\[
\mu_t^\varepsilon
=
\mathcal B(\varepsilon)^t\mu_0.
\]

Expanding the product gives

\[
\boxed{
\mu_t^\varepsilon
=
\sum_{|\alpha|\le t}
\varepsilon^\alpha\mu_{t,\alpha}.
}
\]

Each coefficient \(\mu_{t,\alpha}\) is a signed sum over source-valid product-state
walks of length \(t\) containing exactly \(\alpha_j\) uses of direction
\(D_j\) and all remaining controller-transition factors from \(P_0\).

This walk interpretation is the bridge from finite-memory computation topology to
polynomial derivative order.

### Source-marginal conservation

For every \(\varepsilon\), summing over memory gives the source marginal:

\[
\sum_m\mu_t^\varepsilon(s,m)=\Pr(S_t=s),
\]

which is independent of the controller parameters. Therefore every nonconstant
occupancy coefficient obeys

\[
\boxed{
\sum_m\mu_{t,\alpha}(s,m)=0,
\qquad |\alpha|>0.
}
\]

This cancellation is important in the support lower-bound proof below.

## 3. Decoder quotient and exact coefficient factorization

Define an equivalence relation on memory states by equality of decoder rows:

\[
m\sim m'
\quad\Longleftrightarrow\quad
q_m=q_{m'}.
\]

Let \(\mathcal C\) be the resulting decoder classes and write \(q_C\) for the
common decoder row of class \(C\).

For every multi-index \(\alpha\), define the quotient construction operator
coefficient

\[
G_\alpha(C,x)
=
\frac1T
\sum_{t=0}^{T-1}
\sum_{s}
\sum_{m\in C}
\mu_{t,\alpha}(s,m)e_s(x).
\]

Substitution into the loss gives the exact coefficient identity

\[
\boxed{
[\varepsilon^\alpha]L_T
=
c_\alpha
=
-\sum_{C\in\mathcal C}
\sum_{x\in\mathcal X}
G_\alpha(C,x)\log q_C(x).
}
\]

Equivalently,

\[
\boxed{c_\alpha=-\langle G_\alpha,\log q\rangle.}
\]

This is the main algebraic separation:

- the source and transition construction geometry determine \(G_\alpha\);
- equality of decoder rows determines the quotient classes;
- the numerical decoder values determine whether a nonzero operator coefficient
  is visible or cancelled in the scalar loss.

## 4. Three orders

### 4.1 Support construction order

Build the source-memory product graph. From \((s,m)\), symbol \(x\) may be used
only when

\[
e_s(x)>0.
\]

The source successor is \(f(s,x)\). A controller transition to \(m'\) has:

- cost zero when \(P_0(m'\mid m,x)>0\);
- cost one when at least one perturbation direction satisfies
  \(D_j(m'\mid m,x)\ne0\).

The same physical edge can have both kinds of contribution; the walk expansion
remembers whether the base or perturbative factor was selected.

Let \(C_0\) be the decoder class containing the reset memory \(m_0\). Define
\(d_{\rm support}\) as the minimum number of perturbative factors on any
source-valid product-state walk, within the finite horizon, that reaches a memory
state outside \(C_0\).

If no such walk exists, set \(d_{\rm support}=\infty\).

The interesting construction-origin regime has \(d_{\rm support}\ge1\).

### 4.2 Operator order

Define

\[
d_{\rm operator}
=
\min\{|\alpha|>0:G_\alpha\ne0\},
\]

with value \(\infty\) if every nonconstant quotient operator coefficient vanishes.

### 4.3 Scalar loss order

Define

\[
d_{\rm loss}
=
\min\{|\alpha|>0:c_\alpha\ne0\},
\]

again using \(\infty\) if the finite-horizon objective is locally constant.

These are different objects. The point of the theorem is not merely to name them,
but to identify the exact mechanisms that can separate them.

## 5. The construction-order sandwich

### Theorem 1

Assume \(d_{\rm support}\ge1\). Then

\[
\boxed{
 d_{\rm support}
 \le
 d_{\rm operator}
 \le
 d_{\rm loss}.
}
\]

### Proof of \(d_{\rm support}\le d_{\rm operator}\)

Fix a nonzero multi-index \(\alpha\) with

\[
|\alpha|<d_{\rm support}.
\]

Every walk contributing to \(\mu_{t,\alpha}\) uses exactly \(|\alpha|\)
perturbative transition factors. By definition of \(d_{\rm support}\), no such
source-valid walk can reach a decoder class different from \(C_0\). Hence

\[
\mu_{t,\alpha}(s,m)=0
\quad\text{for every }m\notin C_0.
\]

For the remaining class, source-marginal conservation gives

\[
\sum_{m\in C_0}\mu_{t,\alpha}(s,m)
=
\sum_m\mu_{t,\alpha}(s,m)
=0.
\]

Therefore

\[
G_\alpha(C,x)=0
\quad\text{for every }C,x.
\]

So no quotient operator coefficient can appear below total degree
\(d_{\rm support}\), which proves

\[
d_{\rm operator}\ge d_{\rm support}.
\]

### Proof of \(d_{\rm operator}\le d_{\rm loss}\)

The coefficient factorization gives

\[
c_\alpha=-\langle G_\alpha,\log q\rangle.
\]

Thus

\[
G_\alpha=0\Longrightarrow c_\alpha=0.
\]

Every scalar coefficient below \(d_{\rm operator}\) therefore vanishes, so

\[
d_{\rm loss}\ge d_{\rm operator}.
\]

This completes the proof.

## 6. Meaning of strict inequalities

The sandwich separates two qualitatively different cancellation mechanisms.

### 6.1 Support-to-operator gap

If

\[
d_{\rm operator}>d_{\rm support},
\]

then source-valid walks of the minimal perturbative cost do exist, but their signed
occupancy contributions cancel after aggregation into decoder classes. This is a
**construction/path cancellation**.

Graph distance alone cannot detect it; the quotient operator can.

### 6.2 Operator-to-loss gap

If

\[
d_{\rm loss}>d_{\rm operator},
\]

then the first nonzero construction operator exists, but the decoder log-vector
lies in its annihilator:

\[
\langle G_\alpha,\log q\rangle=0
\quad\text{for every }|\alpha|=d_{\rm operator}.
\]

This is **decoder-value cancellation**.

The repository's neutral-decoder fixture realizes the strict triple

\[
\boxed{(d_{\rm support},d_{\rm operator},d_{\rm loss})=(1,1,2).}
\]

That witness is why the operator level is mathematically necessary rather than a
notational convenience.

## 7. Generic decoder visibility

Fix the source, affine transition family, horizon, and decoder-equality partition.
Suppose

\[
d=d_{\rm operator}<\infty.
\]

Stack the nonzero degree-\(d\) operators into a matrix \(\mathcal G_d\). The vector
of all degree-\(d\) scalar coefficients is

\[
c_d(q)=-\mathcal G_d\operatorname{vec}(\log q_C).
\]

### Proposition 2

Assume \(|\mathcal X|\ge2\), and sample each decoder class from a distribution with
a density on the interior of its probability simplex while preserving the fixed
equality partition. Then

\[
\boxed{
\Pr(d_{\rm loss}>d_{\rm operator})=0.
}
\]

Hence

\[
\boxed{
d_{\rm loss}=d_{\rm operator}\quad\text{almost surely}.}
\]

### Proof

Because \(\mathcal G_d\ne0\), choose a nonzero row \(g\). Consider

\[
f(q)=g^T\operatorname{vec}(\log q_C).
\]

This is real analytic on the product of open decoder simplices.

It is not identically zero. To see this, choose a decoder class whose block
\(g_C\) is nonzero. If \(g_C^T\log q_C\) were constant throughout that simplex,
then every tangent derivative would vanish. In coordinates
\(q_r=1-\sum_{i<r}q_i\), this would require

\[
\frac{g_i}{q_i}-\frac{g_r}{q_r}=0
\]

for every interior \(q\) and every \(i<r\), which is possible only when the whole
block \(g_C\) is zero, a contradiction.

Therefore \(f\) is a nonzero real-analytic function. Its zero set has Lebesgue
measure zero. Simultaneous cancellation of *all* degree-\(d\) coefficients is a
subset of that zero set, proving the claim.

### Conditional support genericity

An analogous statement applies to the first inequality, but it requires an
explicit nondegeneracy assumption.

Fix a source/transition support cell and decoder-equality partition. Inside an
irreducible positive-probability support cell, the candidate degree
\(d_{\rm support}\) quotient operator is analytic in the numerical source/base
transition probabilities and perturbation weights. If it is nonzero for at least
one admissible assignment on that support cell, then its simultaneous zero set is
an analytic exceptional set of measure zero. Under that assumption,

\[
d_{\rm operator}=d_{\rm support}
\quad\text{almost surely}.
\]

The qualification matters because an exact signed symmetry may force the first
candidate operator to vanish identically on an entire support cell.

## 8. Source-aware dormant rows

The second main theorem concerns exact forward equivalence rather than local
Taylor coefficients.

For a fixed reference controller \(P_0\), call the transition row \((m,x)\)
**forward-active through horizon \(T\)** when there exists a time
\(t\in\{0,\ldots,T-2\}\) and a source state \(s\) such that

\[
\mu_t^0(s,m)>0
\quad\text{and}\quad
e_s(x)>0.
\]

Otherwise the row is **forward-dormant**.

This definition is source-aware. A symbol-conditioned row of a reachable memory
state can still be dormant when that symbol is impossible on every source state
that co-occurs with the memory state.

## 9. Dormant forward-equivalence theorem

### Theorem 3

Let \(P_0\) and \(\widetilde P_0\) be two controller transition tables with the
same source, reset memory, and horizon. Suppose they agree on every row that is
forward-active under \(P_0\):

\[
\widetilde P_0(\cdot\mid m,x)
=P_0(\cdot\mid m,x)
\quad\text{for every forward-active }(m,x).
\]

Then their source-memory occupancies are identical throughout the horizon:

\[
\boxed{
\widetilde\mu_t(s,m)=\mu_t(s,m),
\qquad t=0,\ldots,T-1.
}
\]

Consequently, for **every fixed decoder** \(q\), the two systems have identical
finite-horizon expected prediction loss.

### Proof

At \(t=0\), both occupancies equal

\[
\pi(s)\mathbf1\{m=m_0\}.
\]

Assume inductively that

\[
\widetilde\mu_t=\mu_t.
\]

Consider any term that can contribute positive probability to the next occupancy.
It must have

\[
\mu_t(s,m)>0
\quad\text{and}\quad
e_s(x)>0.
\]

Therefore \((m,x)\) is forward-active under the reference process, and the two
controllers use the same transition row on that term. All terms in the occupancy
recurrence are consequently identical, so

\[
\widetilde\mu_{t+1}=\mu_{t+1}.
\]

Induction proves occupancy equality through the horizon. Substitution in the loss
formula gives equality for every decoder.

### Important asymmetry in the definition

Dormancy is defined using the reference process only. This is sufficient: the
induction shows the candidate process never acquires probability on a new row
before the reference does, because every row on the shared support uses the same
transition law.

## 10. Why dormant topology can change construction order

The forward-equivalence theorem applies only to the **base point**. The affine
perturbation family may contain a direction that enters a previously dormant
region. Once such a perturbative factor is selected in the coefficient expansion,
base transitions inside that region become relevant.

Hence rewiring a forward-dormant row can change the number of additional
perturbative factors required to reach a different decoder class.

In graph language:

- the dormant rewire changes only zero-cost edges in a region absent from the base
  forward process;
- a perturbative entrance edge can expose that region;
- the shortest mixed zero-cost/unit-cost construction path can therefore change;
- \(d_{\rm support}\), and generically \(d_{\rm loss}\), can change even though
  the entire current occupancy process is unchanged.

This is the exact sense in which

\[
\boxed{
\text{forward equivalence does not imply accessibility equivalence.}
}
\]

## 11. Arbitrary-order forward-equivalent family

For every integer \(R\ge1\), take memory states

\[
0,1,\ldots,R,
\]

with reset state zero and a decoder whose final state has a different predictive
row from the collapsed states.

At the fully collapsed base controller, all rows return to state zero. Introduce a
chain of candidate construction links

\[
0\to1\to2\to\cdots\to R.
\]

For a chosen \(r\in\{1,\ldots,R\}\):

1. leave the first \(r\) links absent and available as perturbative directions;
2. prewire links \(r+1,\ldots,R\) as base transitions;
3. keep the entrance link absent, so every prewired downstream row remains
   forward-dormant at the base point.

The resulting base controller is forward-equivalent to the fully collapsed one.
Its shortest construction path to the final decoder class uses exactly \(r\)
perturbative factors, so

\[
d_{\rm support}=r.
\]

For the delayed-repeat source/readout family used in the exact witness, the
corresponding degree-\(r\) quotient operator and scalar coefficient are nonzero,
giving

\[
\boxed{
(d_{\rm support},d_{\rm operator},d_{\rm loss})=(r,r,r).
}
\]

Thus one current forward-equivalence class can contain controllers whose useful
computation appears at every order

\[
\boxed{1,2,\ldots,R.}
\]

The deterministic prefix construction varies which links are treated as missing
perturbative directions. A separate 1,000-instance census is stronger on a
different axis: it fixes the source, architecture, decoder, reachable base
behavior, **and the trainable direction family**, and varies only dormant zero-cost
wiring; orders one through five still all occur.

## 12. Connection to construction time

Suppose the first beneficial local contribution is an isolated homogeneous
monomial of total degree \(d\), and consider the balanced positive ray. Then the
leading gradient flow has the scalar form

\[
\dot s=C s^{d-1},\qquad C>0.
\]

The exact threshold times have three asymptotic classes:

\[
\boxed{
\tau_d(\delta)
\sim
\begin{cases}
O(1), & d=1,\\
O(\log(1/\delta)), & d=2,\\
\Theta(\delta^{-(d-2)}), & d\ge3.
\end{cases}}
\]

This homogeneous-flow mathematics has strong prior art. The finite-memory theorem
contributes the structural map that tells us **which degree \(d\)** a latent
computation has, and the dormant-equivalence theorem shows that \(d\) can change
inside one exact current forward-equivalence class.

The conceptual consequence is therefore stronger than a constant-factor gradient
change: dormant topology can move an otherwise identical current predictor between
finite, logarithmic, and polynomial bootstrap regimes.

## 13. Regular-coordinate robustness

Let a scalar loss germ satisfy

\[
F(\varepsilon)-F(0)
=P_d(\varepsilon)+O(\|\varepsilon\|^{d+1}),
\qquad P_d\not\equiv0.
\]

For a smooth local change of coordinates

\[
\varepsilon=\phi(\theta),
\qquad
D\phi(0)=J,
\]

with \(J\) invertible,

\[
F(\phi(\theta))-F(0)
=P_d(J\theta)+O(\|\theta\|^{d+1}).
\]

Because an invertible linear map cannot annihilate a nonzero homogeneous
polynomial,

\[
\boxed{
\operatorname{ord}_0(F\circ\phi)=\operatorname{ord}_0F.
}
\]

This is classical function-germ mathematics. Its role is only to close the
coordinate-artifact objection: a construction-order gap between two
forward-equivalent systems cannot be erased by ordinary local diffeomorphisms.

Singular parameterizations are different and can change the order.

## 14. What is theorem and what is evidence

The paper should keep the following hierarchy explicit.

### Exact finite-memory theorems

- occupancy polynomial expansion;
- coefficient factorization through \(G_\alpha\);
- support/operator/loss sandwich;
- decoder genericity under stated assumptions;
- source-aware dormant forward equivalence;
- explicit forward-equivalent arbitrary-order family.

### Exact but auxiliary dynamics

- isolated monomial completion times;
- weighted balance laws;
- positive diagonal conditioning effects;
- shared-route and bilinear leading-mode solutions.

### Adversarial scope-boundary results

- decoder/path cancellations;
- higher-order route near-tie reversals;
- spectral-gap mode reversals;
- finite-step breaking of continuous balance laws.

### External-validity evidence

- random construction-order censuses;
- the single forward-equivalence-class dormant-topology census;
- the independent differentiable linear-state-space validation.

The external validations support the mechanism but are not used to prove the
finite-memory theorem.

## 15. Proof-level claim boundary against prior art

The paper should not claim novelty for:

- same-function parameter points having different Hessian/flatness geometry;
- inactive or dormant neural units at critical embeddings;
- generic high-order vanishing or KL order in singular models;
- staged dormant-feature activation from small initialization;
- balancedness, deep-linear singular modes, or homogeneous escape exponents;
- regular-coordinate invariance of vanishing order.

The narrower theorem-level contribution is:

\[
\boxed{
\begin{array}{c}
\text{source-valid finite-memory construction topology}\\
\Downarrow\\
d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}\\
\Downarrow\\
\text{exact useful derivative order of a latent memory computation},
\end{array}}
\]

plus the intervention theorem that source-aware dormant rewiring can change that
order while preserving the complete current finite-horizon source-memory process.

That is the object the final related-work section must compare directly against
automata/FSC learning, same-function neural embeddings, singular-learning order,
and dormant-feature dynamics.
