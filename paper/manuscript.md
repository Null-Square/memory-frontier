# Same Predictor, Different Learnability: Construction Order in Finite-State Memory

> **Working manuscript draft.** This file is intended to become the main paper,
> not a catalogue of repository results. Proof details and secondary results will
> be moved to appendices during the next pass.

## Abstract

A finite-memory architecture may have enough capacity to implement a predictive
computation while gradient-based learning is locally unable to construct it. We
make this distinction exact for finite-state memory predictors. Around an affine
controller transition family, the finite-horizon prediction loss is polynomial in
local transition strengths. We introduce three notions of **construction order**:
(1) a source-valid support cost, the minimum number of missing transition factors
needed to reach a different predictive readout class; (2) an exact quotient
occupancy-operator order; and (3) the first nonzero degree of the scalar loss. We
prove

\[
d_{\rm support}\le d_{\rm operator}\le d_{\rm loss},
\]

with the two possible gaps corresponding respectively to signed path cancellation
and decoder cancellation. Generically, these orders coincide.

We then prove a source-aware forward-equivalence theorem: changing transition rows
that cannot be exercised by the current source-memory process leaves the complete
finite-horizon occupancy process, and hence the current predictor for every fixed
decoder, unchanged. Nevertheless, such behaviorally dormant rewiring can change
construction order. A single exact forward-equivalence class with fixed source,
architecture, decoder, reachable dynamics, horizon, and trainable transition
directions realizes orders one through five under random dormant topology.

Construction order has a direct dynamical consequence. For an isolated leading
construction of degree \(d\), known homogeneous-flow dynamics give finite bootstrap
time for \(d=1\), logarithmic divergence for \(d=2\), and
\(\Theta(\delta^{-(d-2)})\) divergence for \(d\ge3\) as initialization scale
\(\delta\to0\). Regular local reparameterizations cannot erase the scalar order
gap, and positive diagonal conditioning changes the isolated-flow prefactor but
not this degree-controlled asymptotic class. Finally, a smooth linear
state-space delay model trained with ordinary autograd independently reproduces
loss orders one through five while all compared initializations implement the
same current zero predictor. These results separate predictive capacity and
current behavior from **gradient accessibility**: two models can compute the same
thing now yet expose the same future useful computation to optimization at
radically different local orders.

## 1. Introduction

Memory is usually discussed as a question of representational capacity: how many
states, dimensions, or parameters are required to encode enough history for a
prediction task? But a trainable memory system faces another question. Even if a
useful memory computation lies inside the architecture's capacity, **can local
optimization see how to build it from the current parameter point?**

These questions need not have the same answer.

Consider two recurrent controllers with the same number of memory states, the same
source, the same decoder, and exactly the same trajectories from the reset state.
Suppose their current predictions are therefore identical on every source-valid
history. Standard forward analysis regards the two controllers as equivalent.
Yet they can differ on transition rows that the current process never visits. If
learning later opens an entrance into one of those dormant regions, its internal
wiring determines how many additional missing transitions must be constructed
before a predictive state is reached. The dormant structure is behaviorally
irrelevant now but can determine the local derivative order of a future useful
computation.

This paper formalizes that phenomenon.

### 1.1 Capacity, behavior, and accessibility

We separate three notions:

1. **Predictive capacity:** does the architecture contain a controller that can
   implement the useful memory computation?
2. **Current behavior:** what source-conditioned predictor does the current
   controller actually implement?
3. **Gradient accessibility:** at what local derivative order does a useful
   computation become visible to the training objective around the current
   controller?

The first two are forward notions. The third depends on counterfactual
construction paths in parameter space.

Our central message is

\[
\boxed{\text{forward equivalence does not imply accessibility equivalence}.}
\]

The claim is intentionally narrower than “the same function can have different
optimization geometry,” which is well established for overparameterized neural
network embeddings. It is also narrower than the observation that finite-state
controller gradients can vanish, which has explicit prior art. The object we add
is an exact **local first useful derivative degree** for finite-memory
computations, together with a source-aware structural factorization and an
intervention theorem showing how dormant topology changes that degree without
changing the current forward process.

### 1.2 Contributions

Our main contributions are:

- **Construction-order hierarchy.** We define a source-valid support construction
  cost \(d_{\rm support}\), an exact quotient construction-operator order
  \(d_{\rm operator}\), and the scalar loss order \(d_{\rm loss}\), and prove
  \[
  d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}.
  \]
  Every scalar Taylor coefficient factors as
  \[
  c_\alpha=-\langle G_\alpha,\log q\rangle,
  \]
  which separates construction/path cancellation from decoder cancellation.

- **Dormant forward equivalence.** We prove that rewiring only source/horizon
  forward-dormant controller rows preserves the complete current source-memory
  occupancy process and therefore preserves finite-horizon prediction loss for
  every fixed decoder. Such rewiring can nonetheless change construction order.

- **Same-forward order spectrum.** We construct forward-equivalent controllers
  realizing arbitrary orders \(1,\ldots,R\). In a separate 1,000-instance audit,
  we fix source, architecture, decoder, current reachable dynamics, horizon, and
  the trainable direction family, vary only dormant zero-cost wiring, and recover
  all orders one through five with zero hierarchy violations.

- **Optimization consequence and robustness.** Once an isolated leading
  construction has degree \(d\), standard homogeneous-flow dynamics yield a
  finite/logarithmic/polynomial small-initialization bootstrap hierarchy. We show
  that regular local coordinate changes preserve scalar construction order and
  that positive diagonal conditioning changes only the metric/prefactor of the
  isolated monomial flow, not the degree-controlled class.

- **Independent smooth-memory validation.** A five-state differentiable linear
  delay model trained by ordinary PyTorch autograd exhibits exactly the same
  order spectrum while all base initializations implement the same current
  predictor.

The optimization mathematics for isolated homogeneous monomials, balancedness,
and deep-linear modes is not claimed as new. The finite-memory contribution is
the map from latent computation topology to the local degree to which those known
dynamics apply.

## 2. Finite-memory prediction setup

Let \(\mathcal S\) be a finite source-state set and \(\mathcal X\) a finite
alphabet. The source is unifilar: in source state \(s\), symbol \(x\) is emitted
with probability \(e_s(x)\), and the next source state is a deterministic function
\(f(s,x)\). Let \(\pi\) denote the stationary source distribution.

A controller has finite memory states \(\mathcal M=\{1,\ldots,K\}\), reset state
\(m_0\), transition probabilities \(P(m'\mid m,x)\), and decoder distributions
\(q_m(x)>0\). At each step, the decoder predicts the next source symbol from the
current memory state, after which the observed symbol updates source and memory.

We study a local affine transition family

\[
P_\varepsilon
=
P_0+
\sum_{j=1}^n\varepsilon_jD_j,
\]

where each direction \(D_j\) has zero row sum. The affine strengths
\(\varepsilon_j\) are local construction coordinates. Coefficient identities may
be read formally; when actual stochastic controllers are required we restrict to
a sufficiently small valid neighborhood.

Let

\[
\mu_t^\varepsilon(s,m)=\Pr(S_t=s,M_t=m)
\]

denote the product occupancy before prediction at time \(t\). For horizon \(T\),
expected log loss is

\[
L_T(\varepsilon)
=
-\frac1T\sum_{t=0}^{T-1}
\sum_{s,m,x}
\mu_t^\varepsilon(s,m)e_s(x)\log q_m(x).
\]

Because the controller transition is affine and the horizon is finite,
\(L_T\) is an exact multivariate polynomial of degree at most \(T-1\). Polynomial
finite-horizon FSC objectives have prior art; our focus is not their maximum
degree, but the **lowest nonconstant degree around a specified base controller**.

### 2.1 Occupancy coefficients as source-valid construction walks

Write the one-step product-state propagation operator as

\[
\mathcal B(\varepsilon)
=
\mathcal B_0+
\sum_j\varepsilon_j\mathcal B_j.
\]

Then

\[
\mu_t^\varepsilon
=
\mathcal B(\varepsilon)^t\mu_0
=
\sum_{|\alpha|\le t}
\varepsilon^\alpha\mu_{t,\alpha}.
\]

A coefficient \(\mu_{t,\alpha}\) is a signed sum over source-valid product-state
walks containing exactly \(\alpha_j\) uses of transition direction \(D_j\), with
all other controller factors supplied by the base transition \(P_0\).

The source marginal is independent of the controller. Consequently every
nonconstant occupancy coefficient satisfies

\[
\sum_m\mu_{t,\alpha}(s,m)=0,
\qquad |\alpha|>0.
\]

This conservation identity is what turns a combinatorial path lower bound into a
loss-order lower bound.

## 3. Construction order

### 3.1 Quotienting predictive equivalence

Memory states with identical decoder rows are equivalent for the current
prediction objective. Define

\[
m\sim m'
\quad\Longleftrightarrow\quad
q_m=q_{m'}.
\]

Let \(\mathcal C\) be the resulting readout classes and \(q_C\) the decoder row
of class \(C\). For each multi-index \(\alpha\), aggregate occupancy coefficients
within classes:

\[
G_\alpha(C,x)
=
\frac1T\sum_{t=0}^{T-1}
\sum_s\sum_{m\in C}
\mu_{t,\alpha}(s,m)e_s(x).
\]

Substitution into the loss gives the exact factorization

\[
\boxed{
[\varepsilon^\alpha]L_T
=
c_\alpha
=-\sum_{C,x}G_\alpha(C,x)\log q_C(x)
=-\langle G_\alpha,\log q\rangle.
}
\]

This separates source/transition construction geometry from the numerical decoder
values.

### 3.2 Three orders

Let \(C_0\) be the readout class containing the reset state.

**Support construction order.** Form the source-memory product graph. A
source-valid controller transition supplied by \(P_0\) has cost zero; a transition
supplied by any perturbation direction has cost one. Define
\(d_{\rm support}\) as the minimum number of perturbative factors on a source-valid
walk, within the horizon, that reaches a memory state outside \(C_0\). We focus on
the construction-origin regime \(d_{\rm support}\ge1\).

**Operator order.** Define

\[
d_{\rm operator}
=
\min\{|\alpha|>0:G_\alpha\ne0\}.
\]

**Loss order.** Define

\[
d_{\rm loss}
=
\min\{|\alpha|>0:c_\alpha\ne0\}.
\]

The three quantities answer different questions: whether a useful predictive
class is structurally reachable with a given number of missing factors; whether
the signed source-weighted occupancy effect survives aggregation; and whether the
surviving effect couples to the actual decoder.

### 3.3 Main theorem

**Theorem 1 (construction-order hierarchy).** In the construction-origin regime,

\[
\boxed{
d_{\rm support}
\le d_{\rm operator}
\le d_{\rm loss}.
}
\]

**Proof sketch.** If \(|\alpha|<d_{\rm support}\), no source-valid walk using that
many perturbative factors can leave the reset decoder class. Therefore all
coefficient occupancy outside \(C_0\) is zero. Inside \(C_0\), source-marginal
conservation forces the aggregate nonconstant occupancy coefficient to sum to
zero. Hence \(G_\alpha=0\), proving the first inequality. The second follows
immediately from
\(c_\alpha=-\langle G_\alpha,\log q\rangle\): if the operator coefficient is zero,
so is the scalar coefficient. Full proofs are in the appendix.

### 3.4 Why the inequalities can be strict

The hierarchy exposes two distinct failure modes.

If

\[
d_{\rm operator}>d_{\rm support},
\]

minimal-cost source-valid construction walks exist, but their signed contributions
cancel after quotient aggregation. This is **path/operator cancellation**.

If

\[
d_{\rm loss}>d_{\rm operator},
\]

the construction operator exists but its degree-leading image is annihilated by
the decoder log-vector. This is **decoder cancellation**.

An explicit exact fixture realizes

\[
(d_{\rm support},d_{\rm operator},d_{\rm loss})=(1,1,2),
\]

showing that the intermediate operator level is necessary.

### 3.5 Generic equality

Fix the source, transition family, horizon, and decoder equality partition, and
suppose \(d=d_{\rm operator}<\infty\). The degree-\(d\) coefficient vector is a
nonzero real-analytic function of the decoder probabilities through
\(\log q\). Because a nonzero real-analytic function has a measure-zero zero set,
continuously sampled decoder values within the fixed partition satisfy

\[
\boxed{d_{\rm loss}=d_{\rm operator}\quad\text{almost surely}.}
\]

A similar result holds for
\(d_{\rm operator}=d_{\rm support}\) inside a fixed irreducible support cell,
conditional on the degree-\(d_{\rm support}\) operator not vanishing identically
by structural symmetry. Thus strict gaps are exact and important, but generic
continuous parameter choices typically saturate the hierarchy.

## 4. Dormant topology: same forward process, different accessibility

The preceding theorem analyzes a local transition family. We next show that the
base controller itself can be changed in ways that are **exactly invisible to the
current forward process** yet alter the construction geometry seen after a
perturbative entrance.

### 4.1 Source-aware dormant rows

For reference controller \(P_0\), call transition row \((m,x)\)
**forward-active through horizon \(T\)** if for some transition step there exists
a source state \(s\) such that the current product process has positive occupancy
at \((s,m)\) and \(e_s(x)>0\). A row is forward-dormant otherwise.

This definition is source-aware. Even a symbol-conditioned row of a reachable
memory state can be dormant if the relevant symbol is impossible whenever that
memory state co-occurs with the source.

### 4.2 Forward-equivalence theorem

**Theorem 2 (source-aware dormant forward equivalence).** Let \(P_0\) and
\(\widetilde P_0\) have the same source, reset memory, and horizon. If they agree
on every row forward-active under \(P_0\), then

\[
\boxed{
\widetilde\mu_t(s,m)=\mu_t(s,m)
\quad\text{for every }s,m,t<T.
}
\]

Therefore, for every fixed decoder, they have exactly the same finite-horizon
prediction loss.

**Proof sketch.** Both processes start from the same occupancy. Assume their
occupancies agree at time \(t\). Any term contributing positive probability to
\(\mu_{t+1}\) must use a row that is forward-active under the reference process,
so both controllers use the same transition distribution on that term. Thus the
next occupancies agree. Induction proves the result.

The theorem is stronger than equality of one scalar loss: the whole
source-memory process is identical.

### 4.3 How dormant rewiring changes construction order

The forward-equivalence theorem concerns the base process only. A perturbative
transition direction may enter a region that is dormant at the base point. Once
that happens, zero-cost base wiring inside the dormant region becomes available
to the construction walk.

Thus a dormant rewire can alter the shortest mixed path

\[
\text{perturbative entrance}
\;\to\;
\text{dormant zero-cost scaffold}
\;\to\;
\text{predictive readout class}
\]

without altering the current process at all.

For every depth \(R\), a delay-chain construction gives forward-equivalent base
controllers with useful orders

\[
\boxed{1,2,\ldots,R.}
\]

The construction is simple: the entrance remains missing, so downstream states
are unreachable at the base point; varying how much downstream chain is prewired
changes how many perturbative links remain necessary after that entrance.

## 5. Randomized audit inside one forward-equivalence class

A hand-built chain could be dismissed as a specially designed witness. We
therefore performed a stricter randomized audit in which the only varying object
is dormant topology.

The experiment fixes:

- one source;
- one six-state controller architecture;
- one decoder;
- one current reachable collapsed controller;
- one finite horizon;
- the same five trainable perturbation directions.

It then randomizes only zero-cost transition wiring on rows certified to be
forward-dormant. Every sampled controller therefore belongs to the same exact
current forward-equivalence class by Theorem 2.

Across 1,000 samples, the exact construction-order oracle obtained:

| exact triple | count |
|---|---:|
| \((1,1,1)\) | 235 |
| \((2,2,2)\) | 282 |
| \((3,3,3)\) | 244 |
| \((4,4,4)\) | 155 |
| \((5,5,5)\) | 84 |

There were zero hierarchy violations.

This experiment is not a proof of genericity; the analytic genericity result is
separate. Its role is adversarial: it shows that a broad accessibility spectrum
does not require redesigning the current predictor, source, decoder, or trainable
directions. Random dormant graph structure alone is sufficient.

## 6. From construction order to bootstrap time

Construction order matters because a degree-\(d\) useful term generates a
small-initialization bottleneck.

Consider an isolated beneficial square-free monomial

\[
L-L_0=-C\prod_{j=1}^d\varepsilon_j,
\qquad C>0,
\]

and equal positive initialization \(\varepsilon_j(0)=\delta\). Symmetry reduces
leading gradient flow to

\[
\dot x=Cx^{d-1}.
\]

The exact time to reach a fixed threshold \(\theta>\delta\) is

\[
\tau_1=\frac{\theta-\delta}{C},
\]

\[
\tau_2=\frac1C\log\frac{\theta}{\delta},
\]

and, for \(d\ge3\),

\[
\tau_d
=
\frac{\delta^{2-d}-\theta^{2-d}}{C(d-2)}.
\]

Hence

\[
\boxed{
\tau_d(\delta)
\sim
\begin{cases}
O(1),&d=1,\\
O(\log(1/\delta)),&d=2,\\
\Theta(\delta^{-(d-2)}),&d\ge3.
\end{cases}}
\]

The homogeneous escape law itself has prior art. Our contribution is the
structural route from a latent finite-memory computation to the degree \(d\).
Together with Theorem 2, this implies that behaviorally dormant rewiring can move
an unchanged current predictor between finite, logarithmic, and polynomial
bootstrap regimes.

At exact zero initialization, the distinction is sharper: an isolated first-order
construction moves immediately, whereas an isolated degree-two-or-higher
construction is exactly stationary until some other mechanism perturbs it.

## 7. Robustness and scope

### 7.1 Regular local coordinates preserve scalar order

One possible objection is that construction order is merely a coordinate
artifact. Let

\[
F(\varepsilon)-F(0)
=P_d(\varepsilon)+O(\|\varepsilon\|^{d+1}),
\qquad P_d\not\equiv0,
\]

and let \(\varepsilon=\phi(\theta)\) be a smooth local coordinate change with
invertible Jacobian \(J=D\phi(0)\). Then

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

This classical fact means that the scalar order gap between two forward-equivalent
controllers cannot be removed by an ordinary local diffeomorphism.

Singular parameterizations are different. Under a coordinatewise power map
\(\varepsilon_i=\theta_i^{r_i}\), the exact pullback degree becomes

\[
\min_{c_\alpha\ne0}\sum_i r_i\alpha_i.
\]

Thus singular charts can genuinely alter the apparent bootstrap order.

### 7.2 Conditioning changes geometry, not the isolated degree class

Ordinary Euclidean gradient flow is parameterization-dependent, so order
invariance does not imply identical trajectories. For an isolated monomial
\(c\prod_i x_i^{\alpha_i}\) under a positive diagonal preconditioner
\(M=\operatorname{diag}(m_i)\), the exact invariants become

\[
\frac{x_i^2}{m_i\alpha_i}
-
\frac{x_j^2}{m_j\alpha_j}
=\text{constant}.
\]

On the corresponding metric-balanced manifold,

\[
\dot s
=(-c)\left[\prod_i(m_i\alpha_i)^{\alpha_i/2}\right]s^{d-1},
\qquad d=\sum_i\alpha_i.
\]

The metric changes the balance geometry and prefactor, but not the degree-driven
finite/logarithmic/polynomial class. We do not claim arbitrary optimizer
invariance; degenerate metrics, adaptive optimizers, finite step size, and
singular parameterizations can change the picture.

### 7.3 Leading order needs a margin

Construction order is a local theory, not a guarantee that the lowest-order route
wins every later nonlinear competition. If the leading useful degree is \(d\) and
the first route-distinguishing correction is degree \(p>d\), then on the local
construction scale the normalized correction is of order

\[
\delta^{p-d}.
\]

Therefore route or mode labels predicted by the leading term need a margin larger
than this scale. We constructed exact near-tie finite-memory examples in which
higher-order terms reverse both a scalar route winner and a nearly degenerate
bilinear construction mode. We use these examples as scope boundaries, not as
additional headline theorems.

## 8. Independent validation in a smooth linear memory model

The finite-state theorem is exact but specialized. To test whether the mechanism
survives outside finite stochastic controllers, we use a minimal differentiable
linear state-space delay line:

\[
h_{t+1,0}=w_0x_t,
\qquad
h_{t+1,i}=w_i h_{t,i-1},
\quad i=1,\ldots,4.
\]

The final state predicts the input from five updates earlier. For binary
unit-variance inputs, its effective delayed-memory gain is

\[
g=\prod_{i=0}^4w_i,
\]

and mean squared prediction loss is exactly

\[
L(w)=\frac12(g-1)^2.
\]

For each \(d\in\{1,\ldots,5\}\), initialize the first \(d\) weights to zero and
all downstream weights to one. Every parameter remains trainable. Because the
first weight is zero in every case, all five initializations implement the
**identical zero predictor on every sequence**. They differ only in how much of
the downstream multiplicative path is already present but behaviorally dormant.

Along the equal missing-link ray

\[
w_0=\cdots=w_{d-1}=\delta,
\qquad
w_d=\cdots=w_4=1,
\]

we obtain exactly

\[
L(0)-L(\delta)
=
\delta^d-\frac12\delta^{2d},
\]

and ordinary autograd gives

\[
\|\nabla_{w_{<d}}L\|
=
\sqrt d(1-\delta^d)\delta^{d-1}.
\]

Thus

\[
L(0)-L(\delta)=\Theta(\delta^d),
\qquad
\|\nabla_{w_{<d}}L\|=\Theta(\delta^{d-1}).
\]

Direct recurrent PyTorch regressions recover loss slopes one through five and
gradient slopes zero through four. At exact zero, only the first-order case has a
nonzero missing-link gradient.

As an optimizer sanity check, ordinary SGD with missing links initialized to
\(0.05\), prewired links to one, learning rate \(0.05\), and delayed-gain threshold
\(0.2\) crossed the threshold after approximately

\[
4,\;43,\;361,\;3965,\;53327
\]

steps for orders one through five. These iteration counts are deliberately not
presented as universal constants. Their purpose is to show that the local order
separation persists under an ordinary finite-step optimizer in a conventional
smooth recurrent system.

The multiplication-chain mathematics is elementary and closely related to
known deep-linear dynamics. The validation contributes breadth, not a new theorem.

## 9. Related work

### Finite-state-controller policy gradients

Gradient-based learning of finite-state controllers for POMDPs predates this work.
Aberdeen and Baxter (ICML 2002) explicitly analyze zero-gradient regions for
internal-state policies: near-uniform internal transition and action policies can
produce near-zero memory gradients, and under a stated symmetry condition the
internal-state gradient vanishes exactly. They also advocate sparse controller
transition graphs to avoid these regions. We therefore do not claim the discovery
of zero FSC memory gradients or the usefulness of sparse internal-state graphs.

Boularias and Chaib-draa (ICML 2009) establish a still closer algebraic precedent:
finite-horizon FSC and PSR values can be polynomial in policy parameters, and the
representation can alter the polynomial's degree. For a generic fully connected
FSC they derive high history-probability degree and discuss reducing FSC outdegree
to reduce degree. Their analysis concerns the degree of the complete value
polynomial. Our construction order instead concerns the **minimum nonconstant
local degree at a specified base point**, together with the source-valid
support/operator/loss decomposition and a dormant forward-equivalence
intervention.

Braziunas and Boutilier (AAAI 2004) identify sequential pathologies of gradient
search for POMDP controllers and motivate stochastic local search. This is
conceptually related to multistep constructions that are locally invisible, but
their work does not provide our local derivative-order factorization.

### Same-function embeddings and singular geometry

Fukumizu et al. (NeurIPS 2019) and subsequent neural-network embedding work show
that wider networks can realize the same function as narrower networks through
replicated or inactive units while changing flat/minimum/saddle geometry. Thus
“same function, different optimization landscape” is established prior art. Our
intervention is narrower: within a fixed finite-memory architecture and a fixed
source-conditioned current process, rewiring only dormant transition rows changes
an exact computation-specific local degree.

Recent singular-learning work on “dead directions” assigns KL vanishing orders to
Fisher-degenerate directions. We similarly use an order of vanishing, but derive
it from source-valid finite-memory computation paths and quotient occupancy
operators rather than treating arbitrary singular parameter directions.

### Dormant-feature and homogeneous dynamics

Alternating Gradient Flows and related small-initialization work describe staged
activation of dormant neural features and predict feature-acquisition timing.
Deep-linear/homogeneous analyses provide balancedness, singular-mode dynamics, and
small-initialization saddle escape laws. These results motivate and support our
dynamic interpretation, but our main theorem is upstream of those dynamics: it
identifies the first useful local degree of a particular memory computation from
its construction topology.

## 10. Limitations

Our theorems deliberately concern finite-horizon finite-state memory controllers
with affine local transition coordinates and fixed decoder equality classes.
Several boundaries matter.

First, construction order is a local quantity. Higher-order terms can determine
which route wins when leading margins are small, and we give explicit reversal
examples. Second, scalar loss order is invariant under regular local charts but
not under singular parameterizations; optimizer trajectories are metric- and
step-size-dependent even under regular charts. Third, decoder co-training can
introduce additional required factors depending on the decoder parameterization.
Fourth, source-aware support depends on the stated horizon and source support:
changing either can change which rows are behaviorally dormant or which paths are
valid.

The linear state-space experiment shows the mechanism outside discrete
controllers, but it is intentionally minimal. It does not establish that
construction order is the dominant explanation of training behavior in large
RNNs, modern state-space sequence models, or transformers. Testing that broader
hypothesis is future work.

Finally, novelty claims are intentionally narrow. Prior work already establishes
zero gradients in symmetric FSCs, graph/representation effects on global
polynomial degree, same-function neural embeddings with different local geometry,
higher-order singular directions, dormant feature activation, and homogeneous
escape dynamics. Our claim is the finite-memory structural decomposition and
dormant-rewire intervention described above.

## 11. Discussion

The common language of memory capacity can hide an optimization distinction.
Two models may have the same number of states and the same current predictor, and
even agree on every source-valid trajectory from reset, yet differ in how many
missing computational factors must be jointly constructed before a useful
predictive state becomes visible.

The construction-order hierarchy makes that distinction explicit:

\[
\text{source-valid topology}
\;\longrightarrow\;
 d_{\rm support}
\;\longrightarrow\;
 d_{\rm operator}
\;\longrightarrow\;
 d_{\rm loss}.
\]

The first arrow says which latent computation paths exist. The second tests
whether signed path effects survive aggregation. The third tests whether the
surviving occupancy contrast couples to the current decoder. This hierarchy
separates structural inaccessibility from two kinds of cancellation rather than
collapsing all zero gradients into one phenomenon.

Dormant forward equivalence then supplies the key intervention. We can change the
zero-cost topology that will become available to a future perturbative path while
provably leaving the entire current source-memory process unchanged. In this
sense, initialization can contain **behaviorally invisible computational
scaffolding**. It does not improve today's predictor; it changes how optimization
can build tomorrow's predictor.

The dynamic consequence is potentially substantial. Changing order by one does
not merely multiply a gradient by a constant. At small initialization it can
change a logarithmic bottleneck into a polynomial one, or a polynomial divergence
from \(\delta^{-1}\) to \(\delta^{-2}\), and so on. The exact finite-memory model
lets us isolate this mechanism without confounding it with representation
capacity or current performance.

A broader design question follows: should learned-memory architectures be judged
not only by which computations they can represent, but by the **construction
geometry** they expose around realistic initializations? Dormant scaffolds,
sparsity patterns, state decompositions, or structured initializations may affect
learnability before they affect forward computation. The present work provides an
exact finite-memory setting in which that question can be asked and answered.

## 12. Conclusion

Predictive capacity is not gradient accessibility. In finite-memory predictors,
the first useful derivative degree of a latent computation can be read through a
source-aware hierarchy of support, quotient construction operator, and scalar
loss. Behaviorally dormant rewiring can change this degree without changing the
current source-memory process, and the resulting order determines which known
small-initialization bootstrap regime governs an isolated construction.

The resulting separation is stronger than a vanishing-gradient observation. Two
systems can be exactly forward-equivalent now while requiring qualitatively
different optimization time to build the same future useful computation. A smooth
linear state-space witness shows that the mechanism is not confined to the
finite-controller formalism. These results suggest treating **construction order**
as a distinct axis of learned-memory design, alongside representational capacity
and current predictive performance.

## References to resolve into BibTeX

- Aberdeen, D. and Baxter, J. *Scaling Internal-State Policy-Gradient Methods for
  POMDPs.* ICML, 2002.
- Boularias, A. and Chaib-draa, B. *Predictive Representations for Policy Gradient
  in POMDPs.* ICML, 2009. DOI: 10.1145/1553374.1553383.
- Braziunas, D. and Boutilier, C. *Stochastic Local Search for POMDP Controllers.*
  AAAI, 2004.
- Fukumizu, K., Yamaguchi, S., Mototake, Y., and Tanaka, M. *Semi-flat minima and
  saddle points by embedding neural networks to overparameterization.* NeurIPS,
  2019. arXiv:1906.04868.
- Kunin, D. et al. *Alternating Gradient Flows: A Theory of Feature Learning in
  Two-layer Neural Networks.* 2025. arXiv:2506.06489.
- Shirodkar, T. P. *Dead Directions: Geometric Singular Learning.* 2026.
  arXiv:2606.05957.
- Additional deep-linear, homogeneous-flow, singular-mode, natural-gradient, and
  finite-state-controller references will be resolved from the repository's
  detailed related-work audit before submission.
