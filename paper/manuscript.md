# Same Predictor, Different Learnability: Construction Order in Finite-State Memory

> **Manuscript v2.** This version incorporates the final related-work audit, the
> smooth linear-state-space validation, and the simplex-boundary/softmax analysis.
> Its main conceptual change from v1 is to separate the structural construction
> order from the parameterization-dependent map from local homogeneity to
> optimization time.

## Abstract

A memory architecture can have enough predictive capacity to implement a useful
computation while local optimization is poorly positioned to construct it. We
make this distinction exact for finite-state memory predictors. Around an affine
controller transition family, finite-horizon prediction loss is an exact
multivariate polynomial in local transition strengths. We define three local
construction orders: a source-valid support cost, the first nonzero quotient
occupancy-operator degree, and the first nonconstant scalar-loss degree. In the
construction-origin regime we prove

\[
d_{\rm support}\le d_{\rm operator}\le d_{\rm loss},
\]

and factor every scalar coefficient as

\[
c_\alpha=-\langle G_\alpha,\log q\rangle.
\]

The two possible strict gaps therefore have distinct causes: source-weighted path
cancellation and decoder cancellation. Under stated nondegeneracy conditions,
the three orders coincide almost surely under continuous parameter choices.

We next prove a source-aware forward-equivalence theorem. Rewiring transition
rows that the current source-memory process cannot exercise leaves the complete
finite-horizon occupancy process unchanged, and hence preserves the current
predictor for every fixed decoder. Nevertheless, once a learned entrance makes
those rows reachable, their dormant topology can change construction order. A
single exact forward-equivalence class with fixed source, architecture, decoder,
reachable dynamics, horizon, and trainable transition directions realizes orders
one through five when only dormant zero-cost wiring is randomized.

Construction order determines the leading local homogeneity of a latent
computation, but the conversion of that homogeneity into physical optimization
time depends on the parameterization and metric. For an isolated degree-\(d\)
construction, Euclidean flow in affine transition probabilities gives the known
finite/logarithmic/\(\delta^{-(d-2)}\) hierarchy, while rare-edge Euclidean softmax
flow gives \(\Theta(\delta^{-d})\). In both geometries, reducing construction
order by one removes an asymptotically severe bootstrap factor. Regular local
reparameterizations preserve the scalar order, whereas the simplex boundary is a
singular limit. Finally, an independently simulated five-link linear state-space
memory trained with ordinary PyTorch autograd reproduces loss orders one through
five while all compared base initializations implement exactly the same current
zero predictor. These results separate predictive capacity and current behavior
from **gradient accessibility**: forward-equivalent systems can expose the same
future useful computation to learning at radically different local orders.

## 1. Introduction

Memory is usually framed as a representational question: how many states,
dimensions, or parameters are needed to encode enough history for prediction? A
trainable memory system faces another question. Even when a useful computation is
inside the architecture's capacity, **at what local derivative order does the
training objective reveal how to build it from the current parameter point?**

We call this second property **gradient accessibility**.

Capacity, current behavior, and accessibility are not equivalent notions. Two
controllers can have the same number of states, the same source, the same decoder,
and exactly the same source-conditioned trajectories from their reset state. They
can therefore implement the same current predictor. Yet they may differ on
transition rows that are never exercised by the current process. If learning
later opens an entrance to one of these dormant regions, the pre-existing wiring
inside that region can determine how many additional missing transition factors
must be constructed before a predictive state is reached.

The dormant wiring is irrelevant to what the system computes now, but relevant
to how the system can learn what to compute next.

Our central message is

\[
\boxed{\text{forward equivalence does not imply accessibility equivalence}.}
\]

This statement is intentionally narrower than several known phenomena. Finite-
state-controller gradients can vanish or become very small near symmetric
controllers [Aberdeen & Baxter, 2002]. Finite-horizon FSC objectives can be
polynomial in policy parameters, and controller structure can affect their global
polynomial degree [Boularias & Chaib-draa, 2009]. Overparameterized neural
networks admit same-function embeddings with different local landscape geometry
[Fukumizu et al., 2019]. Dormant neural features can activate sequentially under
small-initialization dynamics [Kunin et al., 2025], and singular-learning work
studies higher-order dead directions [Shirodkar, 2026].

Our object is more specific: the **minimum nonconstant local degree of a particular
latent finite-memory computation around a fixed base controller**, factored into
source-valid construction paths, a quotient occupancy operator, and the decoder.
The accompanying intervention theorem changes only behaviorally dormant topology
inside a fixed current forward-equivalence class.

### 1.1 Contributions

The paper has four central results and two validation/boundary results.

1. **Construction-order hierarchy.** We define source support order
   \(d_{\rm support}\), quotient operator order \(d_{\rm operator}\), and scalar
   loss order \(d_{\rm loss}\), and prove
   \[
   d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}.
   \]
   Every scalar coefficient factors as
   \[
   c_\alpha=-\langle G_\alpha,\log q\rangle,
   \]
   which distinguishes structural/source constraints, path cancellation, and
   decoder cancellation.

2. **Dormant forward equivalence.** If two controllers agree on all
   source/horizon forward-active rows, then their complete source-memory occupancy
   processes agree throughout the horizon. Rewiring only dormant rows therefore
   preserves the current predictor for every fixed decoder.

3. **Same-forward accessibility spectrum.** Dormant rewiring can alter the
   shortest mixed zero-cost/perturbative construction paths once learning makes a
   dormant region reachable. A deterministic family realizes arbitrary orders
   \(1,\ldots,R\) inside one forward-equivalence class. Separately, a
   1,000-instance census fixes the source, architecture, decoder, current
   reachable dynamics, horizon, and trainable direction family, randomizes only
   dormant zero-cost wiring, and obtains orders one through five with zero
   hierarchy violations.

4. **Structural order versus optimization geometry.** Construction order is a
   local homogeneity of the scalar objective and is preserved by smooth local
   coordinate changes with nonsingular Jacobian. The map from order to physical
   optimization time is metric dependent. We give two exact maps for isolated
   constructions: affine probability-coordinate flow and rare-edge softmax flow.

5. **Smooth recurrent validation.** A continuous-state linear delay line trained
   by ordinary autograd reproduces the predicted loss and gradient orders while
   every compared base initialization has the same current predictor.

6. **Scope boundary for competing routes.** Higher-order terms can reverse a
   leading route or singular mode when the leading margin is of the same order as
   the first neglected correction. We use these counterexamples to delimit, not
   enlarge, the main theorem.

## 2. Finite-memory prediction setup

Let \(\mathcal S\) be a finite source-state set and \(\mathcal X\) a finite
alphabet. The source is unifilar: in source state \(s\), symbol \(x\) is emitted
with probability \(e_s(x)\), and the next source state is the deterministic value
\(f(s,x)\). Let \(\pi\) be the stationary source distribution.

A controller has memory states \(\mathcal M=\{1,\ldots,K\}\), reset state
\(m_0\), transition probabilities \(P(m'\mid m,x)\), and decoder distributions
\(q_m(x)>0\). Before each update, the decoder predicts the next source symbol from
the current memory state; the observed symbol then updates source and memory.

We study a local affine transition family

\[
P_\varepsilon=P_0+\sum_{j=1}^n\varepsilon_jD_j,
\]

where every direction \(D_j\) has zero row sum. Coefficient identities can be
read formally; when actual stochastic controllers are needed, restrict to a
sufficiently small valid neighborhood.

Let

\[
\mu_t^\varepsilon(s,m)=\Pr(S_t=s,M_t=m)
\]

be the source-memory occupancy before prediction at time \(t\). The finite-horizon
log loss is

\[
L_T(\varepsilon)
=-\frac1T\sum_{t=0}^{T-1}\sum_{s,m,x}
\mu_t^\varepsilon(s,m)e_s(x)\log q_m(x).
\]

Because the transition family is affine and the horizon is finite,
\(L_T\) is an exact multivariate polynomial of degree at most \(T-1\). This
polynomiality is not our novelty claim. The object of interest is the **lowest
nonconstant degree around the chosen base controller**.

### 2.1 Source-valid construction walks

Write the one-step product-state propagation operator as

\[
\mathcal B(\varepsilon)=\mathcal B_0+\sum_j\varepsilon_j\mathcal B_j.
\]

Then

\[
\mu_t^\varepsilon
=\mathcal B(\varepsilon)^t\mu_0
=\sum_{|\alpha|\le t}\varepsilon^\alpha\mu_{t,\alpha}.
\]

Each coefficient \(\mu_{t,\alpha}\) is a signed sum over source-valid
source-memory walks that use direction \(D_j\) exactly \(\alpha_j\) times and use
base-controller edges for the remaining steps. The source marginal does not
depend on the controller, so every nonconstant occupancy coefficient satisfies

\[
\sum_m\mu_{t,\alpha}(s,m)=0,
\qquad |\alpha|>0.
\]

This conservation identity is what converts a path-cost statement into a
loss-order lower bound.

## 3. Construction order

### 3.1 Quotienting predictive equivalence

Two memory states are readout-equivalent when their decoder rows coincide:

\[
m\sim m'\Longleftrightarrow q_m=q_{m'}.
\]

Let \(\mathcal C\) be the resulting decoder classes. For each multi-index
\(\alpha\), aggregate occupancy coefficients within decoder classes:

\[
G_\alpha(C,x)
=\frac1T\sum_{t=0}^{T-1}\sum_s\sum_{m\in C}
\mu_{t,\alpha}(s,m)e_s(x).
\]

Substitution into the loss gives the exact factorization

\[
\boxed{
[\varepsilon^\alpha]L_T
=c_\alpha
=-\sum_{C,x}G_\alpha(C,x)\log q_C(x)
=-\langle G_\alpha,\log q\rangle.
}
\]

The operator \(G_\alpha\) depends on source statistics and transition construction
geometry; the decoder enters only through \(\log q\).

### 3.2 Three local orders

Let \(C_0\) be the readout class containing the reset memory state.

**Support order.** Assign cost zero to source-valid transitions supplied by the
base controller and cost one to transitions supplied by a perturbation direction.
Define \(d_{\rm support}\) as the minimum perturbative cost of a source-valid walk,
within the horizon, that reaches a memory state outside \(C_0\). We focus on the
construction-origin regime \(d_{\rm support}\ge1\).

**Operator order.**

\[
d_{\rm operator}=\min\{|\alpha|>0:G_\alpha\ne0\}.
\]

**Loss order.**

\[
d_{\rm loss}=\min\{|\alpha|>0:c_\alpha\ne0\}.
\]

These answer three different questions: whether a useful predictive class is
structurally reachable with a given number of missing factors; whether the signed
source-weighted occupancy effect survives aggregation; and whether that effect
couples to the numerical decoder.

### 3.3 Construction-order hierarchy

**Theorem 1.** In the construction-origin regime,

\[
\boxed{
d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}.
}
\]

**Proof sketch.** Fix \(|\alpha|<d_{\rm support}\). No source-valid walk using that
many perturbative factors can reach a decoder class distinct from \(C_0\), so the
corresponding coefficient occupancy vanishes outside \(C_0\). Inside \(C_0\), the
source-marginal conservation identity forces the aggregate nonconstant
coefficient to cancel. Hence \(G_\alpha=0\), proving
\(d_{\rm support}\le d_{\rm operator}\). The second inequality follows directly
from
\(c_\alpha=-\langle G_\alpha,\log q\rangle\).

The inequalities can be strict for two qualitatively different reasons.

- **Path/operator cancellation:** minimal-cost construction walks exist, but their
  signed source-weighted quotient contributions cancel, giving
  \(d_{\rm operator}>d_{\rm support}\).
- **Decoder cancellation:** the first nonzero operator exists, but its image is
  annihilated by \(\log q\), giving
  \(d_{\rm loss}>d_{\rm operator}\).

An exact regression fixture realizes

\[
(d_{\rm support},d_{\rm operator},d_{\rm loss})=(1,1,2),
\]

so the intermediate operator level is necessary.

### 3.4 Generic equality

Fix the source, transition family, horizon, and decoder-equality partition, and
suppose \(d=d_{\rm operator}<\infty\). At degree \(d\), the scalar coefficient
vector is a nonzero real-analytic function of decoder probabilities through
\(\log q\). Therefore continuously sampled decoder values in the interior of the
fixed equality partition satisfy

\[
\boxed{d_{\rm loss}=d_{\rm operator}\quad\text{almost surely}.}
\]

A similar conditional genericity statement holds for
\(d_{\rm operator}=d_{\rm support}\) inside a fixed irreducible support cell,
provided the degree-\(d_{\rm support}\) operator is not forced to vanish
identically by structural symmetry. Under both nondegeneracy conditions,

\[
\boxed{
d_{\rm support}=d_{\rm operator}=d_{\rm loss}
\quad\text{almost surely}.}
\]

The hierarchy remains essential because exact nongeneric cancellations are both
possible and diagnostically interpretable.

## 4. Dormant topology and forward equivalence

The construction hierarchy concerns a local family around a base controller. We
now show that the base controller can itself be modified in ways that are exactly
invisible to the current source-conditioned forward process yet change the
construction geometry after learning opens a new entrance.

### 4.1 Source-aware dormant rows

Call row \((m,x)\) **forward-active through horizon \(T\)** when, at some update
step, the reference product process can occupy memory state \(m\) jointly with a
source state that emits \(x\) with positive probability. Otherwise the row is
**forward-dormant**.

The definition is source-aware. A symbol-conditioned row of a reachable memory
state can be dormant when that symbol is impossible whenever the source and
memory jointly occupy the relevant configuration.

### 4.2 Dormant forward-equivalence theorem

**Theorem 2.** Suppose two controllers have the same source, reset memory, and
horizon, and agree on every transition row that is forward-active under the
reference controller. Then

\[
\boxed{
\widetilde\mu_t(s,m)=\mu_t(s,m)
\quad\text{for every }s,m,t<T.
}
\]

Consequently every fixed decoder produces exactly the same finite-horizon
predictor and the same finite-horizon loss under the two controllers.

**Proof sketch.** Both product processes start from the same occupancy. If their
occupancies agree at step \(t\), any positive-probability contribution to the next
occupancy uses a row that is forward-active under the reference process. The two
controllers agree on that row, so their next occupancies agree. Induction gives
the result.

This theorem is stronger than equality of one scalar loss: the complete
source-memory occupancy process is identical.

### 4.3 Same forward process, different construction order

A perturbative edge can make a previously dormant region reachable. The zero-cost
base wiring already stored inside that region then determines how many additional
perturbative factors are needed to reach a distinct decoder class. Dormant
rewiring can therefore change \(d_{\rm support}\), and generically changes
\(d_{\rm loss}\), without changing the current predictor.

A deterministic delay-chain family realizes orders

\[
\boxed{1,2,\ldots,R}
\]

inside one current forward-equivalence class. This family is an existence result;
its direction set is chosen to realize the requested missing-prefix length.

A stricter 1,000-instance census fixes the source, six-state architecture,
decoder, horizon, current reachable dynamics, and the entire trainable-direction
family. It changes only dormant zero-cost wiring. Every sampled base controller is
certified forward-equivalent to the same collapsed process. The exact order
triples are

| order triple | count |
|---|---:|
| \((1,1,1)\) | 235 |
| \((2,2,2)\) | 282 |
| \((3,3,3)\) | 244 |
| \((4,4,4)\) | 155 |
| \((5,5,5)\) | 84 |

with zero hierarchy violations.

This isolates dormant graph topology as the only varying object and shows that a
broad accessibility spectrum is not a hand-designed prefix-chain artifact.

## 5. From construction order to optimization time

Construction order is a property of the local scalar loss germ. Optimization
time is a property of that loss **plus** a parameterization, metric, initialization
protocol, and optimizer. The distinction matters most at stochastic-simplex
boundaries.

We therefore state the dynamic consequence in two layers.

### 5.1 Structural layer: local homogeneity

Suppose the first beneficial isolated term has square-free degree \(d\):

\[
L-L_0=-C\prod_{i=1}^d p_i+\text{higher-order terms},
\qquad C>0.
\]

The integer \(d\) determines the first local homogeneity of the useful
construction. Dormant scaffolding can reduce this integer while leaving the
current predictor unchanged.

The mapping from this degree to a wall-clock escape law is not intrinsic.

### 5.2 Euclidean flow in affine probability coordinates

For direct Euclidean gradient flow in the independent edge probabilities and a
symmetric initialization \(p_i(0)=\delta\), the leading isolated dynamics reduce
to

\[
\dot p=Cp^{d-1}.
\]

The exact time to a fixed threshold has the familiar classes

\[
\tau_d^{\rm prob}(\delta)
\sim
\begin{cases}
O(1), & d=1,\\
O(\log(1/\delta)), & d=2,\\
\Theta(\delta^{-(d-2)}), & d\ge3.
\end{cases}
\]

These homogeneous-flow exponents are not claimed as new general optimization
mathematics. Their role here is to translate an exactly computed finite-memory
construction order into a bootstrap class in a specified metric.

Positive diagonal conditioning changes the natural weighted-balanced manifold and
the time prefactor, but leaves this isolated probability-coordinate exponent
class unchanged.

### 5.3 Rare-edge softmax flow

The affine construction origin contains exact absent probabilities \(p=0\). A
finite logit cannot represent this point: \(p=0\) corresponds to logit
\(-\infty\). The ordinary smooth-coordinate invariance theorem therefore does not
cover the limiting simplex boundary.

For independent binary logits \(p_i=\sigma(z_i)\), Euclidean logit flow on the
same symmetric isolated degree-\(d\) construction gives

\[
\dot p=Cp^{d+1}(1-p)^2.
\]

The exact completion time from \(\delta\) to fixed \(\theta\in(0,1)\) has a closed
form, with boundary asymptotic

\[
\boxed{
\tau_d^{\rm logit}(\delta,\theta)
=\frac{1}{Cd}\delta^{-d}(1+o(1)).
}
\]

Thus the absolute escape exponent changes under the singular approach to the
simplex boundary.

The exponent is not a binary-sigmoid curiosity. For a full \(K\)-way softmax row
with rare target-edge probability \(p\) and non-target probabilities \(q_a\),

\[
\|\nabla_zp\|^2
=p^2\left[(1-p)^2+\sum_{a\ne *}q_a^2\right].
\]

Because

\[
\frac{K}{K-1}p^2(1-p)^2
\le \|\nabla_zp\|^2
\le 2p^2(1-p)^2,
\]

an isolated symmetric degree-\(d\) construction still has

\[
\dot p=\Theta(p^{d+1}),
\qquad
\tau_d^{K\text{-softmax}}(\delta)=\Theta(\delta^{-d}),
\]

up to row-dependent constants.

Softmax optimization slowness is established prior art; this calculation is not a
claim that softmax can be slow. It is a bridge from the finite-memory construction
degree to a common practical parameterization.

### 5.4 What remains structural

The corrected interpretation is

\[
\boxed{
\text{topology}
\to
\text{construction order/local homogeneity}
\to
\text{parameterization and metric}
\to
\text{optimization time}.
}
\]

Although the absolute time exponent differs between probability and logit
coordinates, the relative scaffold advantage remains asymptotically severe. For
binary logits,

\[
\frac{\tau_d^{\rm logit}}{\tau_{d-1}^{\rm logit}}
\sim \frac{d-1}{d}\delta^{-1}.
\]

Each unit reduction in construction order removes one power of rare-edge
initialization from the bootstrap time.

### 5.5 Regular coordinate changes

Away from singular boundaries, construction order is not an arbitrary coordinate
artifact. If

\[
F(\varepsilon)-F(0)=P_d(\varepsilon)+O(\|\varepsilon\|^{d+1})
\]

and \(\varepsilon=\phi(\theta)\) is smooth with nonsingular
\(D\phi(0)=J\), then

\[
F(\phi(\theta))-F(0)=P_d(J\theta)+O(\|\theta\|^{d+1}),
\]

so

\[
\boxed{\operatorname{ord}_0(F\circ\phi)=\operatorname{ord}_0F.}
\]

The underlying function-germ fact is classical. Its role here is to show that the
forward-equivalent order separation cannot be erased by an ordinary regular local
chart. Singular maps and simplex-boundary limits are different and must be
analyzed with their induced geometry.

## 6. Independent smooth-memory validation

The exact finite-controller theory could still be dismissed as a peculiarity of
stochastic automata. We therefore test the mechanism in a continuous-state linear
recurrent memory trained with ordinary PyTorch autograd.

Consider a five-link scalar delay line

\[
h_{t+1,0}=w_0x_t,
\qquad
h_{t+1,i}=w_i h_{t,i-1},\quad i=1,\ldots,4.
\]

The output is the final state. With five multiplicative links, the valid output at
sequence index \(t\) corresponds to the input four index steps earlier; the
important construction object is the five-factor propagation gain

\[
g=\prod_{i=0}^4w_i.
\]

For a unit-variance binary target sequence, the exact objective is

\[
L=\frac12(g-1)^2.
\]

Choose base initializations with \(d\) missing links set to zero and the remaining
links prewired to one. Every base point has \(w_0=0\) and therefore implements the
same current zero predictor. Along the equal small perturbation ray for the
missing links,

\[
g=\delta^d
\]

and the exact loss improvement is

\[
\boxed{
L(0)-L(\delta)=\delta^d-\frac12\delta^{2d}.
}
\]

The missing-link gradient norm is

\[
\boxed{
\|\nabla_{\rm missing}L\|
=\sqrt d\,(1-\delta^d)\delta^{d-1}.
}
\]

Therefore the predicted loss and gradient orders are exactly \(d\) and \(d-1\).
The recurrent simulation and PyTorch autograd regressions recover orders one
through five and zero-initialization motion only for the first-order case.

A fixed-step SGD audit under one frozen optimizer configuration gives rapidly
increasing threshold-crossing times with construction order. These step counts are
illustrative evidence only, not a universal theorem or a new deep-linear result.
The value of this experiment is narrower: dormant downstream prewiring changes
local accessibility order in a smooth recurrent memory outside the finite-state
controller formalism.

## 7. When leading order is not enough

Construction order identifies the first useful local homogeneity. It does not by
itself determine which of several nearly tied computations will eventually win.

Let the leading construction degree be \(d\), and suppose the first correction
that distinguishes competing routes appears at degree \(p>d\). On the natural
small-initialization construction scale, the normalized leading-geometry error is
of order

\[
O(\delta^{p-d}).
\]

Consequently, a route coefficient or spectral gap comparable to this scale is not
robust. We freeze exact finite-memory examples in which cubic corrections reverse
the winner predicted by a quadratic shared-route model, and in which a nearly
degenerate bilinear construction spectrum is rotated by higher-order terms.

These examples delimit the theory:

\[
\boxed{
\text{leading order is reliable only with an adequate leading margin}.
}
\]

They are not separate headline novelty claims. Classical perturbation theory and
deep-linear singular-mode dynamics already provide the surrounding mathematical
context.

## 8. Related work and novelty boundary

### Finite-state-controller optimization

Policy-gradient methods for finite-state controllers are longstanding. Aberdeen
and Baxter (2002) explicitly analyze zero memory gradients near undifferentiated
controllers and motivate sparse controller structure. Braziunas and Boutilier
(2004) discuss sequential/local-search difficulty in POMDP controllers. Boularias
and Chaib-draa (2009) show that finite-horizon FSC/PSR objectives can be polynomial
and relate representation structure to polynomial degree.

We therefore do **not** claim that FSC gradients can vanish, that sequential
controller structure affects optimization, or that FSC objectives can be
polynomial. Our distinction is between a representation's global/maximal
polynomial degree and the **minimum useful local Taylor degree around a fixed
forward-equivalent base**, with an exact source-valid support/operator/decoder
factorization.

### Same-function embeddings and dormant features

Fukumizu et al. (2019) and later embedding-principle work show that
same-function neural-network embeddings can change flat/minimum/saddle geometry.
Kunin et al. (2025) analyze staged activation of dormant neural features. These
works make broad claims such as “same function, different landscape” or “dormant
features activate in stages” inappropriate novelty statements for this paper.

Our intervention is narrower: source, architecture, decoder, current reachable
dynamics, and—within the random census—the trainable direction family are fixed;
only behaviorally dormant zero-cost wiring changes. The resulting object is an
integer-valued local construction order with a finite-memory path/operator
interpretation.

### Singular learning and high-order flatness

Singular-learning theory studies degenerate parameter directions and higher-order
vanishing; recent work explicitly assigns orders to dead directions. We do not
claim the abstract notion of vanishing order. We compute the order of a specified
latent finite-memory construction and tie it to source-valid graph topology and
readout quotient structure.

### Homogeneous/deep-linear dynamics

Balancedness, singular-mode learning, and small-initialization saddle escape have
substantial prior art, including Saxe et al. (2014). We use these dynamics after
the finite-memory construction topology identifies the relevant degree. The
homogeneous escape exponent itself is not a novelty claim.

### Reparameterization and softmax geometry

Natural-gradient and path-geometry work establish that Euclidean optimization
trajectories depend on parameterization. Softmax policy-gradient convergence can
also be extremely slow in chain-like problems. Our parameterization results have
a narrower role: regular charts preserve the scalar loss order, while the
softmax-boundary calculation gives an explicit degree-to-time map near absent
transition edges.

## 9. Limitations

The finite-state theorems are exact but deliberately local and finite-horizon.
Several limitations should remain explicit.

1. **Finite horizon.** The polynomial expansion is exact because the horizon is
   finite. Infinite-horizon stationary objectives require separate control of
   limits and mixing.
2. **Affine transition family.** The support/operator theorem is formulated in
   affine construction directions. The scalar loss order survives regular local
   reparameterization, but support and operator orders remain structural objects
   tied to the affine transition family.
3. **Local accessibility, not global convergence.** A low construction order does
   not guarantee global optimization success; a high order does not rule out
   nonlocal jumps or alternate routes.
4. **Optimizer dependence.** Construction order is structural, but physical
   training time depends on metric, parameterization, learning-rate schedule,
   noise, adaptive methods, and coupling among routes.
5. **Boundary formulas.** The exact softmax antiderivative is derived for a
   symmetric binary-logit isolated construction. The full \(K\)-way result fixes
   the rare-edge exponent up to constants, not the exact prefactor for arbitrary
   coupled rows.
6. **Model scale.** The linear-SSM experiment shows that the mechanism survives
   outside finite-state controllers, but does not establish that construction
   order dominates optimization in large neural sequence models.
7. **Novelty evidence.** The related-work audit found several close conceptual
   neighbors. We found no direct prior theorem matching the source-valid
   support/operator/dormant-rewire construction, but absence from a targeted
   search is not proof of absolute novelty.

## 10. Conclusion

Predictive capacity answers whether a memory architecture *can* implement a useful
computation. Current behavior answers what it computes *now*. Neither determines
how readily local optimization can construct a latent computation from the
current parameter point.

For finite-state memory, this accessibility question admits an exact local
hierarchy:

\[
\boxed{
d_{\rm support}\le d_{\rm operator}\le d_{\rm loss}.
}
\]

Behaviorally dormant topology can change these orders while leaving the entire
current source-memory process unchanged. Randomizing only dormant wiring inside a
single fixed forward-equivalence class produces a broad accessibility spectrum.
The resulting scalar order is stable under ordinary local diffeomorphisms and
reappears in a smooth linear recurrent memory.

The dynamic consequence is best stated in two stages rather than as an
optimizer-independent escape theorem:

\[
\boxed{
\text{construction topology}
\to
\text{local construction order}
\to
\text{optimizer geometry}
\to
\text{bootstrap time}.
}
\]

Affine probability flow and rare-edge softmax flow translate the same degree into
different absolute escape exponents, yet both retain a severe advantage for
dormant scaffolding that lowers the order. In this sense, two systems can be
indistinguishable as predictors and still be fundamentally different as objects
for learning.
