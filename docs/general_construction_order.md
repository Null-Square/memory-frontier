# General construction-order theorem

This note isolates the central finite-memory optimization statement in a form that
separates three different notions of order:

1. **support construction cost**: how many perturbative transition uses are
   required by any source-valid path before the controller can reach a memory
   with a different decoder row;
2. **quotient operator order**: the first total degree at which the exact
   source-memory occupancy polynomial changes the distribution over distinct
   decoder classes;
3. **scalar loss order**: the first total degree visible after the actual decoder
   values are contracted with that operator.

The exact hierarchy is

\[
\boxed{
 d_{\mathrm{support}}
 \;\le\;
 d_{\mathrm{operator}}
 \;\le\;
 d_{\mathrm{loss}}.
}
\]

The two possible strict inequalities have different causes. Transition-path
cancellation can raise `d_operator` above the combinatorial support distance.
Decoder-value cancellation can raise `d_loss` above the first nonzero operator
degree.

This distinction turns the earlier chain examples into instances of one general
finite-horizon statement rather than separate constructions.

## Setup

Let the source be a finite unifilar process with hidden source state `s`, emitted
symbol `x`, emission probability

\[
p(x\mid s),
\]

and deterministic source-state update after each symbol.

Let a stochastic finite-memory controller have transition family

\[
P(\varepsilon)
=
P_0+
\sum_{j=1}^{r}\varepsilon_j P_j,
\]

where `P0` is row-stochastic and every perturbation direction has zero row sum:

\[
\sum_{m'}P_j(m,x,m')=0.
\]

The decoder attached to memory `m` is the strictly positive distribution

\[
q_m(x).
\]

For a finite horizon `T`, the expected log loss is an exact multivariate
polynomial in the transition perturbations:

\[
L(\varepsilon)
=
\sum_{\alpha} c_{\alpha}\varepsilon^{\alpha}.
\]

Here

\[
|\alpha|=\sum_j\alpha_j
\]

counts the number of perturbative transition factors used by a term.

## Exact occupancy expansion

Let

\[
D_{t,\alpha}(s,m)
\]

be the coefficient of `eps**alpha` in the source-memory occupancy at time `t`.
The finite-horizon loss coefficient is

\[
c_{\alpha}
=-\frac1T
\sum_{t,s,m,x}
D_{t,\alpha}(s,m)
\,p(x\mid s)
\log q_m(x).
\]

This formula is just the exact path expansion of the affine transition family.
No sampling, automatic differentiation, STE, or finite-difference approximation
is involved.

### Probability-conservation identity

For every nonconstant exponent

\[
|\alpha|>0,
\]

the occupancy coefficient has zero total mass at each source state:

\[
\boxed{
\sum_mD_{t,\alpha}(s,m)=0.
}
\]

The identity follows by induction. It is true initially because every
nonconstant occupancy coefficient is zero. Propagation through `P0` preserves
zero total mass because its rows sum to one; propagation through any `Pj`
creates zero total mass because the rows of `Pj` sum to zero.

This conservation identity is the reason equal-readout memory states can be
quotiented exactly.

## Quotient by decoder equality

Partition memories into classes

\[
C_1,\ldots,C_K
\]

such that two memories lie in the same class exactly when their decoder rows are
equal. Denote the shared decoder of class `C` by `q_C`.

Define the **construction operator coefficient**

\[
G_{\alpha}(C,x)
=
\frac1T
\sum_t\sum_s\sum_{m\in C}
D_{t,\alpha}(s,m)p(x\mid s).
\]

Then the scalar loss coefficient factors exactly as

\[
\boxed{
c_{\alpha}
=-\sum_{C,x}
G_{\alpha}(C,x)\log q_C(x).
}
\]

`construction_order_operator` computes every `G_alpha` directly.

The important separation is:

- source statistics and transition construction geometry determine `G_alpha`;
- equality of decoder rows determines the quotient classes;
- numerical decoder values only enter through the final contraction with
  `log q_C`.

## Three orders

### 1. Support construction cost

`minimum_readout_class_construction_cost` assigns:

- cost zero to nonzero base-transition support;
- cost one to every nonzero perturbation-direction edge.

It searches the exact source-memory product graph for a source-valid path from
the initial memory into a different decoder class within the finite horizon.
The minimum cost is

\[
d_{\mathrm{support}}.
\]

This is purely combinatorial. It knows which token histories are possible, but
it does not know signed path coefficients.

### 2. Quotient operator order

Define

\[
d_{\mathrm{operator}}
=
\min\{\,|\alpha|>0:G_{\alpha}\ne0\,\}.
\]

This is source-aware and coefficient-aware. Signed route contributions have
already been summed, but numerical decoder values have not yet been used.

### 3. Scalar loss order

Finally,

\[
d_{\mathrm{loss}}
=
\min\{\,|\alpha|>0:c_{\alpha}\ne0\,\}.
\]

This is the derivative order seen by optimization for the actual fixed decoder.

## The support lower bound

Assume `d_support=d>0`. For any exponent with

\[
0<|\alpha|<d,
\]

no source-valid term using `|alpha|` perturbative edges can reach a memory in a
different decoder class. Therefore all nonzero occupancy contributions at that
degree remain inside the initial decoder class.

But the nonconstant occupancy coefficient has zero total memory mass at each
source state. After aggregation inside the single reachable decoder class,

\[
G_{\alpha}=0.
\]

Hence

\[
\boxed{
d_{\mathrm{support}}\le d_{\mathrm{operator}}.}
\]

This is a hard structural lower bound. It does not assume generic decoder
values.

## Decoder contraction bound

If

\[
G_{\alpha}=0,
\]

then necessarily

\[
c_{\alpha}=0.
\]

Therefore

\[
\boxed{
d_{\mathrm{operator}}\le d_{\mathrm{loss}}.}
\]

At degree `d_operator`, stack all flattened `G_alpha` rows into a matrix

\[
\mathcal G_d.
\]

Let

\[
z=\operatorname{vec}(\log q_C).
\]

The complete degree-`d` coefficient vector is

\[
\boxed{
c_d=-\mathcal G_d z.}
\]

Thus the exact condition for decoder-value cancellation at the first available
construction degree is

\[
\boxed{
\mathcal G_d z=0.
}
\]

The cancellation is not mysterious: the decoder log-vector lies in the nullspace
of the first nonzero construction operator.

`decoder_cancellation_residual` evaluates the norm of this vector exactly up to
floating-point arithmetic.

## Existing results become special cases

### Missing-link chains

For the delay-chain families with independent missing links,

\[
d_{\mathrm{support}}
=d_{\mathrm{operator}}
=d_{\mathrm{loss}}
=d.
\]

The mixed monomial containing all missing transitions is the first available
construction.

### Dormant scaffolding

Prewiring behaviorally unreachable downstream transitions changes the support
cost without changing the current predictor. In the delay-5 family, every order

\[
1,2,3,4,5
\]

is realized by leaving only that many prefix links missing. All three orders
track the missing prefix exactly.

This gives the clean functional-equivalence statement:

\[
\boxed{
\text{same current predictor, different construction order.}
}
\]

### Decoder symmetry

If every memory has the same decoder row, there is only one quotient class.
Probability conservation then forces every nonconstant quotient operator to
vanish:

\[
d_{\mathrm{operator}}=d_{\mathrm{loss}}=\infty
\]

for transition-only perturbations.

This is the fixed-decoder version of the readout symmetry trap. Joint decoder
training requires the separate joint-order analysis because changing decoder
parameters changes the quotient itself.

### Exact neutral-decoder cancellation

The existing second-order Markov cancellation fixture has

\[
\boxed{
(d_{\mathrm{support}},d_{\mathrm{operator}},d_{\mathrm{loss}})
=(1,1,2).
}
\]

A distinct decoder class is structurally accessible at first order, and the
first-order construction operator is nonzero, but its contraction with the
chosen neutral decoder log-vector vanishes exactly. The degree-two contraction
is nonzero.

This separates a **construction barrier** from a **readout cancellation** in a
way graph distance alone cannot.

### Shared-route systems

The non-delayed shared-route witness has

\[
(d_{\mathrm{support}},d_{\mathrm{operator}},d_{\mathrm{loss}})
=(2,2,2).
\]

Only after this order has been established does the quadratic route matrix and
its singular-mode competition become relevant. Construction order therefore
precedes route-selection geometry in the hierarchy.

## Consequence for gradient-flow time

When the first beneficial isolated construction has degree `d_loss=d`, the
small-initialization flow results already established in this repository apply:

\[
\tau_d=
\begin{cases}
O(1), & d=1,\\
O(\log(1/\delta)), & d=2,\\
\Theta(\delta^{-(d-2)}), & d\ge3.
\end{cases}
\]

The new theorem identifies where that `d` comes from:

\[
\boxed{
\text{source-valid topology}
\to
\text{quotient construction operator}
\to
\text{decoder contraction}
\to
\text{derivative order}
\to
\text{construction time}.
}
\]

The flow exponent itself is not claimed as new general optimization mathematics.
The contribution here is the exact finite-memory map from computation structure
to the degree entering that dynamics.

## Random exact-family audit

`experiments/general_construction_order_census.py` generates many random finite
families with:

- a random second-order unifilar source;
- random zero-cost topology inside a behaviorally collapsed uniform-readout
  memory class;
- several independently parameterized transition edits;
- two random informative target decoders.

For every accepted instance it computes the three orders independently and
checks

\[
d_{\mathrm{support}}
\le d_{\mathrm{operator}}
\le d_{\mathrm{loss}}.
\]

This experiment is outside CI because its purpose is breadth, not regression.
The deterministic CI fixtures freeze the exact algebraic identities and the
known strict decoder-cancellation example.

## Relation to established work

Finite-state-controller optimization difficulties are established. Aberdeen and
Baxter's internal-state policy-gradient work, and Aberdeen's 2003 thesis in
particular, analyze regions where symmetric FSC parameterizations give zero or
very small memory gradients and motivate structural constraints. That prior work
means zero memory gradients or symmetry traps by themselves are not novelty
claims here.

Small-initialization and saddle-to-saddle dynamics are also established in deep
linear and homogeneous neural networks, including exact balance/conservation
laws and singular-mode descriptions. The escape-time consequences in this
repository should therefore be read as an application of that optimization
geometry to an exactly identified finite-memory construction degree.

Modern state-space-model work likewise shows that memory initialization can
strongly affect optimization conditioning. The distinction here is that the
finite-memory systems admit an exact finite-horizon polynomial and a discrete
source-valid construction graph, allowing the derivative order to be traced to
specific latent computational routes.

The strongest project-specific statement is therefore not "memory can have bad
gradients." It is the exact separation

\[
\boxed{
\text{predictive capacity}
\neq
\text{gradient accessibility},
}
\]

with accessibility decomposed into support construction cost, source-aware
operator order, and decoder-specific loss order, all while behaviorally dormant
topology can vary these quantities without changing the current predictor.

## Scope

The theorem above is for finite-horizon expected log loss with affine stochastic
transition perturbations and fixed strictly-positive decoder rows.

It does not claim:

- invariance to arbitrary reparameterization;
- equality of support and loss order without checking cancellations;
- a joint transition/decoder theorem when the decoder equality partition itself
  changes during training;
- a universal route-winner theorem after the first construction becomes active;
- optimizer independence beyond the separately analyzed gradient-flow and
  finite-step results.
