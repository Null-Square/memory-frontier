# Forward-equivalent dormant rewiring can change construction order

The construction-order theorem separates structural support cost, quotient
construction-operator order, and scalar loss order. Earlier dormant-scaffold and
counterfactual-automaton witnesses showed that unreachable internal structure can
matter to gradients despite being invisible to current behavior.

This note gives the general finite-horizon forward-equivalence statement that
connects those observations directly to the construction-order theorem.

## Source-aware active transition rows

Fix a unifilar source, a stochastic memory transition tensor `P`, an initial
memory state, and a horizon `T`.

Let

\[
\mu_t(s,m)
\]

be the source-memory occupancy at prediction time `t`. A memory transition row
`(m,x)` is **forward-active** when there exist

- a time `t<T-1`,
- a source state `s` with \(\mu_t(s,m)>0\), and
- positive source probability \(p(x\mid s)>0\).

Otherwise the row is **forward-dormant** over the specified source and horizon.

This definition is finer than ordinary memory-state reachability. A memory state
may be reachable while one of its symbol-conditioned rows is impossible under
the source histories that co-occur with it.

`finite_horizon_forward_support` computes this source-memory support exactly.

## Dormant-rewire theorem

Let `P` be the reference controller and let `P_tilde` be another valid stochastic
transition tensor. Assume

\[
\widetilde P(m,x,\cdot)=P(m,x,\cdot)
\]

for every forward-active row `(m,x)` of the reference process. The candidate may
differ arbitrarily on every forward-dormant row.

Then

\[
\boxed{
\widetilde\mu_t(s,m)=\mu_t(s,m)
\quad\text{for every }t<T.
}
\]

### Proof

At `t=0` both controllers have the same source distribution and initial memory,
so their product occupancies agree.

Assume their occupancies agree at time `t<T-1`. Every transition contribution
with nonzero probability comes from a product state `(s,m)` in the common
support and a symbol `x` with positive source emission probability. By definition,
`(m,x)` is a forward-active row of the reference process. The two transition
tensors agree on that complete row, so every contribution to the next product
occupancy is identical. Therefore the occupancies also agree at `t+1`.

Induction proves the result.

Because expected prediction loss at each time is a linear functional of the
source-memory occupancy for any fixed decoder,

\[
\boxed{
L_T(P,Q)=L_T(\widetilde P,Q)
\quad\text{for every decoder }Q.
}
\]

Thus this is stronger than equality of one selected scalar loss: the complete
current finite-horizon forward process is identical.

`is_source_horizon_dormant_rewire` implements the sufficient row-support
certificate used by the regressions.

## Why construction order can still change

Forward dormancy is a statement about the **base process**. Construction order
asks what becomes reachable after perturbative transition directions are used.

A perturbative edge can enter a memory region that has zero base occupancy. Once
that happens, pre-existing base transitions inside the previously dormant region
become zero-cost continuation edges in the construction graph.

Therefore a dormant rewire can change

\[
d_{\mathrm{support}}
\]

without changing the base predictor at all. Through the general hierarchy

\[
d_{\mathrm{support}}
\le d_{\mathrm{operator}}
\le d_{\mathrm{loss}},
\]

and the genericity results for the two inequalities, the dormant rewire
generically changes the actual derivative order as well.

This gives the paper-level separation:

\[
\boxed{
\text{forward equivalence does not imply accessibility equivalence.}
}
\]

## Exact arbitrary-order family

The regression freezes a delay-4 family, and the same construction extends to
arbitrary delay `R`.

Use `R+1` memory states. At the base point, the first chain entrance remains
missing, so memory never leaves state zero. Hence all variants execute the same
collapsed current predictor.

For each

\[
r\in\{1,\ldots,R\},
\]

leave exactly the first `r` chain links perturbative and prewire all downstream
links. Every prewired downstream row is forward-dormant because the missing
first link prevents the current process from entering the chain.

The variants are therefore forward-equivalent, while the construction theorem
gives

\[
\boxed{
(d_{\mathrm{support}},d_{\mathrm{operator}},d_{\mathrm{loss}})
=(r,r,r)
}
\]

for the generic delay-matched decoder used in the exact chain family.

For the CI fixture with `R=4`, the same current predictor therefore realizes all
four accessibility orders

\[
\boxed{1,2,3,4}
\]

solely by changing behaviorally dormant topology.

Earlier delay-5 tests already freeze the analogous orders `1..5`; the present
result supplies the general forward-equivalence theorem that explains why those
base controllers are genuinely behaviorally identical.

## Source-aware dormancy is stronger than unreachable-state dormancy

A second regression uses a one-state source that emits symbol zero with
probability one. Memory state zero is reachable, but transition row `(0,1)` can
never be exercised because symbol one is source-impossible.

Rewiring that row to a completely different target is certified dormant and
leaves the exact loss unchanged even when the target memory has a radically
different decoder.

Changing row `(0,0)`, by contrast, is active, is not certified dormant, and does
change the forward loss.

Thus the relevant equivalence is not merely graph reachability in the controller:
it is reachability in the **source-memory product process**.

## Relation to the hard-forward counterfactual result

The earlier straight-through counterfactual-automaton theorem showed that
unreachable hard-state transitions can alter the STE gradient while leaving the
hard forward trajectory fixed.

The present statement is different and complementary:

- it is about the exact stochastic finite-horizon loss polynomial rather than an
  STE surrogate;
- it proves complete forward equivalence under source-aware dormant rewiring;
- the construction-order theorem then explains how that same dormant topology
  changes the first nonzero derivative degree once perturbative routes are
  considered.

Together they show the same conceptual distinction in both exact smooth and
hard-forward/surrogate settings.

## Prior-art boundary

Removing unreachable automaton states without changing accepted behavior is
classical, and finite-state-controller optimization is established. We do not
claim the behavioral irrelevance of unreachable states as new.

The project-specific result is the optimization consequence:

\[
\boxed{
\text{within one exact forward-equivalence class, behaviorally dormant finite-memory
wiring can realize different construction orders and therefore different local
learnability classes.}
}
\]

This is the cleanest formal statement so far of the project's central distinction
between predictive capacity/current behavior and gradient accessibility.