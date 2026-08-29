# Optimization diagnostics

The next research phase compares gradient-trained hard-state predictors with an exactly enumerable combinatorial objective. The goal is to separate three questions that ordinary validation loss conflates:

1. **capacity:** does a `K`-state algorithm with the desired loss exist?
2. **combinatorial landscape:** which hard algorithms are global/local minima?
3. **surrogate optimization:** does the gradient estimator point toward improving hard edits and does training discover the exact optimum?

## Exact hard-controller graph

For fixed source, memory budget `K`, reset state `m0`, and horizon `T`, every deterministic table

\[
F:[K]\times\Sigma\to[K]
\]

is a node. A directed edge changes exactly one transition target. For edge `F -> F'`, define

\[
G(F\to F')=L_T(F)-L_T(F').
\]

`G>0` means the one-edit move strictly improves the exact finite-horizon objective. Because the graph is finite, global minima, one-edit local minima, exact gaps, and complete descent structure are ground truth rather than estimates.

At `K=2` with a binary alphabet there are 16 labeled transition tables and 64 directed one-edit edges.

## Canonical straight-through field

A hard transition table does not uniquely determine a neural parameter vector, so a table-level surrogate field needs an explicit embedding convention. `canonical_ste_field` uses:

- symmetric transition logits with a chosen positive margin,
- a hard argmax transition in the forward pass,
- a temperature-scaled softmax Jacobian in the backward pass,
- the exact Bayes readout of the hard controller on occupied memory states, and
- the source marginal as the declared readout convention for an unoccupied memory state.

The last choice matters: an unused memory state has no data-defined optimal readout, so differentiating through an analytically minimized entropy without declaring a convention is ill-defined at that boundary.

For a one-edit alternative, let `g` be the gradient with respect to transition logits and define

\[
P=g_{current}-g_{alternative}.
\]

Under gradient descent, positive `P` decreases the current-vs-alternative margin, so `P>0` favors switching. We can therefore compare the signs of `P` and the exact gain `G` directly.

This is called **sign fidelity**. It is not asserted to be a complete theory of optimization; it is a controlled local diagnostic whose assumptions are encoded in the API.

## Active-memory filter

Tables with an unreachable memory state can have trivial zero-gradient/tied directions. A surrogate-stable hard table is therefore not interpreted as a substantive false basin unless every memory state has positive finite-horizon occupancy. `CanonicalSTEField.fully_active()` exposes this filter.

## Verified pilot facts

The committed regression tests currently freeze two qualitative findings.

### Original four-state witness, `K=2`, `T=32`

The exact optimum with reset memory 0 is

```text
[[1, 0],
 [1, 0]]
```

At canonical logit margins `0.05, 0.25, 1.0, 4.0`, exactly two of its four one-edit decisions have surrogate pressure with the opposite sign from the exact hard finite difference: cells `(memory=0, symbol=0)` and `(memory=1, symbol=1)`.

Thus an exact global hard optimum need not be a fixed point of the canonical straight-through field. At large margins the disagreeing gradients shrink, so training can still remain near the hard optimum through saturation.

### Frozen horizon-switch source, `K=2`

The exact optimum changes from

```text
T=4:  [[0, 1],
       [0, 1]]
```

to

```text
T=32: [[0, 1],
       [0, 0]]
```

The `T=32` optimum has fully aligned canonical one-edit gradient signs at margin 1 and is canonical-surrogate stable.

These two fixtures intentionally show that surrogate misalignment is source/algorithm dependent rather than a universal property of the estimator.

## Exact-distribution learner

`train_exact_distribution_ste_batch` trains a literal hard categorical memory state with a learned readout. It never samples token sequences. Instead it propagates the exact source-memory distribution at every step and optimizes the exact expected sequence NLL with a straight-through transition estimator.

Multiple seeds are tensor-batched only for speed; their parameters are disjoint. The optimizer trajectory records every change of the hard transition table and attaches its exact oracle loss.

The reference pilot command is:

```bash
python -m pip install -e '.[optimization]'
python examples/ste_reference_report.py
```

The fixed pilot protocol is 40 seeds (`0..39`), `K=2`, `T=32`, 180 Adam steps, learning rate `0.05`, transition-logit initialization scale `0.25`, and reset memory state 0. These numbers are a pilot diagnostic, not yet a tuned benchmark or a final scientific claim.

## Interpretation discipline

- Do not call a canonical field a property of a hard table without naming the margin/readout convention.
- Do not count unreachable-state zero gradients as substantive optimizer traps.
- Do not compare learned validation loss with a different horizon/reset oracle.
- Prefer exact algorithm recovery and exact oracle loss to sampled validation accuracy.
- Hyperparameter tuning must be separated from the frozen evaluation suite before stronger optimizer comparisons are made.
