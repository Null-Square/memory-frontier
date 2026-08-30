# Exact hard-cell dynamics

Training a hard-forward finite-memory model is naturally a hybrid dynamical system. Transition logits evolve continuously while their argmax table is fixed; when an argmax boundary is crossed, the discrete controller changes and the continuous dynamics switch to a new cell.

For a fixed hard table `F`, source, horizon `T`, and reset memory, the entire forward occupancy

\[
d_t(s,m)=P(S_t=s,M_t=m)
\]

is fixed. Decoder logits may change prediction quality, but cannot change `d_t` until the hard transition table changes.

## Cached cell factorization

Let the current decoder distribution be `r_m(x)` and define the immediate source-memory cost

\[
c(s,m)=-\sum_x P(x\mid s)\log r_m(x).
\]

The finite-horizon loss is a fixed linear functional of these costs. More importantly, the counterfactual transition derivative is also linear:

\[
\boxed{G_F=K_F c}
\]

for a tensor `K_F` depending only on the source, hard table, horizon, and reset semantics.

`build_hard_cell_oracle` precomputes:

- exact forward source-memory occupancy;
- exact memory/token mass used by decoder gradients;
- the linear counterfactual cost-to-transition kernel `K_F`.

`HardCellOracle.evaluate(z, a)` then computes exact loss, transition-tensor gradients, STE transition-logit gradients, and decoder-logit gradients without repeating the source/controller dynamic program.

The cell is valid exactly while

\[
\operatorname{argmax} z = F.
\]

After a hard boundary crossing, rebuild the oracle for the new table.

## Hybrid training view

Exact SGD can therefore be written as

```text
build cell oracle for F_0
while training:
    evaluate exact within-cell gradient
    update continuous logits
    if argmax table changed:
        rebuild oracle for F_1
```

This is not an approximation. It is a factorization of the existing exact-gradient oracle.

## Verification

Regression tests compare the cached evaluator against `exact_ste_gradient` on random decoder/transition logits and non-unit backward temperatures. Errors are at floating-point precision.

A second test runs two copies of the same 50-step SGD trajectory on a delay-2 source with `K=3`:

1. one recomputes the full exact source/controller gradient every step;
2. one reuses the hard-cell oracle and rebuilds only after an argmax crossing.

The trajectory crosses multiple hard-controller cells and the parameter vectors remain equal to better than `1e-12` throughout.

## Why this matters

The earlier accessibility and edit-alignment theory is local to one hard controller. The hard-cell factorization gives the missing composition rule: training is a sequence

\[
F_0 \xrightarrow{\tau_1} F_1 \xrightarrow{\tau_2} F_2 \rightarrow \cdots
\]

of exact within-cell flows and exact discrete boundary events.

This makes it possible to study full optimization trajectories while retaining exact source-conditioned ground truth at every stage, rather than reducing training to final success/failure statistics.
