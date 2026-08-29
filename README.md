# Memory Frontier

**Research question:** with exactly `K` persistent discrete memory states, can a learned sequence predictor discover the best deterministic online algorithm for a known stochastic source?

This repository starts from **exact ground truth**, not neural training. For small memory budgets we exhaust every deterministic transition table

\[
F : [K] \times \Sigma \to [K]
\]

and score each controller exactly under stationary next-token log loss.

## Operational oracle

For a unifilar source with predictive state `S_t`, emitted symbol `X_{t+1}`, and deterministic controller memory `M_t`,

\[
M_{t+1}=F(M_t,X_{t+1}).
\]

The source starts in its stationary distribution. The controller starts in a chosen discrete state `m0`. We score the **asymptotic Cesàro occupancy** of the product chain `(S_t, M_t)`, so periodic chains and controllers with multiple recurrent classes are defined without an ergodicity shortcut.

For a fixed `F` and `m0`, the optimal readout under log loss is analytic:

\[
q^*(x\mid m)=P(X_{t+1}=x\mid M_t=m),
\]

and the resulting loss is

\[
L(F,m_0)=H(X_{t+1}\mid M_t).
\]

The deterministic oracle is

\[
L_K^{det}=\min_{F,m_0} L(F,m_0).
\]

For a binary alphabet there are only `K^(2K)` transition tables: 16 at `K=2`, 729 at `K=3`, and 65,536 at `K=4`.

## First witness

The included four-state source exhibits three distinct quantities at `K=2`:

| quantity | NLL (nats/token) |
|---|---:|
| Bayes, full predictive state | 0.4662344730 |
| best static 2-way state partition | 0.4840697673 |
| **best deterministic 2-state online controller** | **0.6297909185** |
| best recursively closed 2-way state partition | 0.6886553518 |

The optimal executable controller is

```text
        token 0   token 1
M=0        0         1
M=1        0         1
```

so its memory is simply the previous symbol. Crucially, one true source state has positive stationary mass in **both** memory states. Thus the optimal controller is not a fixed partition `M=g(S)` of the source state: it uses history-dependent aliases.

Run:

```bash
python -m pip install -e '.[dev]'
pytest -q
python examples/witness_report.py
```

## Baselines and a deliberate caveat

`best_static_partition` and `best_recursive_quotient` are representational comparison baselines. The exact online oracle has explicit reset semantics; an abstract recursively closed state partition may assume a source-state correlation that is not reachable from every controller reset. We therefore **do not yet assert a universal sandwich theorem involving the quotient baseline** under the operational reset definition. Establishing the precise conditions for such an ordering is a theory task, not a unit-test assumption.

## Research discipline

1. Derive and freeze exact oracle results before neural training.
2. Keep persistent-state cardinality literal: no hidden continuous state, context window, or KV cache.
3. Compare learned **transition algorithms**, not only losses.
4. Treat stochastic controllers as a separate computational resource class.
5. Prefer falsifiable source families and matched controls over benchmark accumulation.

## Current scope

This first milestone contains only the deterministic oracle, exact Markov-chain scoring, partition baselines, one verified witness, and tests. Neural learners, random-source scans, stochastic controllers, and scaling experiments come after the oracle core is independently hardened.
