# Theory notes

These notes define the current deterministic finite-memory problem precisely. They are working propositions for the research program, not novelty claims.

## Setup

Let a finite unifilar source have state `S_t`, token `X_{t+1}`, emission law `p(x|s)`, and deterministic state update `delta(s,x)`. A `K`-state predictor has persistent memory

\[
M_{t+1}=F(M_t,X_{t+1}), \qquad M_t\in[K].
\]

For a fixed controller, the optimal time-homogeneous readout under log loss is its conditional next-symbol distribution. Thus the asymptotic score is

\[
L(F,m_0)=H(X_{t+1}\mid M_t)
\]

under the Cesaro occupancy of the product chain started from stationary `S_0` and fixed `M_0=m_0`.

We compare three cardinality-`K` quantities:

- `L_static`: best deterministic map `g:S->[K]`, with no requirement that `g(S_t)` be recursively updateable.
- `L_det`: best executable deterministic controller `(F,m0)`.
- `L_quotient`: best recursively closed state partition `g`, meaning equal memory labels always induce equal successor labels under every possible token.

For the source laboratory we impose three conditions: strictly positive emissions, strong connectivity, and exact synchronization. We additionally require distinct emission rows, so different source states already have different one-step predictive laws.

## Proposition 1: static compression is a lower bound

For any deterministic controller with at most `K` memory values,

\[
L_K^{static}\le L(F,m_0).
\]

Sketch: the long-run joint occupancy induces a stochastic channel `R(m|s)`. Since the next symbol is conditionally independent of controller memory given the source state, the controller loss is `H(X|M)` for the Markov chain `X-S-M`. For fixed `p(s,x)`, maximizing `I(X;M)` over the convex set of channels `R(m|s)` attains an optimum at an extreme point, and extreme points are deterministic maps `S->[K]`. Equivalently, the best static deterministic partition is at least as informative for next-token prediction as any `K`-valued stochastic encoding induced by history.

Therefore

\[
L_K^{static}\le L_K^{det}.
\]

## Proposition 2: synchronization makes every recursive quotient executable asymptotically

Assume the source has a reset word `w` that sends every source state to the same state `s*`, and every symbol in `w` has positive probability along every relevant path. Let `g:S->[K]` be a recursively closed partition.

Recursive closure makes

\[
F(g(s),x)=g(\delta(s,x))
\]

well-defined. Applying the reset word sends every quotient label to `g(s*)`. When `w` appears in the observed stream, controller memory and source quotient synchronize; recursive closure keeps them synchronized forever. Under a finite irreducible positive-emission source, the reset word occurs almost surely, so its transient cost vanishes in Cesaro average.

Hence

\[
L_K^{det}\le L_K^{quotient}.
\]

Combining Propositions 1 and 2 gives, on the source-lab domain,

\[
\boxed{L_K^{static}\le L_K^{det}\le L_K^{quotient}}.
\]

The code regression-tests this ordering on fixed random synchronized sources, but the argument above is the reason for imposing synchronization. Without synchronization, the right inequality can fail under fixed-reset operational semantics.

## Proposition 3: a strict quotient advantage forces history-dependent aliasing

Let the source be irreducible with strictly positive emissions. If a deterministic controller satisfies

\[
H(M\mid S)=0,
\]

then `M=g(S)` almost surely for some deterministic map `g`. Because every symbol has positive probability, deterministic controller updating forces

\[
g(s)=g(s')\implies g(\delta(s,x))=g(\delta(s',x))
\]

for every symbol `x`. Thus `g` is a recursively closed quotient.

Therefore

\[
\boxed{L_K^{det}<L_K^{quotient}\implies H(M\mid S)>0}.
\]

So whenever an executable controller strictly beats every fixed quotient, it must use history-dependent aliases of at least one source state.

## Finite-horizon oracle

Neural training normally resets memory at sequence boundaries, so the primary experimental oracle should match that objective rather than only its infinite-horizon limit.

For sequence length `T`, define average product occupancy

\[
\bar\mu_T(s,m)=\frac1T\sum_{t=0}^{T-1}P(S_t=s,M_t=m),
\]

with stationary `S_0` and fixed `M_0=m_0`. A single shared readout `q(x|m)` is optimized using token counts aggregated over all positions, giving exact expected loss

\[
L_T(F,m_0)=H_{\bar\mu_T}(X\mid M).
\]

The finite-horizon oracle is

\[
L_{K,T}^{det}=\min_{F,m_0}L_T(F,m_0).
\]

For small `K` this is again exhaustive. It charges synchronization and transient behavior exactly and can predict changes in the globally optimal transition algorithm as training sequence length changes.

## Current terminology

Two gaps are useful on exactly synchronizing sources:

\[
\tau_K=L_K^{det}-L_K^{static}
\]

is the **realizability gap**: the price of requiring a `K`-state representation to be executable online.

\[
\alpha_K=L_K^{quotient}-L_K^{det}
\]

is the **aliasing advantage**: the gain available to executable history-dependent coding over every fixed source-state quotient.

A positive aliasing advantage implies positive alias entropy `H(M|S)` by Proposition 3. The converse need not hold.
