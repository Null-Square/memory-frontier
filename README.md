# Memory Frontier

**Research question:** with exactly `K` persistent discrete memory states, can a learned sequence predictor discover the globally best deterministic online algorithm for a known stochastic source?

This repository starts from **exact ground truth**, not neural training. For small memory budgets we exhaust every deterministic transition table

\[
F:[K]\times\Sigma\to[K]
\]

and score each controller analytically under next-token log loss.

Finite-state prediction itself is classical. The research target here is narrower: construct source-specific exact controller ground truth, characterize when optimal bounded memory requires history-dependent aliases rather than a fixed quotient of predictive states, freeze those predictions before training, and then test whether gradient descent discovers the predicted algorithms.

## Two exact oracles

For a unifilar source with state `S_t`, emitted token `X_{t+1}`, and deterministic memory `M_t`,

\[
M_{t+1}=F(M_t,X_{t+1}).
\]

### Asymptotic oracle

The source begins in stationarity and the controller in a chosen state `m0`. We score the Cesaro occupancy of the product chain `(S_t,M_t)`, so periodic chains and multiple recurrent classes are handled explicitly. For fixed `(F,m0)`, the optimal readout is analytic and the loss is

\[
L(F,m_0)=H(X_{t+1}\mid M_t).
\]

The deterministic frontier is

\[
L_K^{det}=\min_{F,m_0}L(F,m_0).
\]

### Finite-horizon reset oracle

Neural training usually resets memory at sequence boundaries. For a length-`T` sequence we therefore also compute the exact average occupancy over `t=0,...,T-1` and optimize the single shared readout. This gives

\[
L_{K,T}^{det}=\min_{F,m_0}L_T(F,m_0),
\]

which matches the actual reset-sequence training objective and charges transient/synchronization behavior instead of making it free.

For a binary alphabet there are only `K^(2K)` transition tables: 16 at `K=2`, 729 at `K=3`, and 65,536 at `K=4`.

## Source laboratory

Random experimental sources are deliberately restricted to be:

1. strongly connected,
2. strictly positive in every emission probability,
3. exactly synchronizing, and
4. pairwise distinct in their one-step emission laws.

The synchronization condition is essential. A generic unifilar hidden-state machine need not expose its state from token history, which would confound memory limitation with hidden-state estimation. The included generator searches for a finite reset word and rejects non-synchronizing sources.

On this controlled source class the current theory notes establish the working ordering

\[
L_K^{static}\le L_K^{det}\le L_K^{quotient},
\]

where `static` is the best unrestricted `K`-way source-state partition and `quotient` is the best recursively closed partition. See [`docs/theory.md`](docs/theory.md) for assumptions and proof sketches.

## First witness

The included four-state source is exactly synchronizing: token word `11` sends every source state to A. At `K=2` it gives:

| quantity | NLL (nats/token) |
|---|---:|
| Bayes, full source state | 0.4662344730 |
| best static 2-way state partition | 0.4840697673 |
| **best deterministic 2-state online controller** | **0.6297909185** |
| best recursively closed 2-way state partition | 0.6886553518 |

The optimal asymptotic controller is

```text
        token 0   token 1
M=0        0         1
M=1        0         1
```

so it remembers the previous token. One true source state has positive stationary mass in both memory states, therefore the optimal controller is **not** a fixed map `M=g(S)`: it uses history-dependent aliases.

For finite reset sequences the exact optimal algorithm changes with horizon. On this witness:

```text
T=1  -> trivial memory
T=2  -> [[0,0],[0,1]]
T=3  -> [[0,0],[1,0]]
T>=4 -> [[0,1],[0,1]]  (last-symbol memory, for the tested horizons)
```

These are precomputable algorithm-level predictions for later neural experiments.

## Theory cards

`theory_card(source, K)` records and hashes the pre-training ground truth, including:

- source transition/emission tables and stationary distribution,
- shortest synchronizing word,
- Bayes, static, deterministic, and quotient losses,
- exact optimal deterministic transition table,
- alias entropy `H(M|S)`,
- full controller-spectrum summary,
- distinct-loss gap and near-optimal counts, and
- a SHA-256 digest for freezing the card before model training.

## Run

```bash
python -m pip install -e '.[dev]'
pytest -q
python examples/witness_report.py
python examples/witness_horizon_report.py
python examples/scan_sources.py --count 500 --states 4 --memory 2
```

## Research discipline

1. Derive and freeze exact oracle results before neural training.
2. Keep persistent-state cardinality literal: no hidden continuous state, context window, or KV cache.
3. Match the oracle horizon/reset semantics to the learner's training objective.
4. Compare learned **transition algorithms**, not only losses.
5. Treat stochastic state transitions as a separate computational resource class.
6. Use synchronized source families and matched controls rather than arbitrary HMMs.
7. Treat classical finite-state prediction, recursive information bottleneck, automata reduction, and predictive rate-distortion as prior foundations rather than novelty claims.

## Current milestone

The repository now contains the hardened asymptotic oracle, exact finite-horizon oracle, synchronized random-source laboratory, controller spectra modulo memory-state relabeling, hashed theory cards, theorem notes, and verified witness experiments. Neural learners remain intentionally out of scope until the source families and oracle landscape are frozen.
