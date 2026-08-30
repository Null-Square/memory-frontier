# Perturbative construction distance

The delay-chain result suggests a broader question: when a collapsed memory must assemble a computation through several missing transition links, is the first nonzero optimization signal determined by the number of missing links?

The unconditional equality is false. The correct statement is a structural **lower bound**, with equality whenever the leading path contributions do not cancel.

## Exact finite-horizon polynomial

Fix a stochastic source, decoder, horizon `T`, and an affine finite-memory controller

\[
P(\varepsilon)=P_0+\varepsilon P_1,
\]

where every row of `P_0` is stochastic and every row of `P_1` sums to zero. On the source-memory product chain,

\[
K(\varepsilon)=K_0+\varepsilon K_1.
\]

With initial product distribution `mu_0` and one-step prediction-cost vector `c`, the exact finite-horizon loss is

\[
L_T(\varepsilon)
=\frac1T\sum_{t=0}^{T-1}
\mu_0K(\varepsilon)^t c.
\]

Therefore

\[
\boxed{
L_T(\varepsilon)=\sum_{r=0}^{T-1}\ell_r\varepsilon^r
}
\]

is an exact polynomial of degree at most `T-1`. `affine_controller_loss_coefficients` computes every coefficient directly by propagating a polynomial source-memory occupancy; no finite differencing or autograd is used.

## Structural construction cost

Give every supported transition of `P_0` cost zero. A positive entry of `P_1` is a new-routing edge and costs one. Search on the full source-memory product graph, so token sequences with zero source probability are excluded.

The **decoder construction cost** `d` is the minimum accumulated perturbative cost needed, within the finite horizon, to reach a memory state whose decoder differs from the initial-memory decoder.

For dormant constructions in which all zero-cost reachable memories share the same decoder, any loss contribution that distinguishes memory must traverse at least `d` new-routing factors. Consequently

\[
\boxed{
L_T(\varepsilon)-L_T(0)=O(\varepsilon^d).
}
\]

Equivalently,

\[
\ell_1=\cdots=\ell_{d-1}=0.
\]

This recovers the delay-chain theorem as a special case. If all `R` links are missing, `d=R`; if the downstream dormant scaffold is already wired and only the entrance link is missing, `d=1`.

## Why equality is only generic

If

\[
\ell_d\neq0,
\]

then the perturbative order is exactly the structural construction cost. For fixed source/topology, `ell_d` is an analytic functional of the decoder parameters. Unless it vanishes identically, exact cancellation occurs only on a nongeneric parameter set.

A fixed-seed exploratory scan over 1,181 random two-state observable Markov sources and dormant controller graphs found:

- zero cases with perturbative order below construction cost;
- generic equality in every evaluated nondegenerate instance;
- structural distances from 1 through 4.

An earlier independent scan on 625 delayed-source/dormant-graph instances gave the same pattern. These scans are evidence for the generic statement, not a proof beyond the structural lower bound.

## Exact cancellation witness

The equality can be broken deliberately.

Use a binary second-order source with states `00,01,10,11` and

```text
P(next=0 | 00) = 0.9
P(next=0 | 01) = 0.5
P(next=0 | 10) = 0.5
P(next=0 | 11) = 0.1
```

The stationary distribution is `(5/12, 1/12, 1/12, 5/12)`, and after observing token `0`, the next-token probability of `0` is `5/6`.

Start with three collapsed memory states. Add two missing token-0 links,

```text
0 --0--> 1 --0--> 2,
```

each with probability `epsilon`; all failures return to memory 0. Memory 0 is uniform. Choose memory 1's nonuniform decoder `q` to satisfy

\[
\log2+\frac56\log q+\frac16\log(1-q)=0,
\]

using the nontrivial root

\[
q\approx0.9829741183.
\]

Thus memory 1 is decoder-distinct and structurally only one missing link away, but its entire first-order prediction effect cancels. Let memory 2 use decoder `(0.9,0.1)`.

At horizon 10 the exact polynomial begins

\[
L(\varepsilon)
=\log2
-0.0325581101\,\varepsilon^2
+0.0966168544\,\varepsilon^3
-\cdots.
\]

Hence

\[
\boxed{d_{struct}=1,\qquad d_{actual}=2.}
\]

The quadratic coefficient is beneficial: the first useful prediction gain is genuinely second order even though a decoder-distinct state is one missing link away.

This falsifies the naive slogan `computational distance = optimization order` without qualification.

## Refined research claim

The defensible version is:

> **Perturbative construction distance is a structural lower bound on the local optimization order of a dormant finite-memory computation. Equality holds when the leading source-conditioned path contributions do not cancel; dormant scaffolds lower the bound by supplying transition factors in advance.**

The sign of the leading coefficient remains a separate question. A structurally accessible computation can be useful, harmful, or exactly neutral at its first available order.

## Prior-art boundary

Power-series perturbation theory for stochastic matrices is classical, including Schweitzer's finite-Markov-chain perturbation formalism and later series-expansion work. Singularly perturbed Markov chains and stochastic-stability theory also use powers of a small parameter, graph paths, and resistance/shortest-path ideas to identify asymptotic transitions and stationary behavior.

The intended contribution here is not the general idea that epsilon-exponents correspond to paths. It is the source-conditioned application to **learned finite predictive memory**: exact next-token loss coefficients, construction-cost barriers for dormant memory computations, and their relationship to the accessibility phenomena measured elsewhere in this repository.
