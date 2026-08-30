# Stratified construction-order census

`experiments/general_construction_order_census.py` is an outside-CI breadth audit
for the general construction-order hierarchy.

The generator fixes a requested support depth `d` in `{1,2,3,4}` while
randomizing:

- the second-order unifilar source emission probabilities;
- zero-cost topology inside each behaviorally neutral memory layer;
- the symbols used by planted construction edges;
- parameter identities, including repeated use of the same parameter along a
  route;
- branching and distractor transition edits that may stay in a layer or advance
  by one layer;
- two informative target decoders.

All neutral memories have the same uniform decoder. Every perturbative edge can
advance by at most one layer, and one planted route advances by exactly one layer
at each construction step. Therefore the requested support depth is known by
construction while the exact occupancy polynomial is not hand-specified.

## Frozen reference run

Seed:

```text
20260830
```

Number of cases:

```text
1000
```

The requested depth cycles evenly through `1,2,3,4`. The observed triples were:

```text
(d_support, d_operator, d_loss)   count
(1, 1, 1)                        250
(2, 2, 2)                        250
(3, 3, 3)                        250
(4, 4, 4)                        250
```

Hierarchy violations:

```text
0
```

Thus all 1,000 generic random families saturated

\[
\boxed{
d_{\mathrm{support}}
=d_{\mathrm{operator}}
=d_{\mathrm{loss}}.
}
\]

This should not be interpreted as saying strict inequalities are impossible.
The deterministic neutral-decoder regression deliberately realizes

\[
(d_{\mathrm{support}},d_{\mathrm{operator}},d_{\mathrm{loss}})
=(1,1,2),
\]

by placing the decoder log-vector in the nullspace of the first-order
construction operator. The census instead supports the intended genericity
claim: once exact cancellations are not deliberately tuned, structural support
order usually survives both source-aware aggregation and decoder contraction in
this family.

The census is not a proof of generic equality and is not run in CI. Its role is
to check that the theorem is not supported only by one delayed-chain fixture.
