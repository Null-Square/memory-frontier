# Multivariate memory-construction order

The scalar perturbation family `P(epsilon)=P0+epsilon P1` is useful for measuring a common initialization scale, but it can hide the actual local geometry when distinct transition links are independent parameters. The correct object for optimization is therefore multivariate.

## Exact multivariate polynomial

Let

\[
P(\boldsymbol\varepsilon)
=
P_0+
\sum_{j=1}^r \varepsilon_j P_j,
\]

where every `P_j` has row sum zero. For a fixed source, decoder, and finite horizon `T`, the exact expected prediction loss is a finite multivariate polynomial

\[
\boxed{
L_T(\boldsymbol\varepsilon)
=
\sum_{\alpha\in\mathbb N^r,
\,|\alpha|\le T-1}
\ell_\alpha
\boldsymbol\varepsilon^\alpha.
}
\]

`multivariate_controller_loss_coefficients` computes this sparse polynomial exactly by propagating a dictionary of source-memory occupancy coefficients. No autograd, sampling, or numerical differentiation is required.

## Independent links in the delayed-memory chain

Give every missing link in the delay-`R` construction its own parameter:

\[
\varepsilon_1,\ldots,\varepsilon_R.
\]

The useful decoder can only be reached if all `R` links succeed. Consequently the unique leading useful monomial is

\[
\boxed{
L-L_0
=
-C_R
\varepsilon_1\varepsilon_2\cdots\varepsilon_R
+
\text{higher-order terms}.
}
\]

For delays 2 through 5, the exact polynomial oracle confirms:

- every coefficient of total degree below `R` is zero;
- the leading exponent is exactly `(1,1,...,1)`;
- its coefficient equals the negative of the closed-form delay-chain gain coefficient derived in `gradient_order_barrier.md`.

This makes the optimization barrier stronger than a scalar log-log slope. At the collapsed point:

- for `R >= 2`, every first derivative with respect to a missing link is zero;
- for `R >= 3`, the entire Hessian with respect to those links is zero;
- in general every derivative tensor of total order below `R` vanishes;
- the first useful local derivative is an `R`-way mixed derivative involving all required links.

A dormant scaffold prewires `R-1` downstream links and therefore removes those variables from the product, reducing the first useful derivative to first order in the remaining entrance link.

## Parameter tying can create fake cancellation

Scalar order is not invariant to reparameterization by tying independent links together.

Consider the binary first-order repeat source with switch probability `0.2`. From collapsed memory 0, introduce two independent one-step routing parameters:

```text
epsilon_0: on token 0, 0 -> memory 1
epsilon_1: on token 1, 0 -> memory 2
```

Choose decoder 1 as `(0.8,0.2)` and decoder 2 so its first-order contribution is exactly opposite. In the frozen regression witness,

\[
\ell_{(1,0)}\approx-0.0843258312,
\]

\[
\ell_{(0,1)}\approx+0.0843258312.
\]

Thus the independent gradient is nonzero.

If the two parameters are tied,

\[
\varepsilon_0=\varepsilon_1=\varepsilon,
\]

the linear terms cancel. In the exact witness the full tied scalar family is constant over the tested finite horizon, even though the untied model has two nonzero and opposite gradient components.

Therefore statements such as "the gradient order is `d`" must specify the parameterization. The independent-link multivariate polynomial is the natural representation for the finite-state transition model because each transition decision has its own parameter.

## Construction distance and monomial support

The structural construction-cost lower bound from `perturbative_construction_distance.md` has a sharper multivariate interpretation. Any monomial that changes prediction through a decoder-distinct dormant computation must contain enough perturbative routing factors to realize a source-supported path to that computation.

For a simple chain this forces the support

\[
\{1,2,\ldots,R\}.
\]

For a branching controller there can instead be several alternative minimum-degree monomials, each corresponding to a different perturbative computational route. This suggests representing local memory construction not only by a scalar order but by a **hypergraph of leading monomial supports**:

- vertices: independent missing transition parameters;
- hyperedges: supports of minimum-total-degree source-conditioned loss monomials;
- coefficient sign/magnitude: whether that route is useful, harmful, or weak.

That hypergraph is a natural next object to test against actual training bootstrap behavior.

## Claim boundary

High-order flat saddles, mixed derivatives, multiplicative parameter interactions, and multivariate perturbation expansions are established ideas. The narrower contribution here is an exactly solvable predictive-memory setting in which the missing computational links of a finite-state algorithm map directly onto the support and order of the first nonzero loss derivatives.

The delayed chain gives an exact identity between required independent links and mixed-derivative order; more general graphs require keeping source-conditioned cancellations and alternative paths explicit rather than collapsing everything to one scalar distance.
