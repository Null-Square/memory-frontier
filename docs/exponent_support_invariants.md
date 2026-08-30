# Exponent support controls quadratic balance laws

The isolated-monomial construction-flow results have exact weighted squared
balance invariants. Once several leading computations share parameters, those
pairwise invariants need not survive. The right object is the exponent-support
matrix of the polynomial.

## Polynomial gradient flow

Let

\[
L(x)=\sum_{r=1}^m c_r x^{\alpha_r},
\]

where each distinct nonconstant exponent vector
\(\alpha_r\in\mathbb N^n\) has nonzero coefficient after algebraically identical
monomials are combined.

Consider diagonal quadratic forms

\[
Q_w(x)=\sum_{i=1}^n w_i x_i^2.
\]

Under Euclidean gradient flow,

\[
\dot x_i=-\frac{\partial L}{\partial x_i}.
\]

A direct calculation gives

\[
\boxed{
\frac{d}{dt}Q_w(x)
=-2\sum_{r=1}^m
c_r x^{\alpha_r}
\langle \alpha_r,w\rangle.
}
\]

Let the exponent-support matrix be

\[
A=
\begin{bmatrix}
\alpha_1^\top\\
\vdots\\
\alpha_m^\top
\end{bmatrix}.
\]

Because distinct monomials are linearly independent as functions on the positive
orthant, the derivative above vanishes identically if and only if each monomial
coefficient vanishes separately. Therefore

\[
\boxed{
Q_w\text{ is a global diagonal quadratic invariant}
\iff
Aw=0.
}
\]

Equivalently, the space of such conservation laws is exactly

\[
\boxed{
\ker A,
}
\]

and its dimension is

\[
\boxed{
n-\operatorname{rank}(A).
}
\]

`quadratic_invariant_basis` computes this nullspace numerically from a sparse
polynomial coefficient dictionary.

## Recovery of the isolated-monomial theorem

For one monomial with exponent vector

\[
\alpha=(\alpha_1,\ldots,\alpha_n),
\]

`A` has one row, so its nullspace has dimension `n-1`. Choosing

\[
w_i=1/\alpha_i,
\qquad
w_j=-1/\alpha_j,
\]

and all other entries zero recovers

\[
\frac{x_i^2}{\alpha_i}-\frac{x_j^2}{\alpha_j}=\text{constant}.
\]

Thus the repeated-parameter weighted-balance theorem is the rank-one special
case of the support-matrix result.

## Shared routes destroy balance laws by increasing rank

Consider two beneficial leading routes

\[
L=-a\,xy-b\,yz,
\qquad a,b>0.
\]

The support matrix is

\[
A=
\begin{bmatrix}
1&1&0\\
0&1&1
\end{bmatrix},
\]

whose nullspace is spanned by

\[
(1,-1,1).
\]

Hence the coupled flow conserves exactly one independent diagonal quadratic law,

\[
\boxed{
x^2-y^2+z^2=\text{constant}.
}
\]

Add a third route `xz`. Then

\[
A=
\begin{bmatrix}
1&1&0\\
0&1&1\\
1&0&1
\end{bmatrix}
\]

has full rank. No nonzero diagonal quadratic invariant survives.

So shared computations can be classified by how their exponent support raises
rank and removes balance constraints.

## Exact finite-memory shared-route witness

The theorem is also frozen on an exact finite-memory construction rather than
only abstract monomials.

Use the generic observable order-2 Markov source from the non-delayed witness,
with next-bit probabilities

```text
suffix 00 -> 0.1
suffix 01 -> 0.8
suffix 10 -> 0.6
suffix 11 -> 0.2
```

and four memory states. From the collapsed state, one learned entrance parameter
`x` follows an observed `0` into an intermediate memory state. Two learned exits
then recognize different suffixes:

```text
0 --0,x--> 1 --0,y--> 2   decoder matches suffix 00
              \\--1,z--> 3   decoder matches suffix 01
```

The exact finite-horizon multivariate loss polynomial at horizon 12 has leading
nonconstant support

\[
\boxed{
xy,\quad xz,
}
\]

with both coefficients beneficial. Its leading support matrix is

\[
A_{\rm lead}
=
\begin{bmatrix}
1&1&0\\
1&0&1
\end{bmatrix},
\]

so the exact leading gradient flow conserves

\[
\boxed{
x^2-y^2-z^2.}
\]

This fixture is regression-tested directly from
`multivariate_controller_loss_coefficients`; the route support is not inserted by
hand.

## Interpretation

For one isolated computation, balancedness gives a nearly complete reduction of
the leading flow. For several computations, the surviving balance laws are no
longer attached to individual routes. They are determined by linear dependencies
among all exponent vectors simultaneously.

This suggests a useful structural statistic for competing algorithm formation:

\[
\boxed{
\text{balance-law dimension}
=
\text{parameter count}-\operatorname{rank}(A).
}
\]

A low-rank support retains many conserved quadratic constraints. A full-rank
support has none of this diagonal quadratic integrability left.

## Claim boundary

### Exact

- for a canonical finite polynomial, global diagonal quadratic invariants are
  exactly the nullspace of its nonconstant exponent-support matrix;
- the isolated repeated-parameter balance theorem is the one-row special case;
- adding supported monomials can only preserve or reduce the nullspace dimension;
- the frozen shared-route finite-memory witness has leading support `xy,xz` and
  therefore one exact leading quadratic invariant.

### Not established

- that every useful conservation law is diagonal quadratic;
- that full finite-horizon polynomials retain the invariants of their leading
  support away from the local regime;
- that nullspace dimension alone predicts which competing route wins;
- invariance under nonlinear reparameterization or non-Euclidean optimizers.

The exponent-support nullspace should therefore be viewed as an exact structural
invariant of polynomial Euclidean gradient flow and a candidate descriptor of
shared-route coupling, not a complete integrability theorem.
