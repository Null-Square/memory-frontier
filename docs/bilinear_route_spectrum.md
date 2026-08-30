# Bilinear route spectrum: exact leading dynamics for coupled computations

The earlier route-race results treat isolated two-link monomials, while the
shared-route result solves the rank-one case in which several computations use
one common entrance. The natural quadratic generalization is a whole matrix of
coupled entrance/exit computations.

Let

\[
L_2(x,y)=-x^\top A y,
\]

where `x` contains entrance-side construction coordinates, `y` contains
exit-side coordinates, and `A` records the beneficial degree-two computation
coefficients.

The leading Euclidean gradient flow is

\[
\boxed{
\dot x=A y,
\qquad
\dot y=A^\top x.
}
\]

This system is exactly solvable by the singular-value decomposition of `A`.

## Exact singular-mode decomposition

Write

\[
A=U\Sigma V^\top.
\]

In modal coordinates

\[
\xi=U^\top x,
\qquad
\eta=V^\top y,
\]

every nonzero singular value `sigma_i` produces an independent two-dimensional
hyperbolic block:

\[
\dot\xi_i=\sigma_i\eta_i,
\qquad
\dot\eta_i=\sigma_i\xi_i.
\]

Hence

\[
\boxed{
\xi_i(t)
=\xi_i(0)\cosh(\sigma_i t)
+\eta_i(0)\sinh(\sigma_i t),
}
\]

\[
\boxed{
\eta_i(t)
=\eta_i(0)\cosh(\sigma_i t)
+\xi_i(0)\sinh(\sigma_i t).
}
\]

Coordinates in the left or right nullspace of `A` are frozen exactly.

`bilinear_route_spectrum` returns the full SVD, and `bilinear_route_flow`
implements this solution including rectangular nullspaces.

## Construction spectrum

The singular values have a direct optimization interpretation:

\[
\boxed{
\sigma_i=\text{leading growth rate of computation mode }i.
}
\]

Thus the quadratic computation geometry has a **route spectrum**, not merely a
set of independent path coefficients.

If the top singular mode is seeded and separated from the rest, its exponential
branch grows like `exp(sigma_1*t)` and the entrance/exit construction vectors
align with its left/right singular vectors.

This gives a precise coupled analogue of the earlier route winner:

- diagonal `A`: independent quadratic routes;
- one-row or one-column `A`: the previously solved shared-route rank-one system;
- general `A`: several coupled construction modes that are orthogonal only after
  the SVD change of coordinates.

## Exact quadratic balance laws

Every coupled singular mode conserves

\[
\boxed{
\xi_i^2-\eta_i^2.
}
\]

Summing over modes, including frozen nullspace coordinates, gives the global
balance law

\[
\boxed{
\|x\|^2-\|y\|^2=\text{constant}.
}
\]

This is the matrix-valued version of the two-link and shared-entrance
balancedness results.

These balancedness/SVD facts are classical linear-gradient-flow structure; they
are not a novelty claim here.

## Exact finite-memory full-rank witness

Use the same observable second-order Markov source as the non-delayed route
results:

\[
P(1\mid00)=0.1,
\quad
P(1\mid01)=0.8,
\quad
P(1\mid10)=0.6,
\quad
P(1\mid11)=0.2.
\]

A seven-state controller has two learned first-symbol entrances and two learned
second-symbol exits. Each exit parameter is shared across both intermediate
memories, so all four length-two suffix computations appear as mixed quadratic
terms.

The decoder memories are exact for the four suffixes:

- `00`: `(0.9,0.1)`;
- `01`: `(0.2,0.8)`;
- `10`: `(0.4,0.6)`;
- `11`: `(0.8,0.2)`.

The exact horizon-12 degree-two loss has route matrix

\[
\boxed{
A=
\begin{pmatrix}
0.17526867008023675 & 0.022945804407352086\\
0.002397084946510579 & 0.022945804407352083
\end{pmatrix}.
}
\]

It is full rank. Its exact numerical singular values are

\[
\boxed{
\sigma_1=0.17684673120575908,
\qquad
\sigma_2=0.02243003052995202.
}
\]

The ratio is about `7.884`, so the witness has a strongly separated dominant
construction mode.

The CI regression reconstructs `A` from its SVD, freezes these singular values,
checks the modal/global balance laws, verifies nullspace freezing on a
rectangular fixture, and verifies that the rank-one solver reduces exactly to
the existing shared-entrance formula.

## Full-polynomial audit

The exact SVD solution applies to the degree-two leading polynomial, not to the
whole finite-horizon loss. The witness's first corrections are degree three, so
the local rescaling results predict that the leading bilinear trajectory should
have relative corrections of order

\[
\delta^{3-2}=\delta
\]

on bounded rescaled-time windows.

`experiments/bilinear_route_spectrum.py` starts with

\[
x(0)=\delta(1,1),
\qquad
y(0)=0,
\]

and compares the exact bilinear solution with high-accuracy integration of the
**full exact horizon-12 polynomial** at physical time `t=20`.

Representative relative errors are approximately

```text
delta      relative error
0.0100     0.1741
0.0050     0.0960
0.0010     0.0209
0.0005     0.0106
```

The small-`delta` error is consistent with linear scaling. This is a deterministic
finite audit, not a global error theorem.

## Connection to near-tie route reversal

The previous near-tie result showed that a scalar route-strength advantage of
order

\[
\delta^{p-d}
\]

is not uniformly protected from degree-`p` corrections.

The bilinear spectrum suggests the matrix analogue. Away from spectral
degeneracy, a large singular gap stabilizes the leading construction subspace.
Near a repeated or nearly repeated singular value, however, higher-order terms
can compete with the leading spectral splitting and rotate which computation
mixture grows fastest.

This motivates the next adversarial target:

\[
\boxed{
\sigma_1-\sigma_2
\sim
\delta^{p-d}
}
\]

as the natural singular-gap analogue of the scalar route near-tie window.

That statement is a target for falsification/verification, not established by
this note.

## Prior-art boundary

The SVD reduction of bilinear/deep-linear gradient dynamics is classical. A key
reference is Saxe, McClelland, and Ganguli, *Exact solutions to the nonlinear
dynamics of learning in deep linear neural networks* (ICLR 2014; arXiv
1312.6120), which analyzes singular-mode learning dynamics in deep linear
networks. Modern surveys of deep-linear gradient flow likewise treat spectral
mode decompositions and balancedness as established structure.

The contribution claimed here is narrower: the exact finite-memory loss
polynomial induces a concrete computation-route matrix whose singular spectrum
can be interpreted as a **leading construction spectrum**, and that spectrum can
be placed inside the project's existing hierarchy of higher-order breaking,
finite-step breaking, and near-degeneracy margins.

## Claim boundary

### Exact / algebraic

- `dx/dt=A y`, `dy/dt=A.T x` for `L=-x.T A y`;
- exact SVD/cosh/sinh modal solution;
- singular values are modal growth rates;
- per-mode balances `xi_i^2-eta_i^2`;
- global balance `||x||^2-||y||^2`;
- left/right nullspace coordinates are frozen;
- diagonal and shared-entrance systems are special cases;
- the frozen finite-memory witness's route matrix and singular values.

### Deterministic finite audit

- full horizon-12 trajectory approaches the bilinear trajectory roughly
  linearly with initialization over the tested small-`delta` range.

### Not established

- a global singular-mode description after the quadratic local regime;
- a universal mapping from top singular vector to final hard controller;
- robustness under finite learning rate, momentum, Adam, projection, or noise;
- the proposed singular-gap near-degeneracy law, which remains the next target.
