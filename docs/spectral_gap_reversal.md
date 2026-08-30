# Spectral-gap collapse makes leading construction modes nonuniform

The bilinear route spectrum gives exact leading growth modes for

\[
L_2=-x^\top A y.
\]

When the largest singular value is well separated, the associated singular
vectors are a natural local prediction for the fastest coupled construction
mode. This note asks the adversarial question: **how small can that singular gap
be before the first higher-order loss terms alter the dominant local mode?**

The answer mirrors the scalar near-tie result. A leading spectral gap must be
compared with the same higher-order scale `delta**(p-d)` that controls route
near-ties and approximate balance-law breaking.

## Hessian scaling

Let

\[
L=L_d+L_p+\cdots,
\qquad p>d,
\]

and evaluate local geometry at

\[
x=\delta y.
\]

A degree-`r` homogeneous term contributes to the Hessian at scale

\[
\nabla^2L_r(\delta y)
=
\delta^{r-2}\nabla^2L_r(y).
\]

After factoring out the leading Hessian scale `delta**(d-2)`, the first
higher-order correction is therefore

\[
\boxed{
O(\delta^{p-d}).
}
\]

For gradient flow the local linearized growth operator is

\[
J=-\nabla^2L.
\]

If the relevant normalized leading eigengap is `gamma`, standard symmetric
subspace perturbation theory gives stability proportional to correction size
divided by the gap. Therefore a leading mode is uniformly protected only when

\[
\boxed{
\gamma\gg\delta^{p-d}.
}
\]

When

\[
\gamma=O(\delta^{p-d}),
\]

the higher-order Hessian must be included in the leading local mode-selection
problem.

This scaling is standard matrix perturbation theory, not a novelty claim. The
classical Davis--Kahan sin-theta theorem controls invariant-subspace rotation of
symmetric operators by perturbation size relative to spectral separation, and
Wedin's 1972 theorem gives the corresponding singular-subspace bounds. The
finite-memory contribution here is an exact construction-polynomial witness in
which the predicted instability actually reverses which computation mode is
dominant.

## Bilinear growth modes

For

\[
L_2=-x^\top A y,
\]

the gradient-flow Jacobian is

\[
J_2=
\begin{pmatrix}
0&A\\
A^\top&0
\end{pmatrix}.
\]

If

\[
A=U\Sigma V^\top,
\]

then every singular triplet gives the positive-growth eigenmode

\[
\boxed{
g_i=\frac1{\sqrt2}\binom{u_i}{v_i},
\qquad
J_2g_i=\sigma_i g_i.
}
\]

`bilinear_positive_growth_modes` exposes these vectors directly, and
`project_symmetric_operator` projects any symmetric local correction into such a
mode subspace.

## Exact repeated-mode finite-memory witness

Start from the full-rank two-entrance/two-exit controller in
`bilinear_route_spectrum.md`, but make the off-diagonal suffix decoders uniform:

- suffix `01`: `(0.5,0.5)`;
- suffix `10`: `(0.5,0.5)`.

Keep suffix `11` exact at `(0.8,0.2)`. Tune the suffix-`00` decoder probability
of symbol `1` to

\[
q_*=0.46869740167476925\ldots,
\]

the same useful-side coefficient tie identified in the scalar near-tie result.

The exact degree-two route matrix becomes

\[
\boxed{
A_0=bI_2,
\qquad
b=0.02294580440735208\ldots.
}
\]

Thus the two positive bilinear growth modes

\[
g_1=\frac1{\sqrt2}(1,0,1,0)^\top,
\qquad
g_2=\frac1{\sqrt2}(0,1,0,1)^\top
\]

are exactly degenerate at growth rate `b`.

## Cubic correction inside the degenerate subspace

Evaluate the cubic gradient-flow Jacobian on the entrance ray

\[
(x_1,x_2,y_1,y_2)=(1,1,0,0).
\]

Projecting the exact degree-three correction into
`span(g_1,g_2)` gives

\[
\boxed{
K
=
b
\begin{pmatrix}
-1.71&-0.405\\
-0.405&-1.08
\end{pmatrix}.
}
\]

The correction is not proportional to the identity. Therefore the full loss
already distinguishes orientations that the quadratic construction spectrum
regards as exactly tied.

At the exact tie, the less-suppressed first-order mode is mostly `g_2`, with
absolute modal weights approximately

\[
(0.439,0.898).
\]

So the cubic finite-memory geometry resolves the quadratic degeneracy in a
specific computation direction.

## Open an O(delta) quadratic singular gap

Now tune

\[
q_{00}=q_*-\kappa\delta.
\]

The first diagonal route strength becomes

\[
a(q_{00})
=
b+c\kappa\delta+O(\delta^2),
\]

where

\[
c=-a'(q_*)=0.7050441225357232\ldots.
\]

For every fixed `kappa>0` and sufficiently small `delta`, the **quadratic** route
matrix therefore has

\[
\sigma_1>\sigma_2
\]

and its unique leading growth mode is `g_1`.

But in the repeated-mode subspace, the first-order full local operator is

\[
\boxed{
K_\kappa
=
K+
\begin{pmatrix}
c\kappa&0\\
0&0
\end{pmatrix}.
}
\]

The dominant eigenvector has equal absolute weight on `g_1` and `g_2` when its
diagonal entries tie. This gives the exact asymptotic critical constant

\[
\boxed{
\kappa_{\rm spec}
=
\frac{0.63b}{c}
=
0.020503478171891763\ldots.
}
\]

Therefore, for

\[
0<\kappa<\kappa_{\rm spec},
\]

the quadratic SVD says the first mode is uniquely dominant, but the full local
Hessian's dominant growth eigenvector remains weighted more strongly toward the
**second** canonical computation mode.

That is the matrix-valued analogue of the scalar route near-tie reversal.

## CI-frozen reversal

The regression uses

\[
\kappa=0.018
\]

and

\[
\delta\in\{0.02,0.01,0.005,0.0025\}.
\]

For every case:

1. the exact degree-two route matrix is diagonal with first entry larger than
   the second;
2. its singular values satisfy `sigma_1>sigma_2`, so the leading bilinear mode is
   `g_1`;
3. the dominant eigenvector of the **full exact horizon-12** gradient-flow
   Jacobian `-H_L(delta,delta,0,0)` has larger absolute projection on `g_2` than
   on `g_1`.

Using `kappa=0.025`, above the asymptotic boundary, reverses that inequality and
recovers the quadratic leading mode over the same tested scales.

No ODE integration is needed for these CI claims.

## Finite-delta boundary

`experiments/spectral_gap_reversal.py` numerically locates the value of `kappa`
at which the two canonical modal projections of the full top growth eigenvector
are equal. Representative values are

```text
delta      critical kappa
0.040      0.02163
0.020      0.02106
0.010      0.02078
```

and continue toward

\[
0.02050347817\ldots
\]

as `delta` decreases.

The finite values include quartic and higher corrections; the limiting constant
comes from the exact cubic projected operator.

## Interpretation

The project now has two closely related confidence-margin laws.

### Scalar route ordering

A leading route coefficient gap is not uniformly decisive when

\[
\Delta a=O(\delta^{p-d}).
\]

### Coupled construction-mode ordering

A leading normalized spectral gap is not uniformly decisive when

\[
\boxed{
\Delta\sigma=O(\delta^{p-d}).
}
\]

In the second regime, it is not enough to report the top singular vector of the
leading computation matrix. The first higher-order Hessian projected into the
nearly degenerate singular subspace is part of the leading local description.

This suggests a more robust local diagnostic:

\[
\boxed{
\text{leading construction spectrum}
+
\text{spectral gap / higher-order scale}
+
\text{projected correction in near-degenerate subspaces}.
}
\]

## Prior-art boundary

The mathematical sensitivity statement belongs to classical perturbation theory:

- Davis and Kahan (1970), *The Rotation of Eigenvectors by a Perturbation. III*,
  SIAM Journal on Numerical Analysis, DOI `10.1137/0707001`;
- Per-Ake Wedin (1972), *Perturbation bounds in connection with singular value
  decomposition*, BIT 12, 99--111, DOI `10.1007/BF01932678`.

Likewise, singular-mode learning dynamics for linear networks are established,
for example in Saxe, McClelland, and Ganguli (2014), *Exact solutions to the
nonlinear dynamics of learning in deep linear neural networks*.

The claim here is narrower: the exact finite-memory computation polynomial
produces a repeated leading construction spectrum whose first higher-order terms
resolve the degeneracy and can overturn the uniquely top mode after an
`O(delta**(p-d))` spectral splitting.

## Claim boundary

### Exact / algebraic

- Hessian degree scaling `delta**(r-2)`;
- positive bilinear growth modes `[u_i;v_i]/sqrt(2)`;
- the exact repeated finite-memory route matrix `A_0=b I`;
- the exact projected cubic correction matrix shown above;
- the first-order reduced operator `K_kappa`;
- the asymptotic critical constant
  `kappa_spec=0.020503478171891763...`.

### Exact finite regressions

- below-boundary `kappa=0.018`: leading SVD favors `g_1`, full local Hessian
  favors `g_2` over the tested initialization scales;
- above-boundary `kappa=0.025`: full local Hessian recovers `g_1`.

### Not established

- that the locally dominant Hessian mode always determines the first hard
  computation built;
- a global trajectory theorem through nonlocal thresholds;
- the same constants under finite-step GD, momentum, Adam, projection, or noise;
- coordinate-invariance of the numerical mode basis or critical coefficient.
