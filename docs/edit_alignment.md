# Accessibility versus usefulness: exact edit alignment

The accessibility operators answer whether decoder symmetry breaking can create transition-gradient directions. They do **not** answer whether the corresponding hard transition changes improve the exact finite-state objective.

This note introduces a second exact linearization so those questions can be separated.

## Two maps from the same perturbation

Fix a hard controller `F`, transition logits `z` inside its current argmax cell, a decoder-symmetric base point `a0`, horizon `T`, and a one-edit alternative `F_e`.

For decoder-logit perturbation `D`, the Gradient Accessibility Operator induces a first-order STE pressure on edit `e`:

\[
\boxed{
 p_e(\epsilon)
 =
 \epsilon (P_F D)_e + O(\epsilon^2).
}
\]

Positive pressure means gradient descent favors taking the edit.

Independently, compare the actual hard controllers with the same fixed decoder perturbation. Their exact finite-horizon loss gain has expansion

\[
\boxed{
L(F,a_0+\epsilon D)-L(F_e,a_0+\epsilon D)
=
\epsilon (H_F D)_e+O(\epsilon^2).
}
\]

Positive `(H_F D)_e` means the hard edit genuinely improves the exact objective to first order.

`P_F` and `H_F` therefore answer different questions:

- `P_F`: which edit does the surrogate make accessible?
- `H_F`: which edit is actually useful to the hard predictor?

Both are computed exactly without sampling.

## Exact hard-gain operator

At the decoder-symmetric base point all memory rows share decoder distribution `q`. For hard controller `F`, let

\[
A^F_{mx}
=
\frac1T\sum_t P_F(M_t=m,X_{t+1}=x)
\]

be its exact finite-horizon memory/token mass. The decoder-logit gradient is

\[
R^F_{mx}
=
\left(\sum_y A^F_{my}\right)q_x-A^F_{mx}.
\]

For one-edit controller `F_e`, the hard-gain linear functional is

\[
\boxed{
H_e = R^F-R^{F_e}.
}
\]

So the hard-edit operator requires only exact hard occupancies; there is no STE in this object.

## Gradient-edit alignment

Collect all one-edit pressure rows into matrix `P_F` and all hard-gain rows into `H_F`. For an isotropic decoder perturbation `D`,

\[
\mathbb E[(P_FD)^\top(H_FD)]
=
\langle P_F,H_F\rangle_F.
\]

This motivates normalized **gradient-edit alignment**

\[
\boxed{
\Gamma_F
=
\frac{\langle P_F,H_F\rangle_F}
{\|P_F\|_F\|H_F\|_F}.
}
\]

The same definition can be evaluated after restricting decoder perturbations to a chosen subspace, such as token-contrast directions on behaviorally unreachable memory states.

Interpretation:

- `Gamma > 0`: accessible gradient directions tend to agree with exact improving edits;
- `Gamma < 0`: the surrogate tends to expose directions opposite to exact hard improvement;
- high accessibility with poor alignment can produce fast but harmful hard transitions;
- high hard-edit gain with poor accessibility produces useful moves that gradient descent cannot readily reach.

## Delay-3 population result

On the binary delay-3 source, use `K=4`, horizon 20, transition margin `0.7`, backward temperature `0.8`, and restrict decoder perturbations to binary contrasts on the three unreachable decoder rows. Every sampled controller has the same reachable collapsed row and therefore the same current forward function.

Across 180 random unreachable topologies, restricted alignment had:

```text
median Gamma      0.453
10th percentile  -0.0705
90th percentile   0.848
negative Gamma    17.2% of topologies
```

Across 180 topologies x 12 decoder directions = 2,160 runs:

```text
first-escape rate                              0.611
useful first escape among escapes              0.677
median edgewise sign fidelity                  0.667
```

The edit selected by maximum linearized GAO pressure was labeled useful/harmful by `H_F` with **98.3% agreement** with the actual finite-`epsilon` hard-loss change among runs that escaped.

The selected linearized hard-gain magnitude correlated with the actual finite-perturbation first-escape gain at

\[
r\approx0.979.
\]

At topology level, `Gamma_F` correlated about `0.736` with the fraction of useful escapes among the topology's escaping directions.

## Delay-depth robustness

Independent topology/direction samples give:

| Delay | K | Runs | Useful among escapes | Linear sign agreement | corr(linear gain, actual gain) | corr(Gamma, useful fraction) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 360 | 0.744 | 0.978 | 0.960 | 0.712 |
| 3 | 4 | 360 | 0.674 | 0.991 | 0.998 | 0.756 |
| 4 | 5 | 360 | 0.879 | 0.965 | 0.979 | 0.530 |

## Important negative result

Accessibility alone is insufficient.

In a separate delay-3 sample of 1,200 runs:

- `68.4%` of first escapes improved exact hard loss;
- `31.6%` worsened it;
- among sampled non-escaping runs, `92.9%` still had at least one exact improving reachable-row one-edit available.

Thus both failure modes occur:

\[
\text{accessible but harmful}
\]

and

\[
\text{useful but inaccessible}.
\]

This is why accessibility and alignment should remain separate axes rather than being collapsed into one notion of optimization difficulty.

## Three-layer research picture

The project now has three exact levels:

1. **Capacity** — what finite-memory algorithm globally exists?
2. **Accessibility** — which alternative computations can the local surrogate gradient reach from this parameterization?
3. **Alignment** — among accessible one-edit directions, which agree with actual hard-objective improvement?

All three are source-conditioned, finite, and computable before stochastic neural training.

The claim boundary remains narrow: mixed derivatives, discrete local search, policy gradients, and initialization sensitivity are established ideas. The distinctive object here is their exact combination in a hard-forward/soft-backward learned automaton where forward behavior is held fixed while unreachable topology changes both accessibility and alignment.
