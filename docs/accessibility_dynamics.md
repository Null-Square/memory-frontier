# Intrinsic accessibility and first-escape dynamics

The Gradient Accessibility Operator (GAO) describes the first-order map from decoder symmetry breaking to hard-transition STE gradients. This note separates the intrinsic hard-controller part of that map from the transition-softmax parameterization, then tests whether the resulting local field predicts an actual hard behavior change.

## Counterfactual Accessibility Operator

Fix a deterministic hard transition table `F`, a common decoder-logit row `a0` shared by every memory state, and a finite prediction horizon `T`. Let

\[
G_{mxn}=\frac{\partial L}{\partial H_{mxn}}
\]

be the exact counterfactual transition-tensor derivative from the value-gradient oracle, where `H[m,x,n]` temporarily treats routing into target memory `n` as a continuous weight while the forward pass remains at the hard table.

Perturb decoder logits by a direction `D`. A target-wise constant value inside one `(memory, token)` transition row cannot affect any softmax logit gradient. Let `Pi_K` remove that constant target mode. Define the **Counterfactual Accessibility Operator**

\[
\boxed{
\mathcal C_F
=
\Pi_K\frac{\partial G}{\partial a}.
}
\]

Unlike the GAO, `C_F` does not depend on transition-logit margin or backward temperature. It is intrinsic to the current hard topology, source, decoder base distribution, reset convention, and horizon.

## Exact factorization

For transition logits `z` and backward temperature `tau`, let `J_softmax(z/tau)` denote the block-diagonal Jacobian of every transition-softmax row. The GAO factors as

\[
\boxed{
\mathcal A_F
=
J_{\mathrm{softmax}}(z/\tau)\mathcal C_F.
}
\]

The equality is exact inside the current hard argmax cell.

For any finite logits and `tau>0`, each softmax Jacobian has kernel exactly equal to the constant target direction and is nonsingular on the target-contrast quotient. Since `C_F` has already projected out the constant target mode,

\[
\boxed{
\operatorname{rank}\mathcal A_F
=
\operatorname{rank}\mathcal C_F.
}
\]

Thus accessibility **rank** is invariant to finite margin and temperature, although singular values and therefore gradient strength are not.

A scratch stress test on 120 random delay-3 forward-equivalent topologies and four very different `(margin, temperature)` pairs gave 0 rank mismatches in 480 comparisons. A direct matrix factorization check agreed to approximately `2.8e-17` absolute error.

## Rank bound

Decoder logits have `K` row-wise constant gauge directions. In addition, applying the same token-distribution perturbation to every memory decoder preserves decoder equality and cannot create a memory-specializing transition gradient; this contributes another `|Sigma|-1` null directions after row gauges are removed. Therefore

\[
\boxed{
\operatorname{rank}\mathcal C_F
=
\operatorname{rank}\mathcal A_F
\le
(K-1)(|\Sigma|-1).
}
\]

For binary tokens and `K=4`, the maximum nontrivial rank is 3, exactly matching the maximum observed in the exhaustive 4,096-topology scan.

## From an infinitesimal operator to a hard transition

Use the binary delay-3 source with `rho=0.1`, `K=4`, horizon 20, transition margin `a=0.7`, and backward temperature `0.8`. Fix reachable transition row 0 to the collapsed target 0. Randomize only the three unreachable hard rows, so every initialization has exactly the same forward trajectory and prediction loss before learning.

For each topology, restrict decoder perturbations to normalized binary token contrasts on unreachable decoder rows. For a unit coefficient vector `c`, let `D(c)` be the resulting decoder-logit direction, and define the directional GAO response

\[
R(c)=\mathcal A_F D(c).
\]

The most favorable first-order pressure away from the currently selected reachable target is

\[
p_{\mathcal A}(c)
=
\max_{x,j\ne0}
\left(R_{0x0}(c)-R_{0xj}(c)\right).
\]

Initialize the fixed decoder perturbation at magnitude `epsilon` and train transition logits only with exact expected STE gradients. Until the first reachable hard transition changes, the hard table and decoder are fixed, so the counterfactual transition-tensor derivative `G` is constant; only the transition-softmax Jacobian evolves.

For small perturbations the initial margin changes approximately by `eta * epsilon * p_A` per step. This predicts

\[
\boxed{
N_{\mathrm{escape}}
\approx
\frac{a}{\eta\epsilon p_{\mathcal A}}
}
\]

when `p_A>0`.

## Delay-3 population experiment

The reproducible experiment samples 320 distinct forward-equivalent unreachable topologies with seed `260830`, then 8 independent normalized decoder directions per topology, for 2,560 runs total. Parameters are

```text
rho             = 0.1
horizon         = 20
margin          = 0.7
temperature     = 0.8
epsilon         = 0.03
learning rate   = 5.0
escape horizon  = 120 updates
```

The independent scratch verification produced:

```text
escape rate                                      0.61836
corr(GAO pressure, reciprocal censored escape)   0.96203
corr(GAO pressure, escape indicator)              0.75439
corr(linear GAO pressure, finite-epsilon pressure) 0.99963
corr(restricted GAO Frobenius, reciprocal escape) 0.54211
```

The rank-conditioned escape rates were:

| Restricted rank | Runs | Escape rate | Median escape step among successes |
|---:|---:|---:|---:|
| 0 | 24 | 0.000 | — |
| 1 | 392 | 0.500 | 49.5 |
| 2 | 856 | 0.592 | 50.0 |
| 3 | 1288 | 0.683 | 48.0 |

Rank is therefore only a coarse structural statistic. The **directional pressure** generated by the actual symmetry-breaking direction is much more predictive of escape time.

For isotropic Gaussian coefficients in the restricted decoder-contrast coordinates, linear algebra gives the exact first-order energy identity

\[
\mathbb E\|\mathcal A_F D\|^2
=
\epsilon^2\|B_F\|_F^2,
\]

where `B_F` is the GAO restricted to those contrast coordinates. A Monte Carlo check across sampled topologies gave empirical/theoretical energy ratio `1.002 +/- 0.010`.

## Scale test

A second matched experiment changes decoder perturbation size while keeping `eta * epsilon = 0.15` fixed:

```text
epsilon = 0.01, eta = 15.0
epsilon = 0.03, eta = 5.0
epsilon = 0.09, eta = 1.666666...
```

Across 120 topologies x 4 directions, the two smaller perturbations agreed on escape/no-escape in `99.8%` of matched cases and differed by a median of only one update when both escaped.

The median ratios of actual escape step to the first-order prediction were approximately

```text
epsilon 0.01: 0.966
epsilon 0.03: 0.984
epsilon 0.09: 1.044
```

The blind rank-0 topology has transition-gradient norm scaling as `epsilon^2` (measured log-log slope `1.99999`), while a rank-positive topology scales as `epsilon` (slope `1.00023`). This is the operational signature of first-order accessibility versus higher-order-only escape.

## Delay-depth robustness

Using the same fixed-perturbation transition-only protocol on independently sampled forward-equivalent topologies gives:

| Delay | K | Runs | Escape rate | corr(pressure, reciprocal censored escape) | Median actual/predicted escape ratio |
|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 350 | 0.749 | 0.977 | 0.932 |
| 3 | 4 | 350 | 0.680 | 0.958 | 1.004 |
| 4 | 5 | 350 | 0.380 | 0.902 | 1.076 |

Longer delayed dependencies become substantially harder to escape within the fixed 120-step budget, but the directional accessibility pressure remains strongly predictive.

## Interpretation and claim boundary

The result should not be framed as discovery of mixed derivatives, finite-state policy gradients, automata spectra, or generic initialization sensitivity. Those ingredients all have established literatures.

The narrower object is an exactly solvable hard-forward / soft-backward recurrent memory learner in which:

1. current model behavior can be held exactly fixed;
2. behaviorally unreachable hard-state topology changes `C_F` and `A_F`;
3. accessibility rank is intrinsic to that hard counterfactual topology rather than transition-logit margin or temperature;
4. the directional accessibility response quantitatively predicts the first hard algorithmic transition under exact gradient descent.

This moves the project from post-hoc optimization diagnostics toward a preregisterable theory: compute capacity, counterfactual accessibility, and a predicted escape timescale before running the optimizer.

## Reproduction

The population experiment is implemented in `experiments/gao_escape_prediction.py` and is intentionally excluded from CI. The primary protocol is

```bash
python experiments/gao_escape_prediction.py \
  --delay 3 \
  --topologies 320 \
  --directions 8 \
  --seed 260830
```

Delay-depth robustness can be rerun by changing `--delay` and the sample count. The experiment uses only exact finite-state dynamic programs; it does not sample token sequences or call an external model.
