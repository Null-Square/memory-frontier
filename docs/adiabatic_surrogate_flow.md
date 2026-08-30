# Adiabatic surrogate algorithm flow

The hard-cell stability criterion induces an exact deterministic dynamics over the finite controller space.

Assume that inside a hard cell `F` the decoder equilibrates to its exact finite-horizon Bayes readout before the hard transition table changes. Let

\[
\Delta_{mxn}=G_{mx,F(m,x)}-G_{mxn}
\]

be the intrinsic counterfactual improvement advantage after that equilibration.

Define the adiabatic successor map

\[
\Phi(F)=
\begin{cases}
F_e,& e=\arg\max_e\Delta_e,\ \Delta_e>0,\\
F,& \max_e\Delta_e\le0.
\end{cases}
\]

This is a parameterization-independent idealization of the surrogate's hard policy-improvement tendency: transition margin, learning rate, and backward temperature affect residence time but not the chosen intrinsic target-cost minimizer.

## Exact delay-2 / K=3 census

For the binary delay-2 source with switch probability `0.1`, `K=3`, and horizon `T=20`, all

\[
3^{3\times2}=729
\]

hard transition tables are enumerable. The exact adiabatic map has:

```text
fixed points                   189
nontrivial attractor cycles      8
all nontrivial cycles have length 2
starting controllers captured by 2-cycles 20
```

The four globally optimal hard controllers have loss

\[
L^*=0.507119750658\ \text{nats/token}.
\]

Only

\[
146/729\approx20.0\%
\]

of starting controllers flow to one of those global optima under the adiabatic map.

Basin accounting over all 729 starts is:

```text
end at a true hard local minimum       476
end at surrogate-stable but nonlocal   233
end in a two-cycle                      20
```

The categories above count starting controllers by their eventual attractor; the first category includes the global optima.

## Why cycles matter

The adiabatic map is not descent on the exact hard-controller loss. `G` evaluates a counterfactual target using the current cell's occupancy and decoder. After taking the edit and re-equilibrating, the new controller can prefer to reverse the same transition.

A frozen regression witness is the exact two-cycle

```text
(0,1,0,0,2,0)
        ->
(2,1,0,0,2,0)
        ->
(0,1,0,0,2,0)
```

on the delay-2 task.

Thus even an idealized exact surrogate policy-improvement procedure can cycle before optimizer momentum, stochasticity, or finite decoder equilibration are introduced.

## Interpretation

This gives a clean hierarchy of progressively richer dynamics:

1. **hard objective graph** — edges labeled by exact re-equilibrated hard loss change;
2. **adiabatic surrogate graph** — edges chosen by exact counterfactual target advantage after decoder equilibration;
3. **hybrid gradient dynamics** — continuous logits and optimizer state determine residence time and whether the adiabatic preference is reached before other quantities change.

The adiabatic graph is therefore a useful intermediate object between static local diagnostics and full optimizer trajectories. It exposes traps and cycles that are properties of the surrogate update rule itself rather than consequences of noisy training.
