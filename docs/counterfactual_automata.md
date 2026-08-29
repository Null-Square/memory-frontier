# Counterfactual automata and gradient memory

A deterministic automaton's unreachable states are behaviorally irrelevant: they can be removed without changing any trajectory from the initial state. For a hard-forward straight-through recurrent learner, however, unreachable states can still be **gradient-relevant**.

This note records an exact witness where two hard controllers have identical forward behavior and identical loss for every sequence from the reset state, yet produce different gradients because their unreachable-state transitions differ.

## Forward-equivalent hard controllers

Use two memory states and reset in state 0. Let the reachable row be absorbing:

```text
F(0,0)=0
F(0,1)=0
```

Compare two choices for unreachable state 1.

Fully collapsed:

```text
F(1,0)=0
F(1,1)=0
```

Counterfactual self-loop:

```text
F(1,0)=1
F(1,1)=1
```

From reset state 0, state 1 is never reached in either machine. Therefore the two hard controllers compute exactly the same forward function and have identical expected prediction loss.

Yet a straight-through derivative can infinitesimally move probability from state 0 toward state 1. What happens to that perturbation at later time steps is determined by the **hard transition rules of state 1**, even though those transitions are never exercised by the forward trajectory.

## Lagged predictive operators

For a stationary token process define

\[
Q_\ell(x,y)=P(X_{t+\ell}=y\mid X_t=x).
\]

Let the collapsed decoder predict stationary token marginal `mu`, and let decoder 1 have logits `log(mu)+d`. Define

\[
\psi_\mu(d)=\log\mathbb E_{Y\sim\mu}[e^{d_Y}].
\]

If unreachable state 1 is a self-loop, a derivative-induced perturbation into it persists for every later prediction. For a horizon of `T` predictions, the exact transition pressure from reachable state 0 toward state 1 is

\[
\boxed{
p_x=J\mu_x\sum_{\ell=1}^{T-1}
\frac{T-\ell}{T}
\left((Q_\ell d)_x-\psi_\mu(d)\right),
}
\]

where `J>0` is the local softmax-Jacobian factor determined by the transition-logit margin, memory cardinality, and backward temperature.

After stationary rescaling and centering,

\[
\boxed{
\Pi_\mu D_\mu^{-1}p
=J\sum_{\ell=1}^{T-1}\frac{T-\ell}{T}Q_\ell\Pi_\mu d.
}
\]

By contrast, when state 1 also collapses immediately to state 0, the counterfactual perturbation is erased after one update and only `Q_1` remains. This recovers the higher-order blindness theorem.

Thus **unreachable hard-state structure determines the temporal depth of the gradient's predictive operator**.

## Delayed-copy witness

For the binary delay-2 source with repeat probability `0.9`, adjacent tokens are independent:

\[
Q_1=
\begin{pmatrix}
1/2&1/2\\
1/2&1/2
\end{pmatrix}.
\]

But even lags carry the hidden delayed dependence. Let

\[
R=
\begin{pmatrix}
0.9&0.1\\
0.1&0.9
\end{pmatrix}.
\]

Then

\[
Q_{2k}=R^k,
\qquad
Q_{2k+1}=Q_1.
\]

With decoder contrast `d=(0.4,-0.4)`, horizon `T=20`, transition margin `0.7`, and backward temperature `0.8`, the two forward-equivalent hard tables give qualitatively different pressure:

```text
fully collapsed unreachable row:  (-0.0192, -0.0192)
self-loop unreachable row:          (+0.0377, -0.4222)
```

The first machine cannot see the delayed predictive structure and repels the unused state for both tokens. The second machine receives a positive routing signal for one token because its counterfactual state preserves the derivative long enough to expose the lag-2 dependency.

The regression test computes these pressures through PyTorch autograd and independently through the exact lag-operator formula.

## A particularly strange property

The unreachable state's **own transition parameters receive zero direct gradient**, because the forward trajectory never occupies that state. Nevertheless, its frozen hard argmax transitions alter gradients on the reachable state's parameters.

So initialization can install a behaviorally invisible counterfactual automaton that acts as a scaffold for later learning.

This suggests a sharper distinction than ordinary parameter initialization sensitivity:

- **forward computation:** the automaton actually executed from reset;
- **counterfactual computation:** the automaton an infinitesimal surrogate perturbation would execute;
- **gradient accessibility:** predictive information exposed by that counterfactual computation.

Two models can be identical in the first sense and different in the latter two.

## Research implication

Automata theory normally removes unreachable states because they cannot affect accepted behavior. In a hard-forward differentiable parameterization, removing or changing those same states can alter the training vector field without changing the current model function.

This phenomenon should not be overgeneralized to all recurrent neural networks. It is an exact property of the hard-forward/straight-through memory parameterization studied here. The next question is whether analogous behavior appears in richer discrete-state or sparsely gated recurrent architectures, where inactive components may likewise carry counterfactual gradient pathways.
