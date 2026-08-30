# Hard-cell fixed points and full training trajectories

The exact hard-cell factorization lets us separate two notions of local optimality that are easy to conflate.

For a fixed hard controller `F`, first set each occupied memory state's decoder to its exact finite-horizon Bayes readout. Let

\[
G_{mxn}
\]

be the exact pre-softmax counterfactual transition derivative for routing row `(m,x)` to target memory `n`.

Define the intrinsic improvement advantage

\[
\Delta_{mxn}=G_{mx,F(m,x)}-G_{mxn}.
\]

A hard cell is **surrogate-stable after decoder equilibration** when

\[
\boxed{\Delta_{mxn}\le 0\quad\forall m,x,n.}
\]

Equivalently, every current hard target is a minimizer of its counterfactual target cost.

This condition is independent of transition-logit margin and backward temperature. Those quantities scale the softmax dynamics, but they do not change which target minimizes `G`.

## Surrogate stability is not hard local optimality

A one-edit hard local minimum asks whether replacing one transition target and then re-evaluating that new hard controller can reduce the exact finite-horizon Bayes loss.

The two notions differ because the surrogate transition derivative evaluates counterfactual target states using the **current** controller's occupancy and decoder, rather than fully re-equilibrating the edited controller.

On the binary delay-2 source with `K=3`, horizon `T=20`, all 729 hard transition tables can be enumerated exactly. The census is:

```text
surrogate-stable hard cells       189
true one-edit hard local minima    20
intersection                       12
```

So neither class contains the other.

Two regression witnesses are frozen in the test suite:

1. the fully collapsed controller is surrogate-stable but has a one-edit neighbor improving exact hard loss by more than `0.02` nats/token;
2. a separate hard one-edit local minimum is surrogate-unstable, so the STE field still tries to leave it.

These are the two local failure modes already suggested by the accessibility/alignment experiments:

\[
\text{useful but inaccessible}
\]

and

\[
\text{accessible but harmful}.
\]

## Whole-trajectory census

Using the exact hard-cell oracle, we ran joint transition/readout Adam on the same delay-2, `K=3`, `T=20` problem. Transition logits are Gaussian-initialized, decoder logits start uniform, and every gradient is an exact expected gradient with no token sampling.

For a fixed 300-seed protocol with 180 updates:

```text
global hard-oracle recovery              30.0%
mean hard-cell changes                    3.79
runs revisiting a previously seen cell   33.3%
runs containing a harmful hard edit      46.3%
failed runs ending at hard local minima  68.6%
failed runs ending surrogate-stable      82.9%
```

Extending a 120-seed subset to 1,000 updates raises global recovery to about `40.8%`; among the remaining failures, about `77.5%` finish at a true hard local minimum. A longer 80-seed, 5,000-update check puts approximately `97.8%` of failures in both the intrinsic surrogate-stable set and the hard-local-minimum set.

These long-run numbers are empirical properties of the stated optimizer/protocol, not theorems. The exact objects underneath them—the hard landscape, hard-cell flow, and stability criterion—are deterministic.

## Hybrid-system interpretation

Training is now naturally represented as

\[
(z_t,a_t,F_t),
\qquad F_t=\operatorname{argmax}z_t.
\]

Inside one hard cell, cached deterministic dynamics update `(z,a)`. At an argmax boundary, `F` changes and a new hard-cell oracle is loaded. A trajectory is therefore

\[
F_0\xrightarrow{\tau_1}F_1\xrightarrow{\tau_2}F_2\rightarrow\cdots.
\]

The emerging research decomposition is:

1. **capacity** — which hard finite-memory algorithms exist and which is globally optimal;
2. **accessibility** — which hard edits can receive a first-order gradient signal;
3. **alignment** — whether those accessible edits improve the exact hard objective;
4. **cell dynamics** — how decoder and transition logits evolve before the next discrete boundary;
5. **basin composition** — how those local flows compose into eventual global recovery or a suboptimal recurrent/fixed cell.

The next scientific target is to predict the next cell and residence time from the current cell state well enough to build a deterministic coarse-grained transition model over the 729-controller landscape.
