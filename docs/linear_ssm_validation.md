# Differentiable linear-state validation of construction order

The exact finite-memory results are deliberately combinatorial. This note asks a
smaller external-validity question: does the same accessibility-order mechanism
appear in a conventional smooth recurrent state model where gradients are ordinary
autograd derivatives rather than a finite-controller polynomial oracle?

The answer is yes for a minimal linear state-space delay line.

## Model

Use a scalar input and a depth-`R` recurrent state

\[
h_{t+1,0}=w_0 x_t,
\qquad
h_{t+1,i}=w_i h_{t,i-1},\quad i=1,\ldots,R-1.
\]

The prediction is the final state coordinate. After the warm-up period,

\[
\hat x_{t-R+1}
=
\left(\prod_{i=0}^{R-1}w_i\right)x_{t-R+1}.
\]

For binary inputs `x_t in {-1,+1}` and delayed squared-error prediction,

\[
L(w)
=
\frac12\left(\prod_i w_i-1\right)^2.
\]

Thus the model is simultaneously a genuine recurrent delay system and an exactly
solvable smooth objective.

## Forward-equivalent dormant initializations

Fix `R=5`. For each `d in {1,...,5}`, initialize

\[
w_0=\cdots=w_{d-1}=0,
\qquad
w_d=\cdots=w_4=1.
\]

Every weight remains trainable. These are different parameter points in the same
architecture, but because `w_0=0` they all implement exactly the same current
predictor:

\[
\boxed{\hat x_t=0\quad\text{for every input sequence}.}
\]

The downstream unit-valued transitions are behaviorally dormant at initialization.
They matter only after the missing upstream chain begins to be constructed.

This is the continuous-state analogue of dormant prewiring in the finite-memory
controllers. It is not an application of the finite-state construction-order
theorem; it is an independent smooth recurrent witness.

## Exact accessibility order

Move away from one of the base points along the equal missing-link ray

\[
w_0=\cdots=w_{d-1}=\delta,
\qquad
w_d=\cdots=w_4=1.
\]

Then the recurrent gain is

\[
g(\delta)=\delta^d,
\]

and the improvement over the zero predictor is exactly

\[
\boxed{
L(0)-L(\delta)
=\delta^d-\frac12\delta^{2d}.
}
\]

Therefore the first useful scalar loss degree is exactly `d`.

Ordinary PyTorch autograd also gives the missing-link gradient norm

\[
\boxed{
\|\nabla_{w_{<d}}L\|
=\sqrt d\,(1-\delta^d)\delta^{d-1}.
}
\]

Hence

\[
L(0)-L(\delta)=\Theta(\delta^d),
\qquad
\|\nabla_{w_{<d}}L\|=\Theta(\delta^{d-1}).
\]

At exact zero initialization only the first-order case can move. Every case
`d>=2` has zero gradient on all missing links.

## Autograd regression

`tests/test_linear_ssm_validation.py` simulates the recurrent dynamics directly,
rather than replacing them by the product formula. It verifies:

1. all five dormant-prewire base points produce the identical zero prediction
   sequence and loss;
2. recurrent autograd matches the exact loss-improvement and gradient-norm
   formulas above;
3. log-log slopes recover construction orders one through five;
4. exact zero initialization moves only the first-order case.

The wider outside-CI audit in `experiments/linear_ssm_validation.py` uses scales

\[
0.005,\ 0.0075,\ 0.01125,\ 0.016875,\ 0.0253125.
\]

Its reference slopes are:

| `d` | loss-improvement slope | missing-gradient slope | prediction |
|---:|---:|---:|---:|
| 1 | 0.993789 | -0.012515 | (1, 0) |
| 2 | 1.999820 | 0.999640 | (2, 1) |
| 3 | 2.999995 | 1.999991 | (3, 2) |
| 4 | 4.000000 | 3.000000 | (4, 3) |
| 5 | 5.000000 | 4.000000 | (5, 4) |

The small first-order gradient-slope deviation comes from the finite factor
`1-delta`; it tends to zero as the fitting window approaches the origin.

## Fixed-step SGD audit

The same experiment also trains all five weights with ordinary SGD from missing
links initialized to `0.05`, prewired links initialized to `1`, learning rate
`0.05`, and records the first step where the effective delayed-memory gain

\[
\prod_i w_i
\]

reaches `0.2`.

The reference run gives:

| construction order | threshold step |
|---:|---:|
| 1 | 4 |
| 2 | 43 |
| 3 | 361 |
| 4 | 3965 |
| 5 | 53327 |

These iteration counts are optimizer- and threshold-specific evidence, not a new
universal asymptotic theorem. Their role is simply to show that the local order
difference is not erased immediately by ordinary finite-step optimization.

## What this adds

This witness removes one possible objection to the finite-memory story: the order
spectrum is not dependent on discrete controller states, hard routing, a
straight-through estimator, or the exact finite-state occupancy oracle.

A standard smooth linear recurrent model can have:

- the same architecture;
- the same current input-output function;
- all parameters trainable;
- different behaviorally dormant downstream initialization;
- and different first useful loss/gradient order.

So the project-level interpretation now has evidence in two distinct model
classes:

\[
\boxed{
\text{dormant computational structure can change gradient accessibility without
changing the current predictor.}
}
\]

## Claim boundary

The multiplication-chain mathematics is elementary and closely related to deep
linear network dynamics; it is not claimed as a new general optimization result.
Likewise, same-function embeddings, dormant-neuron activation, and high-order dead
directions have substantial prior art.

The purpose of this validation is narrower: it shows that the finite-memory
construction-order mechanism survives in a conventional differentiable memory
system. The paper's main theorem remains the source-aware finite-memory
support/operator/loss factorization and the exact dormant-topology intervention
inside one forward-equivalence class.
