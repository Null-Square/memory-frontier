# Readout initialization is an optimization resource

A hard categorical memory state can be unused by the current controller. Its
readout is then **not identified by data**. Any value assigned to that unused
readout is therefore an extra optimization convention, not part of the hard
finite-state prediction problem.

This matters for straight-through training because the decoder values determine
which transition edits receive gradient pressure.

## Collapsed-controller stationary point

Let the source begin in stationarity and let

\[
p_X(x)=P(X_{t+1}=x)
\]

be its stationary one-token marginal. Consider a deterministic controller whose
memory is collapsed to one state for the entire scored horizon. Initialize every
memory state's readout to the same source marginal:

\[
q_m(x)=p_X(x) \quad \forall m.
\]

Then the joint trainable system is stationary under the exact-distribution
straight-through objective:

1. the occupied readout is already the optimal predictor available to a memory
   state that carries no source information;
2. unused readouts receive no data gradient; and
3. because every readout is identical, redirecting probability mass to another
   memory target has no first-order effect on prediction loss, so the transition
   gradient is zero.

The repository freezes this fact as a numerical regression using double
precision and the exact expected finite-horizon objective.

## Balanced symmetry-trap witness

`balanced_markov_symmetry_trap()` removes the accidental distinction between
"uniform" and "source-marginal" initialization. It is the symmetric binary
Markov source

```text
state 0: P(0)=0.9, P(1)=0.1
state 1: P(0)=0.1, P(1)=0.9
next source state = emitted token
```

Its stationary source-state and token marginals are both exactly `(0.5, 0.5)`.
A one-bit last-symbol controller is strongly predictive, but a collapsed memory
controller with zero decoder logits has both decoders equal to the exact source
marginal. Therefore the usual zero-logit decoder initialization is itself the
stationary-point construction above.

At `K=2,T=32` the frozen tests verify:

- collapsed-controller loss is exactly `log(2)` up to numerical tolerance;
- the exact hard-controller optimum is below `0.36` nats/token;
- both transition and readout gradient norms at the collapsed zero-logit point
  are below `1e-12`.

Thus a strictly better finite-memory algorithm can exist while first-order
straight-through training has no direction at all from a natural symmetric
initialization.

## Why uniform initialization is not neutral

If all readouts begin uniform while the source marginal is non-uniform, the
occupied readout moves toward the source marginal while an unused readout stays
at its arbitrary initial value. The resulting decoder disagreement creates
transition pressure that can revive the unused state.

Thus a nominally innocuous decoder initialization can act as an **exploration
prior** for the discrete state transition learner. The balanced witness shows the
complement: when uniform happens to equal the source marginal, that accidental
exploration mechanism disappears completely.

`readout_prior.py` therefore makes the convention explicit:

- `uniform`
- `source_marginal`
- `random`

The original learner remains unchanged for historical comparison; new ablations
should name the readout initialization explicitly.

## First-order escape from the symmetric point

Perturb the decoder logits by a small asymmetric amount

\[
r_m = \log p_X + \varepsilon u_m.
\]

For a generic perturbation direction the decoder probabilities differ by
`O(epsilon)`. With the hard transition table and its straight-through Jacobian
held fixed, the induced transition gradient is therefore also generically
`O(epsilon)`.

The balanced witness freezes one concrete check: increasing a fixed decoder
perturbation from `1e-5` to `1e-4` increases the transition-gradient norm by a
factor between `9.99` and `10.01`.

This falsifies an earlier scratch hypothesis that escape might begin only at
second order. The apparent finite perturbation threshold seen in finite-step
optimizer sweeps is therefore a training-time/logit-margin phenomenon, not a
mathematical threshold in the gradient field.

## Interpretation boundary

Unused or dead discrete codes are a known issue in vector quantization, and many
methods use reinitialization, stochastic assignments, or codebook updates to
restore utilization. Recent VQ work continues to identify inactive-code feedback
loops and initialization/distribution matching as important causes of collapse.
The broader dead-code phenomenon is therefore **not** a novelty claim here.

The narrower research opportunity is that the finite-memory prediction objective
is exactly enumerable. We can separate:

1. whether a better hard algorithm exists,
2. how far away it is in the exact discrete landscape,
3. whether the current continuous surrogate supplies a useful direction, and
4. whether an otherwise unidentified decoder prior supplies the symmetry
   breaking required to begin learning.

That separation is normally unavailable in large discrete-latent models.

## Reproduce the pilot ablation

```bash
python -m pip install -e '.[dev,optimization]'
python examples/readout_prior_report.py
```

Do not treat a recovery-rate difference as a theorem. The stationary-point
construction and the first-order local scaling are the controlled claims;
recovery rates remain optimizer- and protocol-dependent empirical measurements.
